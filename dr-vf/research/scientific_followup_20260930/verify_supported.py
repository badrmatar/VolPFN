"""Run only the supported follow-up checks; never rewrite previous audit receipts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/outputs/scientific_followup_20260930/support"
SUITES = {
    "attribution_launcher": "research.scientific_followup_20260930.test_attribution_launcher",
    "gp_likelihood": "research.scientific_followup_20260930.gp.test_likelihood_v1",
    "optimizer_repair": "research.scientific_followup_20260930.test_optimizer_repair",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    for name, module in SUITES.items():
        command = [sys.executable, "-m", "unittest", module, "-v"]
        start = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        log = OUT / (name + "_tests.log")
        log.write_text(process.stdout + process.stderr)
        row = dict(suite=name, command=command, returncode=process.returncode,
                   seconds=time.perf_counter() - start, log=str(log))
        records.append(row)
        print(json.dumps(row), flush=True)
    # Resolve the historical import through its supported module without
    # executing a new attribution scientific fit.
    command = [sys.executable, "-c", "from research.attribution.gain_surgery import main; assert callable(main)"]
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    records.append(dict(suite="historical_module_import", command=command,
                        returncode=process.returncode, stderr=process.stderr,
                        scope="Import only, no attribution refit"))
    (OUT / "test_receipt.json").write_text(json.dumps(dict(records=records,
        passed=all(r["returncode"] == 0 for r in records),
        scope="Changed follow-up implementation and causal forecast checks; no full GPU or GP replay"), indent=2) + "\n")
    if not all(r["returncode"] == 0 for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
