"""Read-only checks separating later benchmark procedures from legacy tail fits."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/outputs/scientific_followup_20260930/support"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_receipts():
    old_path = ROOT / "research/outputs/combined_leaderboard_v2/analysis/fit_diagnostics.csv"
    new_path = ROOT / "research/outputs/frontier_extension_20260926_v1/analysis/results.json"
    older = defaultdict(lambda: defaultdict(int))
    with old_path.open() as handle:
        for row in csv.DictReader(handle):
            for name in ("fit_attempts", "fit_failures", "numerical_failures", "substitutions", "floors"):
                older[row["model"]][name] += int(row[name])
    cells = json.loads(new_path.read_text())["cell_audits"]
    newer = defaultdict(lambda: defaultdict(int))
    violations = []
    for cell in cells:
        adapter = cell.get("adapter", {})
        if "fit_attempt_count" not in adapter:
            continue
        model = cell["model"]
        newer[model]["market_cells"] += 1
        for name in ("fit_attempt_count", "fit_failure_count", "numerical_failure_count",
                     "substitution_count", "floor_count"):
            newer[model][name] += adapter[name]
        for attempt in adapter["attempts"]:
            if attempt.get("accepted") is True and attempt.get("convergence_flag", 0) != 0:
                violations.append(dict(cell=cell["cell"], defect="accepted nonconverged fit", attempt=attempt))
            if "history_end_index" in attempt and "origin_index" in attempt:
                if attempt["history_end_index"] >= attempt["origin_index"]:
                    violations.append(dict(cell=cell["cell"], defect="noncausal recorded fit window", attempt=attempt))
    if violations:
        raise RuntimeError("Recorded later accepted-fit or causal-window gate failed")
    return dict(
        source_receipts=[dict(path=str(p), sha256=sha(p)) for p in (old_path, new_path)],
        historical_2015_2016=dict(older), extension_2019_2022=dict(newer),
        extension_cell_audits=len(cells), accepted_convergence_or_window_violations=violations,
        scope="Saved receipts and source acceptance policies; not a fresh full forecast replay or proof of a global optimum",
        conclusion="The legacy expanding-window tail optimizer defect does not license replacing these distinct later benchmark procedures. Failed later attempts are rejected; causal carry/fallback dates remain explicitly counted.",
        source_policy_paths=[str(ROOT / "research/tier_a_confirmation/econometric.py"),
                             str(ROOT / "research/combined_leaderboard/classical.py")],
    )


def runtime_receipts():
    runtimes = {
        "scientific_legacy_cpu": Path("/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python"),
        "frontier_pinned": ROOT / "research/environments/frontier_v1/bin/python",
        "moirai_isolated": ROOT / "research/environments/moirai_v1/bin/python",
    }
    code = """import importlib.metadata as m, json, platform, sys
names=['numpy','scipy','pandas','torch','arch','numba','chronos-forecasting','uni2ts','tabpfn','tsfm-public']
versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'packages':versions}))
"""
    result = {}
    for name, interpreter in runtimes.items():
        if not interpreter.exists():
            result[name] = dict(path=str(interpreter), available=False)
            continue
        process = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True)
        result[name] = dict(path=str(interpreter), available=True, returncode=process.returncode)
        if process.returncode == 0:
            result[name]["observed"] = json.loads(process.stdout)
        else:
            result[name]["error"] = process.stderr
    return dict(observed_local_runtimes=result,
                portable_lock_verified=False, fresh_clone_verified=False,
                ten_full_gpu_trajectories_replayed=False,
                scope="Observed local package versions only; absence here is not a claim that an optional adapter has never run")


def attribution_fit_receipts():
    studies = {}
    for name, folder in (("compact_v3", "attribution_gain_v3"),
                         ("expanded_drivers_v1", "attribution_driver_swap_v1")):
        path = ROOT / "research/outputs" / folder / "folds.json"
        records = json.loads(path.read_text())
        counts = defaultdict(lambda: {"likelihood": 0, "path": 0, "folds": 0,
                                      "failed_likelihood_folds": [], "failed_path_folds": []})
        for fold, specifications in records.items():
            for spec, receipt in specifications.items():
                row = counts[spec]
                row["folds"] += 1
                row["likelihood"] += int(receipt["likelihood_converged"])
                row["path"] += int(receipt["freerun_converged"])
                if not receipt["likelihood_converged"]:
                    row["failed_likelihood_folds"].append(fold)
                if not receipt["freerun_converged"]:
                    row["failed_path_folds"].append(fold)
        studies[name] = dict(source=str(path), sha256=sha(path), specifications=dict(counts))
    return dict(studies=studies,
                scope="Archived stopping receipts, not independent reruns or proof of global optimality",
                conclusion="The compact-v3 likelihood/path runs all converge. Expanded driver interaction-only comparisons include failed fits: their attained-budget ordering does not establish intrinsically superior or fully optimized model classes. That stronger claim requires matched continued fits; historical tables remain diagnostics.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, result in (("benchmark_fit_receipts", benchmark_receipts()),
                         ("runtime_inventory", runtime_receipts()),
                         ("attribution_fit_receipts", attribution_fit_receipts())):
        path = OUT / (name + ".json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
