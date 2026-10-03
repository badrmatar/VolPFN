"""Exact author-source inspection/replay; no substitute for Bayesian RECH.

The pinned public archive contains LSTM-tGARCH, whereas the 2022 paper used
RNN-GARCH.  Neither a point-estimate fit nor this archive's native demonstration
is a valid RECH2022 forecast in the programme's common daily-return protocol.
This module therefore leaves that comparison slot explicitly unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile


SOURCE_URL = "https://github.com/VBayesLab/RECH"
SOURCE_COMMIT = "184e7f386dd00b3d8efe2ccf40960f1a3da6368c"
PINNED_FILES = {
    "Data/individual_stock_data.mat": "30ea66240b058c7525b1842bc70e367281ff671c476e79a89ffb7ec4938b2d96",
    "Data/realized_data.mat": "67892c46e6d3425e9a3b147a22f7cc8b7324a78013ff200f16ff8396cce6502b",
    "Does anything beat LSTM-tGARCH?/Readme": "45130223d805b43cf96e06e589e9cbb2b1714ca333b3c38a2ecb8a477a5c743c",
    "Does anything beat LSTM-tGARCH?/for individual stocks - largecap.zip": "43b243788185ae7a85303582fa3bc41a8cdac7cc1771b193ea793523f722dfd4",
    "Does anything beat LSTM-tGARCH?/for individual stocks - midcap.zip": "006acba94e27da106c7b3caa362d7e6ad87c37ff29c00ab580930d9568ddb694",
    "Does anything beat LSTM-tGARCH?/for stock index.zip": "44d74b12fc31aa16f6adbe230671bdeeef50b900bb0755871e8e0f10c16f2e3b",
    "README.md": "62b20e93c0c453bad8f0d59e3f3459550bc7c69ff7ba24008d0053d013112db0",
    "RECH (LSTM_tGARCH).7z": "89757621e48a3fe9a237523cfbeaa3107c54c69193923ea0242504edd75960d4",
}
COMMON_PROTOCOL_GAPS = (
    "The pinned public code implements later LSTM-tGARCH, not the 2022 RNN-GARCH specification.",
    "The native examples use close-to-close returns and demean the complete series before splitting; the programme requires causal open-to-close forecasts.",
    "The Student-t degrees-of-freedom prior permits nu <= 2; a finite posterior predictive second moment is not established.",
    "The author forecast uses a plug-in mean-nu conversion, not the posterior expectation of each particle's second moment.",
    "No validated causal Bayesian adapter for the shared market/synthetic protocol has been implemented.",
)


class RECHUnavailable(RuntimeError):
    """Raised instead of silently emitting a surrogate forecast."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(source_root: str | Path | None) -> dict:
    """Check all tracked archive/data contents without executing author code."""
    if source_root is None:
        return {"verified": False, "reason": "No local author-source checkout supplied.", "files": {}}
    root = Path(source_root).expanduser().resolve()
    files = {}
    for relative, expected in PINNED_FILES.items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        files[relative] = {"expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected}
    return {
        "root": str(root), "url": SOURCE_URL, "pinned_commit": SOURCE_COMMIT,
        "verified": all(record["matches"] for record in files.values()),
        "files": files, "license": "No LICENSE file found in the pinned repository or inspected archives; no redistribution permission inferred.",
    }


def _executable(explicit: str | Path | None, name: str) -> str | None:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    if name == "matlab":
        for candidate in sorted(Path("/Applications").glob("MATLAB_R*.app/bin/matlab"), reverse=True):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
    return None


def inspect_runtime(matlab: str | Path | None = None, octave: str | Path | None = None) -> dict:
    """Finding an executable does not confirm licences or Octave equivalence."""
    return {
        "matlab": _executable(matlab, "matlab"), "octave": _executable(octave, "octave"),
        "matlab_toolboxes_checked": False,
        "matlab_required_toolboxes": ["Statistics and Machine Learning Toolbox", "Parallel Computing Toolbox"],
        "octave_author_source_compatibility_verified": False,
    }


def availability(source_root=None, matlab=None, octave=None) -> dict:
    source = inspect_source(source_root)
    runtime = inspect_runtime(matlab, octave)
    reasons = list(COMMON_PROTOCOL_GAPS)
    if not source["verified"]:
        reasons.append("Pinned author-source contents are absent or do not match every expected hash.")
    if runtime["matlab"] is None:
        reasons.append("MATLAB is unavailable; Octave compatibility has not been validated.")
    return {
        "model": "RECH2022", "status": "unavailable", "available": False,
        "common_daily_available": False, "reasons": reasons,
        "source": source, "runtime": runtime,
        "native_author_variant": "LSTM-tGARCH Bayesian SMC; separate historical demonstration only",
        "native_matlab_replay_candidate": source["verified"] and runtime["matlab"] is not None,
        "forecast_arrays_created": False,
    }


class RECH2022:
    """Fail-closed comparison slot.  Intentionally not a torch/MLE model."""

    def __init__(self, source_root=None, matlab=None, octave=None):
        self.receipt = availability(source_root, matlab, octave)

    def _unavailable(self):
        raise RECHUnavailable("RECH2022 unavailable: " + " ".join(self.receipt["reasons"]))

    def fit(self, *args, **kwargs):
        self._unavailable()

    def forward(self, *args, **kwargs):
        self._unavailable()

    def forecast(self, *args, **kwargs):
        self._unavailable()


def _extract_zip(archive: Path, destination: Path):
    """Extract local pinned source only; reject traversal and symbolic links."""
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            mode = (entry.external_attr >> 16) & 0o170000
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or mode == 0o120000:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
        bundle.extractall(destination)


def _matlab_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_author_source(source_root, out, seed: int, matlab=None, timeout_seconds=None) -> dict:
    """Replay the pinned native stock-index example, never the common protocol.

    Requires a caller-provided checkout and fresh output directory.  No MLE,
    port, source edits, downloaded runtime, or cached-results analysis fallback.
    The source's complete-series centring is preserved for provenance; therefore
    this native demonstration cannot enter the causal comparison tables.
    """
    receipt = availability(source_root, matlab=matlab)
    if not receipt["source"]["verified"]:
        raise RECHUnavailable("Native replay rejected: author-source hashes are missing or changed.")
    executable = receipt["runtime"]["matlab"]
    if executable is None:
        raise RECHUnavailable("Native replay unavailable: MATLAB executable was not found.")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    destination = Path(out).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output must be fresh: {destination}")
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Replay output must not overlap the supplied author-source checkout.")
    destination.mkdir(parents=True)
    extracted = destination / "author_source"
    archive = root / "Does anything beat LSTM-tGARCH?/for stock index.zip"
    _extract_zip(archive, extracted)
    entry = extracted / "for stock index/run_RECH_all_stock_index.m"
    if not entry.is_file():
        raise RECHUnavailable("Pinned stock-index entry point is missing after extraction.")
    # Both installation and licence checks are needed before the parfor/SMC run.
    batch = (
        "assert(license('test','Statistics_Toolbox') && exist('gamrnd','file') ~= 0 && exist('pdf','file') ~= 0,"
        "'RECH:MissingStatistics','Statistics toolbox installation/licence required');"
        "assert(license('test','Distrib_Computing_Toolbox') && exist('gcp','file') ~= 0,"
        "'RECH:MissingParallel','Parallel Computing toolbox installation/licence required');"
        f"rng({seed},'twister');cd({_matlab_literal(entry.parent)});run({_matlab_literal(entry.name)});"
    )
    receipt.update({
        "replay_label": "native_author_LSTM_tGARCH_demonstration_NOT_causal_RECH2022_replication",
        "replay_seed": seed, "matlab_batch": batch, "output": str(destination),
        "rng_scope": "Client twister seed only; author parfor worker random streams are not claimed deterministic.",
        "primary_comparison_slot": "unavailable", "native_replay_status": "running",
    })
    receipt_file = destination / "REPLAY_RECEIPT.json"
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (destination / "MATLAB.log").open("w") as log:
            result = subprocess.run([executable, "-batch", batch], cwd=entry.parent,
                                    stdout=log, stderr=subprocess.STDOUT, check=False,
                                    timeout=timeout_seconds)
        receipt["native_replay_exit_code"] = result.returncode
        receipt["native_replay_status"] = "completed" if result.returncode == 0 else "failed"
        receipt["runtime"]["matlab_toolboxes_checked"] = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as error:
        receipt["native_replay_status"] = "failed"
        receipt["native_replay_error"] = str(error)
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--source-root")
    inspect.add_argument("--matlab")
    inspect.add_argument("--octave")
    replay = sub.add_parser("native-replay", help="Separate later LSTM-tGARCH native demonstration; never common RECH2022 forecasts")
    replay.add_argument("--source-root", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--seed", type=int, required=True)
    replay.add_argument("--matlab")
    replay.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args()
    if args.action == "inspect":
        result = availability(args.source_root, args.matlab, args.octave)
    else:
        result = run_author_source(args.source_root, args.out, args.seed, args.matlab, args.timeout_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
