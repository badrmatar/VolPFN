"""Read-only dependency and cached-interface audit; never fit or score a model.

Run with any working Python 3.9+ interpreter. The JSON goes to stdout. Package
probes run in the existing project interpreters, without installing anything.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "experiments/outputs"
SETUP = OUTPUTS / "return_only_benchmark_setup_v1"
CLASSICAL_PYTHON = Path(
    "/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python"
)


def python_probe(executable):
    code = """
import importlib.metadata as metadata
import json, sys
packages = {}
for name in ('numpy', 'pandas', 'scipy', 'arch', 'torch'):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'executable': sys.executable, 'python': sys.version,
                  'packages': packages}))
"""
    if not executable.exists():
        return {"executable": str(executable), "status": "missing"}
    result = subprocess.run([str(executable), "-c", code], capture_output=True,
                            text=True, timeout=30,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if result.returncode:
        return {"executable": str(executable), "status": "probe_failed",
                "returncode": result.returncode, "stderr": result.stderr.strip()}
    return json.loads(result.stdout)


def r_probe():
    executable = shutil.which("Rscript")
    if executable is None:
        return {"status": "Rscript_missing"}
    # Inspect installation and loadability in both the default and project libs.
    code = r'''
cat("version\t", R.version.string, "\n", sep="")
for (scope in c("default", "project_plus_default")) {
  if (scope == "project_plus_default") .libPaths(c(commandArgs(TRUE)[1], .libPaths()))
  cat(scope, "\tlibraries\t", paste(.libPaths(), collapse=";"), "\n", sep="")
  for (p in c("rugarch", "betategarch")) {
    installed <- nzchar(system.file(package=p))
    loaded <- if (installed) requireNamespace(p, quietly=TRUE) else FALSE
    version <- if (installed) as.character(packageVersion(p)) else "ABSENT"
    cat(scope, p, version, loaded, sep="\t"); cat("\n")
  }
}
'''
    result = subprocess.run([executable, "-e", code, str(SETUP / "r-library")],
                            text=True, capture_output=True, timeout=30)
    return {"executable": executable, "returncode": result.returncode,
            "diagnostics": result.stdout.strip().splitlines(),
            "stderr": result.stderr.strip()}


def foundation_cache_probe():
    manifest_path = (OUTPUTS / "programme_closure_20260919_v1"
                     / "benchmark_confirmation_plan_v3/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    adapter_path = ROOT / "experiments/return_only_benchmark/foundation_adapters.py"
    tree = ast.parse(adapter_path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    assets = SETUP / "foundation_cache"
    config = json.loads((assets / "ttm" / constants["TTM_REVISION"]
                         / "config.json").read_text())

    # Forecast artifacts have model ownership in directory/filename. Exclude
    # package and weight caches: these contain examples/configs, not panel runs.
    excluded_dirs = {"foundation_venv", "r-library", "uv_cache", "foundation_cache",
                     "__pycache__", "package_sources"}
    candidates = []
    prediction_files = []
    for directory, dirs, files in os.walk(OUTPUTS):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for name in files:
            path = Path(directory) / name
            if name == "predictions.json":
                prediction_files.append(str(path.relative_to(ROOT)))
            if path.suffix not in {".json", ".csv", ".npz", ".parquet"}:
                continue
            if (any(part.lower() in {"ttm", "sundial"} for part in path.parts)
                    or any(model in name.lower() for model in ("ttm", "sundial"))):
                candidates.append(str(path.relative_to(ROOT)))
    smoke = []
    for name in ("foundation_smoke_attempt1", "foundation_smoke_s1000_attempt1"):
        path = SETUP / name / "timing.json"
        data = json.loads(path.read_text())
        smoke.append({"path": str(path.relative_to(ROOT)), "kind": data["kind"],
                      "market_data_accessed": data["market_data_accessed"],
                      "context_length": data["context_length"],
                      "models": [x["model"] for x in data["results"]]})
    return {
        "required_panel_manifest": str(manifest_path.relative_to(ROOT)),
        "required_evaluation_count": len(manifest["origins"]),
        "required_calibration_count": len(manifest["calibration_origins"]),
        "required_context_counts": sorted({x["context_count"] for x in
            manifest["origins"] + manifest["calibration_origins"]}),
        "adapter_context_length": constants["CONTEXT_LENGTH"],
        "ttm_checkpoint_context_length": config["context_length"],
        "ttm_revision": constants["TTM_REVISION"],
        "sundial_revision": constants["SUNDIAL_REVISION"],
        "ttm_sundial_named_forecast_candidates": sorted(candidates),
        "prediction_artifact_inventory": sorted(prediction_files),
        "synthetic_smoke_evidence": smoke,
    }


def main():
    interpreters = {
        "classical": CLASSICAL_PYTHON,
        "foundation": SETUP / "foundation_venv/bin/python",
        "frontier": ROOT / "experiments/environments/frontier_v1/bin/python",
        "moirai": ROOT / "experiments/environments/moirai_v1/bin/python",
    }
    report = {
        "kind": "optional_dependency_and_cache_audit_no_scoring",
        "python": {key: python_probe(path) for key, path in interpreters.items()},
        "r": r_probe(), "foundation": foundation_cache_probe(),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
