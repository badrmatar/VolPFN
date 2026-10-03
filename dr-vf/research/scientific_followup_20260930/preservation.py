"""Freeze and verify the pre-follow-up evidence without editing it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/outputs/scientific_followup_20260930"
OLD = ROOT / "research/outputs/repository_audit_20260929"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def candidates():
    previous = json.loads((OLD / "original_manifest.json").read_text())
    files = {ROOT / name for name in previous}
    # The completed audit's own copies and receipts are now historical inputs.
    for directory in (OLD, ROOT / "research/dissertation_audited_20260929",
                      ROOT / "research/data/oxfordman_barde",
                      ROOT / "research/gpvol_full", ROOT.parent / "calibeat-var-es"):
        for path in directory.rglob("*"):
            if not path.is_file() or any(part in {"__pycache__", ".git", ".venv", "venv"} for part in path.parts):
                continue
            if path.suffix in {".pyc", ".nbc", ".nbi"}:
                continue
            files.add(path)
    files.add(ROOT / "output/pdf/Matar_MSc_dissertation_audited_20260929.pdf")
    files.add(ROOT / "output/presentations/Matar_Viva_Complete_Transcript_audited_20260929.txt")
    archive = ROOT / "research/outputs/programme_closure_20260919_v1/remote_export_v2/completed_checkpoint_archive.tar"
    if archive.exists():
        files.add(archive)
    for directory in ("attribution_extract_v1", "attribution_extract_full_v1",
                      "frontier_gpvol_20260926_v1", "frontier_gpvol_gpu_20260926_v1"):
        base = ROOT / "research/outputs" / directory
        files.update(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "verify"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_file = OUT / "preserved_inputs.json"
    if args.mode == "freeze":
        if manifest_file.exists():
            raise FileExistsError("Do not replace the initial preservation manifest")
        manifest = {str(path): {"sha256": digest(path), "bytes": path.stat().st_size}
                    for path in candidates()}
        with manifest_file.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"frozen_files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values())}))
    else:
        manifest = json.loads(manifest_file.read_text())
        changed, missing = [], []
        for name, value in manifest.items():
            path = Path(name)
            if not path.is_file():
                missing.append(name)
            elif digest(path) != value["sha256"]:
                changed.append(name)
        report = {"checked_files": len(manifest), "changed": changed, "missing": missing,
                  "passed": not changed and not missing,
                  "scope": "Retained pre-follow-up evidence, data, frozen tail/GP code, audited manuscript, and checkpoint archive; new outputs excluded"}
        (OUT / "preservation_check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError("Pre-follow-up evidence changed")


if __name__ == "__main__":
    main()
