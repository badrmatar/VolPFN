"""Verify final follow-up artifacts and receipts without changing old evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/outputs/scientific_followup_20260930"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(relative):
    return json.loads((OUT / relative).read_text())


def require_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise AssertionError("Stale or changed receipt: " + str(path))
    return actual


def main():
    records = {}
    frozen = read("preserved_inputs.json")
    for name, receipt in frozen.items():
        require_hash(name, receipt["sha256"])
    records["preservation"] = dict(checked_files=len(frozen),
        bytes=sum(v["bytes"] for v in frozen.values()), passed=True)

    with (OUT / "ISSUE_RESOLUTION.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    original = read("resolution_coverage.json")["original_unresolved_ids"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == 22
    assert all(ids.count(i) == 1 for i in original) and len(original) == 14
    for row in rows:
        path, number = row["corrected_source_or_manuscript"].rsplit(":", 1)
        assert Path(path).is_file() and 0 < int(number) <= len(Path(path).read_text().splitlines())
        assert row["resolution"] and row["remaining_unverified_or_historical"]
    records["resolution"] = dict(rows=len(rows), original_unresolved_ids=original,
        all_classified_once=True, exact_correction_anchors_exist=True)

    tests = read("support/test_receipt.json")
    assert tests["passed"] and all(x["returncode"] == 0 for x in tests["records"])
    counts = {}
    for row in tests["records"]:
        if "log" in row:
            text = Path(row["log"]).read_text()
            count = re.search(r"Ran (\d+) tests? in", text)
            assert count and re.search(r"\nOK\s*$", text)
            counts[row["suite"]] = int(count.group(1))
    assert counts == dict(attribution_launcher=3, gp_likelihood=9, optimizer_repair=11)
    records["tests"] = dict(test_cases=sum(counts.values()), counts=counts,
        historical_module_import_passed=True, passed=True)

    manifests = {}
    for name, base in (("optimizer/OUTPUT_MANIFEST.json", ROOT),
                       ("optimizer/repaired_v2/FILE_RECEIPTS.json", OUT / "optimizer/repaired_v2")):
        value = read(name)
        for record in value["files"]:
            path = base / record["path"]
            require_hash(path, record["sha256"])
            assert path.stat().st_size == record["bytes"]
        manifests[name] = dict(files=len(value["files"]), passed=True, sha256=sha(OUT / name))
    records["optimizer_manifests"] = manifests

    quality = read("optimizer/repaired_v2/quality.json")
    fhs = read("optimizer/repaired_v2/fhs_optimizer_quality.json")
    assert quality["refit_cohort_records"] == 3400
    assert quality["repaired_nonconverged"] == quality["scored_substitutions"] == 0
    assert quality["max_archived_objective_reproduction_error"] == 0
    assert quality["max_repaired_objective_deterioration"] <= 1e-7
    assert quality["unchanged_daily_checks"] == 9408
    assert quality["unchanged_methods_max_daily_loss_difference"] == 0
    assert fhs["refit_cohort_records"] == 850 and fhs["max_repaired_objective_deterioration"] <= 1e-7
    protocol = read("optimizer/repaired_v2/protocol.json")
    for path, value in protocol["source_sha256"].items():
        require_hash(path, value)
    summary = read("optimizer/repaired_v2/summary.json")
    require_hash(ROOT / "research/scientific_followup_20260930/optimizer_statistics.py", summary["code_sha256"])
    gate = read("optimizer/repaired_v2/support_statistics_gate.json")
    assert gate["unchanged_scientific_files"] == 222 and not gate["forecasts_rerun"] and not gate["statistics_rerun"]
    records["optimizer_quality"] = dict(main_refit_records=3400, separate_fhs_records=850,
        nonconverged_selected=0, scored_substitutions=0, unchanged_daily_arrays=9408,
        support_gate_scientific_files_unchanged=222, passed=True,
        scope="Post-exposure same-model historical tail sensitivity; not other benchmark windows or a global optimum")

    artifacts = read("manuscript/FINAL_ARTIFACTS.json")
    require_hash(artifacts["final_pdf"], artifacts["final_pdf_sha256"])
    require_hash(artifacts["source_zip"], artifacts["source_zip_sha256"])
    for name, value in artifacts["source_files"].items():
        require_hash(Path(artifacts["source_directory"]) / name, value)
    assert len(artifacts["source_files"]) == artifacts["source_file_count"] == 158
    with zipfile.ZipFile(artifacts["source_zip"]) as z:
        assert z.testzip() is None
        members = [name for name in z.namelist() if not name.endswith("/")]
        assert len(members) == 158
        archive_root = Path(artifacts["source_directory"]).name + "/"
        for name, value in artifacts["source_files"].items():
            matches = [member for member in members if member == archive_root + name]
            assert len(matches) == 1 and hashlib.sha256(z.read(matches[0])).hexdigest() == value
    for name, value in artifacts["verification_files"].items():
        path = OUT / "manuscript" / name
        if name == "RECEIPT.json":
            path = OUT / "manuscript/pdf_qa/RECEIPT.json"
        require_hash(path, value)
    doc = read("manuscript/source_and_document_checks.json")
    qa = read("manuscript/pdf_qa/RECEIPT.json")
    clean = read("manuscript/clean_source_build.json")
    assert doc["pages_total"] == qa["pages_total"] == clean["pages_total"] == 114
    assert doc["formal_appendix_byte_identical"] and all(doc["original_numeric_tables_preserved"].values())
    assert not doc["missing_labels"] and not doc["duplicate_labels"]
    assert not doc["unresolved_references"] and not doc["overfull_boxes"]
    assert qa["pdf_sha256"] == doc["pdf_sha256"] == artifacts["final_pdf_sha256"]
    assert qa["status"] == "passed_visual_inspection"
    for name, value in qa["images"].items():
        require_hash(OUT / "manuscript/pdf_qa" / name, value)
    assert clean["status"] == "passed_clean_extracted_source_build"
    assert clean["archive_sha256"] == artifacts["source_zip_sha256"]
    assert not clean["unresolved_references"] and not clean["overfull_boxes"]
    tables = read("manuscript/optimizer_table_receipt.json")
    for path, value in tables["sources"].items():
        require_hash(path, value)
    records["manuscript"] = dict(pages=114, source_files=158,
        pdf=artifacts["final_pdf"], pdf_sha256=artifacts["final_pdf_sha256"],
        clean_source_build=True, original_numeric_tables_preserved=True,
        formal_proof_byte_identical=True, all_page_rendering_and_qa_receipt_verified=True,
        scientific_result_source_hashes_current=True, passed=True)

    delivered = ["FOLLOWUP_REPORT.md", "WORKFLOW.md", "ISSUE_RESOLUTION.md", "ISSUE_RESOLUTION.csv",
                 "resolution_coverage.json", "support/test_receipt.json", "support/runtime_inventory.json",
                 "support/cli_receipt.json",
                 "support/benchmark_fit_receipts.json", "support/attribution_fit_receipts.json",
                 "manuscript/FINAL_ARTIFACTS.json", "preservation_check.json"]
    records["delivery_sha256"] = {name: sha(OUT / name) for name in delivered}
    records["supported_source_sha256"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "research/scientific_followup_20260930").rglob("*"))
        if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts
    }
    records["genuinely_unverified"] = [
        "Full zero-inclusive posterior predictive optimality and finite-model attainment",
        "Incremental auxiliary-head benefit",
        "Optimized expanded driver-class ordering",
        "Literal theorem estimator on markets and its market assumptions",
        "Complete same-protocol repaired GP ranking, long-run stability and quadrature error",
        "Unique blend information and universal benchmark/statistical superiority",
        "Worldwide first-use novelty",
        "Portable fresh-clone and complete full-GPU reproduction",
    ]
    records["passed"] = True
    records["scope"] = "Delivery integrity and feasible scientific follow-up checks; no stylistic rewrite or claim that missing experiments were performed. Pre-existing Git state is not certified clean."
    (OUT / "FINAL_VERIFICATION.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(passed=True, preserved_files=len(frozen), resolution_rows=len(rows),
                         test_cases=sum(counts.values()), pages=114, source_files=158)))


if __name__ == "__main__":
    main()
