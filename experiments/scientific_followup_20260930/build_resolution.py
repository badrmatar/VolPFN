"""Build the follow-up resolution table without altering the original audit."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs/scientific_followup_20260930"
MAN = ROOT / "experiments/dissertation_scientifically_corrected_20260930"


def line(path, needle):
    path = Path(path)
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if needle in text:
            return str(path) + ":" + str(number)
    raise ValueError("Missing evidence anchor: " + str(path) + " / " + needle)


def manuscript(relative, needle):
    return line(MAN / relative, needle)


def evidence(relative, needle=None):
    path = ROOT / relative
    if needle is not None:
        return line(path, needle)
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path) + (":1" if path.is_file() else "")


def main():
    original = list(csv.DictReader((OUT / "triage/TRIAGE.csv").open()))
    assert len(original) == 14 and len({x["id"] for x in original}) == 14
    rows = []
    actions = {
        "W17": ("2,3,4", "Correct the target to the weighted scored-position law; retain the actual mask and model.", "Exact loss/denominator and bounded target interpretation checked.", "Unconditional zero-inclusive posterior prediction and attainment by the finite shared model remain unverified.", "Retrain an explicitly defined all-target/zero-atom alternative only if that stronger target is required.", manuscript("appendices/B_technical.tex", "P_{\\rm sc}(A)")),
        "W27": ("4", "Preserve recorded GPU backend and qualify execution coverage.", "Saved training records/checkpoint checks identify the backend; observed CPU environments are separately recorded.", "All ten 20,000-update GPU trajectories were not independently replayed.", "Full GPU replay needed only for a complete reproduction claim.", manuscript("appendices/B_technical.tex", "RTX 4090")),
        "W59": ("2,4", "Remove global first-use language from the corrected presentation; keep the bounded dissertation contribution.", "Two priority sentences removed in a separate transcript with hashes; no global priority asserted.", "Worldwide priority remains unverified; no systematic exhaustive novelty search was performed.", "A defined, reproducible literature search is needed if priority is claimed.", manuscript("chapters/08_discussion_new.tex", "worldwide first use")),
        "W64": ("2,3", "Describe the actual Gaussian mean/scale auxiliary NLL and withdraw isolated-benefit claims.", "Gaussian helper, 0.1 loss weight, adjusted label and shared representation match the implementation.", "No no-auxiliary/direct-moment ablation establishes incremental value; the label is not exact conditional market variance.", "Matched auxiliary alternatives and fixed training/selection protocol needed for a benefit claim.", manuscript("chapters/04_forecasting.tex", "Training rewards density")),
        "W65": ("2,3,4", "Restrict superiority to tested adapters, panels and endpoint families; treat GP as a design precedent.", "The manuscript separates the three-comparator benchmark from later information groups and descriptive ranks.", "Superiority over unrun GP or all modern models/markets remains unverified.", "Complete prescribed adapters/coverage and a new declared panel needed for broader claims.", manuscript("chapters/08_discussion_new.tex", "GP-Vol offers")),
        "T14": ("2,3,4", "Attach the proof to its projected estimator, and historical results to the archived mapped-pair safeguards.", "The implementation/proof mismatch and absence of pathwise no-harm are explicitly propagated.", "The market wrapper does not implement the literal theorem estimator.", "A separately versioned projected-estimator experiment is needed to connect proof to execution.", manuscript("chapters/06_recalibration.tex", "The archived wrapper enforces")),
        "T18": ("2,4", "Retain iid/high-moment/density/margin assumptions as conditional mathematical premises.", "Assumptions and the t3 stress-test violation are source-qualified; numerical checks do not establish market assumptions.", "The theorem's real-market assumptions remain unverified.", "Dependence/heavy-tail stress tests can assess sensitivity, but cannot prove universal market assumptions.", manuscript("chapters/06_recalibration.tex", "This implies independence")),
        "T21": ("2,4", "Reject the frozen README's universal grouped-hindsight dominance assertion; retain archive attribution.", "A preserved numerical counterexample refutes the claim; current explanation specifies the constant comparator class.", "Old frozen comments remain inconsistent historical material; no online/population dominance follows.", "No new experiment is needed to withdraw the assertion.", manuscript("chapters/06_recalibration.tex", "hindsight")),
        "T37": ("1,4", "Provide a supported module launcher with an explicitly new output directory.", "Three launcher checks and actual module import pass; frozen scientific source remains unchanged.", "The obsolete direct command remains incorrect; a fresh full exploratory attribution refit was not performed.", "Use the supported invocation; optimizing driver alternatives is a separate scientific check.", manuscript("appendices/C_supplementary.tex", "An optional supported launcher")),
        "T38": ("1,4", "Test the actual production adaptive dispatch in the new supported harness.", "Target/future perturbation checks cover 180 forecaster/wrapper paths; the test reaches adapt separately.", "The frozen obsolete test still has its dispatch bug; bounded tests are not a proof over all workflows.", "No method change or rerun is needed solely for the old test bug.", evidence("experiments/scientific_followup_20260930/test_optimizer_repair.py", "def test_repaired_forecasters_and_all_wrapper_paths_are_causal")),
        "T40": ("3,4", "Keep incomplete primary GP coverage unverified; do not replace the missing cell with a diagnostic seed/model.", "Original coverage gate remains strict; a versioned full two-seed, 28-market recipe is supplied.", "Full primary GP ranking remains unverified and was not run to completion.", "Replay all affected cells consistently from burn-in and complete calibration/strict inference if this ranking is needed.", manuscript("chapters/08_discussion_new.tex", "prescribed complete two-seed")),
        "T42": ("1,3,4", "Repair Gaussian likelihood algebra in a private versioned module; distinguish it from log-moment storage.", "Nine bounded tests, a high-precision oracle and an actual six-update 200-particle prefix pass.", "Long-run stability and prevention of the old GSPTSE collapse remain unverified.", "Full same-protocol replay needed; any prior/particle/fallback change must be another method version.", evidence("experiments/scientific_followup_20260930/gp/likelihood_v1.py", "def log_normal_obs")),
        "R13": ("2,3", "Keep blend findings descriptive and disclose proxy-specific online weights and unmatched controls.", "A separate corrected transcript withdraws unique-VolPFN attribution; original blend outputs are preserved.", "VolPFN-specific synergy and joint superiority over all richer-information comparators remain unverified.", "Matched GJR+TTM/recalibration-only controls and independent future evaluation needed for stronger claims.", evidence("output/presentations/Matar_Viva_Complete_Transcript_scientifically_corrected_20260930.txt", "Without a matched")),
        "R20": ("1,4", "Record supported version-specific local runtimes and commands without inventing a portable lock.", "Current interpreter/package inventory, supported imports and changed-code checks pass on this host.", "Fresh-clone portability, a unified lock and complete GPU/external execution remain unverified.", "Create version-specific clean-install locks/weight manifests and replay receipts for a portable-release claim.", evidence("experiments/outputs/scientific_followup_20260930/support/runtime_inventory.json")),
    }
    for item in original:
        classification, action, checked, remaining, decision, corrected = actions[item["id"]]
        rows.append(dict(id=item["id"], claim=item["claim"], original_status=item["current_status"],
                         category_numbers=classification, resolution=action,
                         verified_scope=checked, remaining_unverified_or_historical=remaining,
                         required_experiment_or_decision=decision,
                         corrected_source_or_manuscript=corrected,
                         original_dissertation_section=item["dissertation_section"],
                         source_evidence="; ".join(x for x in (item["code"], item["configuration"], item["results"], item["additional_evidence"]) if x),
                         detailed_triage=str(OUT / "triage/TRIAGE.csv")))
    def add(identifier, claim, category, action, checked, remaining, decision, corrected, sources, status="reopened or new"):
        rows.append(dict(id=identifier, claim=claim, original_status=status,
                         category_numbers=category, resolution=action, verified_scope=checked,
                         remaining_unverified_or_historical=remaining,
                         required_experiment_or_decision=decision,
                         corrected_source_or_manuscript=corrected,
                         original_dissertation_section="See corrected source and preserved audit ID",
                         source_evidence=sources, detailed_triage=str(OUT / "triage/CLAIM_PROPAGATION.md")))
    add("T28", "Archived nonconverged GARCH/GAS fits may undermine tail comparisons.", "1,3,4",
        "Diagnose material forecast changes and regenerate all affected same-model tail comparisons with convergence and actual historical-objective gates.",
        "Complete 56-cohort v2: all 3,400 main refit records reproduce the archive's training objective exactly; 850 separate FHS archive gates pass within tolerance; every repaired fit converges and is no worse within the 1e-7 acceptance tolerance; zero scored substitutions; 9,408 unchanged daily-loss arrays reproduce exactly. Standalone statistics rejects incomplete/noncanonical rosters and missing/failed main/FHS quality receipts.",
        "Local stopping success is not global optimality. This is an exposed historical numerical sensitivity, not independent confirmation.",
        "Use v2 supported results. V1 is preserved as a failed training-quality-gate sensitivity.",
        evidence("experiments/outputs/scientific_followup_20260930/optimizer/repaired_v2/quality.json"),
        evidence("experiments/scientific_followup_20260930/optimizer_core.py", "def fit") + "; " + evidence("experiments/scientific_followup_20260930/optimizer_paths.py", "historical_th"))
    add("W66", "A disclosure of failed optimizer acceptance is insufficient to validate historical specialist comparisons.", "1,2,4",
        "Keep original tables explicitly historical and supply regenerated tail evidence separately; use repaired outcomes for supported conclusions.",
        "Binding 17/19 development and 16/19 later decisions remain; five of six later Gaussian gains survive historical Holm. One informational later GAS ratio-window decision changes at 2.5%.",
        "Original archival rows are not convergence-checked results; the actual theorem/market estimator boundary persists.",
        "No further retraining is required for these repaired tail comparisons; retained old results must keep their identity.",
        evidence("experiments/outputs/scientific_followup_20260930/optimizer/repaired_v2/later/comparison_to_archive.json"),
        evidence("experiments/scientific_followup_20260930/OPTIMIZER_PROTOCOL.md"))
    add("W42", "Unequal driver fit success does not establish superiority of fully optimized driver classes.", "2,3,4",
        "Withdraw intrinsic optimized additive-over-interaction interpretation; retain attained-budget diagnostic tables and the narrower additive-versus-converged-core result.",
        "Compact v3 all 12 likelihood/path fits converge. Expanded interaction-only likelihood fails two of four folds; additive/core succeed four of four.",
        "Optimized class ranking remains unverified; source disclosure alone does not prove failed alternatives are weaker.",
        "Matched continued fits required only if that stronger mechanistic comparison is claimed.",
        manuscript("chapters/05_analysis.tex", "Only 57"),
        evidence("experiments/outputs/scientific_followup_20260930/support/attribution_fit_receipts.json"))
    add("GP01", "Original Gaussian arithmetic can miscompute finite likelihoods through premature clipping/underflow.", "1,3",
        "Replace only the private supported observation-likelihood binding using log-domain exact Gaussian algebra.",
        "High-precision cancellation/overflow/zero cases, causal filter execution, exact resume and original-module preservation pass.",
        "Nine tests and a prefix do not establish full empirical stabilization or rank.",
        "Use the separately versioned replay if full GP coverage is sought.",
        evidence("experiments/scientific_followup_20260930/gp/likelihood_v1.py", "def log_normal_obs"),
        evidence("experiments/gpvol_full/gpvol_rapcf.py", "def log_normal_obs") + "; " + evidence("experiments/outputs/scientific_followup_20260930/support/gp_likelihood_tests.log"))
    add("GP02", "Finite Gauss-Hermite native-density quadrature is not exact integration.", "2,4",
        "Use numerical-quadrature wording in supported GP documentation; preserve frozen original comments.",
        "The source uses 40-node quadrature, so exact-integration terminology is an overstatement.",
        "Actual density quadrature error remains unverified; no GP result is inserted into the main dissertation comparison.",
        "Order/error diagnostics needed if quadrature accuracy is made a substantive empirical claim.",
        evidence("experiments/scientific_followup_20260930/gp/README.md", "40-node"),
        evidence("experiments/gpvol_full/gpvol_rapcf.py", "hermgauss"))
    add("R15", "The later daily-return leaderboards use different fitting and failure policies from the legacy tail engine.", "2,4",
        "Audit later saved acceptance/window receipts separately; do not silently substitute the tail repair.",
        "644 extension cell audits: no recorded accepted nonconverged or noncausal fits. Rejected-fit carry dates are counted, including APARCH 20 and EGARCH 80 in 2019–2022.",
        "This receipt check is not a full fresh execution or global-optimum guarantee; adaptive/fallback procedures remain part of the comparator.",
        "No demonstrated legacy-optimizer defect requires replacing these distinct panels.",
        evidence("experiments/outputs/scientific_followup_20260930/support/benchmark_fit_receipts.json"),
        evidence("experiments/tier_a_confirmation/econometric.py", "def _fit_one") + "; " + evidence("experiments/combined_leaderboard/classical.py", "convergence_flag"), "previously verified; scope rechecked")
    add("TARGET01", "Squared returns and realised kernel do not automatically evaluate the identical conditional moment.", "2,4",
        "Define h as the issued forecast's second moment and qualify RK as an ex-post variability proxy with a separate endpoint.",
        "Mixture second moment includes squared means; it differs from centered variance for nonzero mean. QLIKE scoring and input/evaluation roles are separated.",
        "RK conditional unbiasedness for next open-to-close squared returns is not established; no proxy-robustness theorem is automatically transferred.",
        "Keep separate proxy claims; a stronger common-target interpretation requires additional observation/measurement assumptions.",
        manuscript("chapters/04_forecasting.tex", "The expected squared return"),
        evidence("src/volpfn/model.py", "def mixture_second_moment") + "; " + manuscript("chapters/03_data_new.tex", "Equality of a realised-kernel"), "new scientific qualification")
    add("STATS01", "Descriptive first-place ranks do not establish statistically significant superiority over every comparator.", "2,3,4",
        "Keep study-specific testing families, exposure, adapter and information-group scopes distinct; reject universal superiority.",
        "The corrected three-comparator family supports its registered contrasts. Later wider/extension families and blends retain their own, narrower adjusted outcomes.",
        "No global familywise statement, unknown-pretraining-overlap resolution or unique-blend claim is established.",
        "Independently declared periods/controls are needed for broader confirmation; no in-sample re-labeling creates it.",
        manuscript("chapters/08_discussion_new.tex", "Separate studies have separate"),
        evidence("experiments/outputs/repository_audit_20260929/TRACEABILITY.csv") + "; " + evidence("experiments/outputs/frontier_extension_20260926_v1/analysis/results.json"), "new scientific qualification")
    fields = list(rows[0])
    with (OUT / "ISSUE_RESOLUTION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    text = ["# Issue-by-issue scientific resolution, 30 September 2026", "",
            "Categories: **1** implementation/reproducibility defect; **2** claim correction; **3** missing experimental evidence; **4** historical difference or verification limit. Multiple categories can apply to one issue.", "",
            "All 14 original unresolved IDs are retained. Additional rows reopen optimizer/driver concerns and record newly identified numerical or target-interpretation issues. A verified correction is not verification of the stronger withdrawn claim. Frozen historical discrepancies remain identifiable.", "",
            "| ID | Categories | Resolution | Still unverified or historical |", "| --- | --- | --- | --- |"]
    for row in rows:
        text.append("| " + " | ".join(row[k].replace("|", "\\|") for k in ("id", "category_numbers", "resolution", "remaining_unverified_or_historical")) + " |")
    text.extend(["", "## Exact evidence and manuscript locations", ""])
    for row in rows:
        text.extend(["### " + row["id"], "", row["verified_scope"], "",
                     "Corrected source/manuscript: `" + row["corrected_source_or_manuscript"] + "`.", "",
                     "Implementation/configuration/results: " + row["source_evidence"] + ".", "",
                     "Required evidence or decision: " + row["required_experiment_or_decision"], ""])
    (OUT / "ISSUE_RESOLUTION.md").write_text("\n".join(text) + "\n")
    (OUT / "resolution_coverage.json").write_text(json.dumps(dict(original_unresolved_ids=sorted(actions),
        original_unresolved_count=len(original), output_rows=len(rows),
        all_original_ids_classified_once=True, full_posterior_auxiliary_GP_priority_portability_verified=False), indent=2) + "\n")
    print("Wrote", len(rows), "resolution rows; all 14 original unresolved IDs covered.")


if __name__ == "__main__":
    main()
