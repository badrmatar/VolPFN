POST-HOC native-tail extension — 5 October 2026

Read results/REPORT.txt or results/results_table.csv for the single nine-row
comparison table. Read results/invalid_forecast_paths.csv for ES-CAViaR failures.
The previously inspected later panel is reused. This is not independent confirmation.

The POST_HOC_PROTOCOL.json file was frozen before market fitting/scoring and has
not changed. It is separate from the previous six-comparison protocol. All prior
files remain unchanged. New GJR model fits follow the existing Tier A defaults;
ES-CAViaR uses the old GP-FZ fit-once implementation without retuning. The copies
in legacy_gp_fz/ are byte-identical to their originals. Numba only accelerates
the same recursion; all 30 candidate results in a full-fit parity check matched.

Code
  models.py       GJR/FHS adapters; original ES-CAViaR fit with diagnostic capture
  run.py          fitting, exact-date scoring, nine-slot Holm and preservation
  test_models.py  causality and exact numerical parity tests
  audit.py        invalid forecast-path inspection and independent Holm check
  report.py       single results table and failure report
  freeze.py      one-time input/protocol freeze; do not rerun on this release

To reproduce in a separate complete project copy with a fresh results directory,
activate the existing VolPFN environment and run from the repository root:
  PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/native_tails_posthoc_20261005 -p test_models.py -v
  PYTHONDONTWRITEBYTECODE=1 python experiments/native_tails_posthoc_20261005/run.py fit --workers 4
  PYTHONDONTWRITEBYTECODE=1 python experiments/native_tails_posthoc_20261005/run.py score
  PYTHONDONTWRITEBYTECODE=1 python experiments/native_tails_posthoc_20261005/audit.py
  PYTHONDONTWRITEBYTECODE=1 python experiments/native_tails_posthoc_20261005/run.py verify
  PYTHONDONTWRITEBYTECODE=1 python experiments/native_tails_posthoc_20261005/report.py

The protocol records actual local input paths and hashes. Original market arrays,
prior native-tail output and the original checkout are required for the complete
preservation checks; no inputs are downloaded or silently replaced. Dependencies
and reused native-tail inputs are included in the ZIP for inspection. They do not
remove the original-path requirements. Runtime versions are recorded in results/.
No existing result, script, dissertation source or GitHub file was changed.
