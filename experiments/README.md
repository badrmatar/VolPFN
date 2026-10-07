# Dissertation experiments

These modules organise the dissertation's training comparisons, baselines and analyses. The VolPFN implementation is in `src/volpfn/`; tests are in `tests/`.

Run experiment modules from the repository root after installing the package, for example:

```sh
python -m experiments.programme_closure.gpu_training --help
python -m experiments.experiment_programme_20260930.programme --help
```

Reading `--help` does not start a study. Actual experiments may need data, checkpoints, frozen manifests and the original runtime. Use a new output location for new work and retain the appropriate protocol and provenance. Some historical runners still depend on original archive paths and hashes; see `docs/SCOPE.md` before replaying them.

The `seed_replication/causal.py` and `training.py` files are short import bridges to the core package. There is one active implementation of each. The `baselines/` directory contains comparison methods and the retained switching-study helpers.

The dated directory names distinguish experiments with different populations, settings and scientific interpretations. They are preserved to avoid presenting separate procedures as one interchangeable benchmark.

## Additional tail experiments — 5 October 2026

The completed native-tail study and its post-hoc extension are now included with their saved results:

- [Native-tail report](native_tails_20261005/results/REPORT.txt) and [result table](native_tails_20261005/results/results_table.csv): exact mixture VaR/ES, archived comparator comparisons, and coverage diagnostics.
- [Additional-model report](native_tails_posthoc_20261005/results/REPORT.txt) and [result table](native_tails_posthoc_20261005/results/results_table.csv): GJR-GARCH-t, GJR-FHS and ES-CAViaR comparisons, including unavailable comparisons and forecast failures.

Both folders include the original calculation code, frozen protocols, source seals, delivery manifests, daily forecast arrays and audit records. All files in the original delivery manifests are preserved byte-for-byte. Statements in those dated reports about being local-only describe the October 5 delivery; the folders were published to GitHub on October 7, 2026. The original README references to ZIP dependencies describe the separate local delivery archives, not extra files bundled in this repository.

These studies reuse an already inspected panel and are not independent confirmation. The post-hoc extension retains its separate nine-comparison adjustment and invalid ES-CAViaR paths. Publishing the saved studies does not rerun training, fitting or scoring, or update the dissertation. Full replay still requires the original inputs and paths recorded in each protocol; a fresh clone alone is insufficient.
