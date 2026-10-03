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
