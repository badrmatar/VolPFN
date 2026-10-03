# Real-market forecast-distribution diagnostic

This stage asks whether simple established HAR controls explain the earlier simulated calibration failures. It adds no neural architecture and makes no novelty claim.

- `PROTOCOL.md`: design fixed before this run's forecast scores, model/source distinctions and interpretation limits.
- `models.py`: Bayesian and residual-bootstrap log-HAR, Gaussian and empirical-residual log-HAR-GARCH, and the published Gaussian square-root HAR-GARCH specification.
- `run.py`: causal rolling fits and 4,096 simulated joint daily paths, separately for both realized measures on all 28 development markets.
- `analyze.py`: quantile scores, coverage, chronological and volatility subsets, per-market results, and shared-date block uncertainty.
- `../test_market_density.py`: independent numerical and information-ordering checks.

From the project root, using the existing environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../venv/bin/python -m unittest research.test_market_density -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../venv/bin/python -u -m research.market_density.run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../venv/bin/python -m research.market_density.analyze
```

The runner refuses to overwrite its output directory. For an explicitly requested replication, pass `--output research/outputs/market_density_replication` and call `summarize()` with that directory. Do not remove prior results to rerun in place.

Outputs are in `../outputs/market_density_v1`: forecast quantiles and scores, fit states, Monte Carlo repeats, source/input hashes, completion status, summaries and the research decision. Source snapshots preserve the executed version. The runner reads the already separated development files only; the reserved 2015+ period is not accessed.

These are distributions of **future realized variance sums**, not return distributions or loss VaR. Coherent here means every horizon is calculated from the same daily path. It does not itself imply calibration, accuracy or novelty. Calendar-date blocks keep markets exposed to the same dates together; treating every market-origin as independent would exaggerate precision.
