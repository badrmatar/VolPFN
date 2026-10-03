# Causal training replication

The frozen specification is in `PROTOCOL.md`. This package preserves all earlier study files and implements a separate training and inference procedure.

The run order is: synthetic tests; freeze source/input hashes; two small-model engineering runs; five base seeds; market evaluation of every completed seed; crossed seed/calendar inference. All five base seeds become terminal before the first market evaluation. Named failed or interrupted seeds are retained and are not replaced.

The current output directory is `research/outputs/seed_replication_v1`. It includes `manifest.json`, top-level `status.json`, `pilot_summary.json`, per-seed status/training/validation/checkpoint files, and later the `evaluation/` caches and `seed_mean_results.json`. The final `REPORT.md` is written automatically when evaluation and aggregation finish.

Check progress from the project directory:

```sh
cat research/outputs/seed_replication_v1/status.json
tail -n 12 research/outputs/seed_replication_v1/run.log
```

Each active seed also writes `base/<seed>/status.json` and `training.jsonl`. Their input-position counters distinguish successful updates from attempted work. Completion includes finite-model checks, but scientific success requires the separately specified statistical tests.

The launcher is `python -m research.seed_replication.run freeze` followed by `python -m research.seed_replication.run run`, using the project environment with torch and the frozen MPS backend. Do not launch another process while the current one is active. Completed batteries refuse a rerun; a changed specification needs a new output directory and freeze. There is no exact mid-seed resume because generator/optimizer/RNG progression is not reconstructed after interruption.

The retained transfer run also has descriptive leave-one-market-out aggregation tables at `research/outputs/seed_replication_v1/transfer_leave_one_out.md`. No model was refitted and no original panel member was removed to produce them.

These are new comparisons on previously researched markets. Causal normalization does not remove prior research exposure; five seeds do not reproduce uncertainty from the entire historical model search. A successful engineering pilot is not evidence of forecasting improvement.
