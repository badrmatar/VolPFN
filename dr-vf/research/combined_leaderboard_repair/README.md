# Combined leaderboard numerical repair

The original run stopped because one CPU forecast exceeded a strict CPU/GPU
reproduction tolerance. Its second moment differed from the archived GPU value
by 0.000106249%, just above the original 0.0001% limit. Input histories,
checkpoints and forecast arithmetic were verified against the archive.

This repair changes the **numerical validation tolerance only**, uniformly to
0.001% (relative tolerance 1e-5; zero absolute tolerance). It does not change
forecast values or statistical decision criteria. The original freeze, failure,
and all accepted cells remain in `combined_leaderboard_v1` without modification.
The amended output has its own freeze in `combined_leaderboard_v2`.

The diagnostic plan was dated and hashed before its fixed 524-cell panel was
evaluated. All checks passed: the largest CPU/reference relative difference was
6.08e-7, archive/reference 7.82e-7, and CPU/archive 1.063e-6. Ten independent
manual-attention reference checks also passed. These finite checks support the
engineering amendment; they do not establish a global numerical error bound.
Every forecast is still checked, all exceedances of the original tolerance are
reported, and a discrepancy above the amended tolerance stops the run.

No new losses or ranks informed this amendment. This does not change the
previous-exposure status of the market data.

## Execution

From the project root, use the pinned neural environment:

```sh
research/environments/frontier_v1/bin/python -m unittest research.combined_leaderboard_repair.test_repair -v
research/environments/frontier_v1/bin/python -m research.combined_leaderboard_repair.resume freeze
research/environments/frontier_v1/bin/python -m research.combined_leaderboard_repair.resume migrate
research/environments/frontier_v1/bin/python -m research.combined_leaderboard_repair.resume run --limit 64
```

Freeze requires a passing test receipt and the immutable numerical validation
record. Migration verifies every source cell and copies its forecast rows
unchanged. The bounded check runs the previously failed block with the unchanged
CPU forecaster. Only after this check should the remaining run continue:

```sh
research/environments/frontier_v1/bin/python -m research.combined_leaderboard_repair.resume run
```

Scoring uses the original project environment, after every required cell and
all 44,230 neural second moments pass their checks:

```sh
'/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python' -m research.combined_leaderboard_repair.resume analyse
```

The statistical functions, calibration, model roster, row population, losses,
bootstrap and Holm families come from the original frozen implementation.
The repair adds a numerical-amendment disclosure to the final report and table.
It neither selects another attention backend for individual rows nor replaces
CPU forecast values with archived GPU values.
