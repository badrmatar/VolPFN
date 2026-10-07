VolPFN native-tail comparison (5 October 2026)

Start with results/REPORT.txt and results/results_table.csv.
The six later-panel primary comparisons are identified by primary=True.
Secondary p-values and all per-market p-values are unadjusted.
The complete later panel had been inspected before this comparison.

Code
  tails.py          mixture CDF root and analytic/numerical lower-tail ES
  mixtures.py       corrected causal window inference; five-density combination
  diagnostics.py    Holm and Christoffersen statistics
  freeze.py         one-time protocol/input freeze (do not rerun on this release)
  run.py            inference, tail calculation, matched scoring and preservation
  test_native.py    eight small scientific/causality tests
  validate.py       replay archived neural-t8 scores and verify return units
  report.py         human-readable tables and interpretation

All original inputs and existing project files are read-only. The only output
location is this study folder. protocol.json and protocol.sha256 were written
before native scoring. source_seal.json was also written before native scoring;
the sealed calculation files remain unchanged. Later-added validate.py and
report.py perform replay checks and presentation, not changes to scoring rules.

Runtime
Use the VolPFN package at repository commit 459edb7551969b7824d0953484de3dd591f62cb6.
The archive includes that commit's core src/ packages under dependencies/src/.
The existing scientific Python environment was used, with NumPy, SciPy, Pandas,
PyTorch and an available MPS device. Exact versions are in results/runtime.json.
No training, checkpoint selection, fitting, market calibration or cloud job ran.

From the repository root, with its environment activated:
  python -m unittest discover -s experiments/native_tails_20261005 -p test_native.py -v
  python experiments/native_tails_20261005/run.py infer
  python experiments/native_tails_20261005/run.py tails
  python experiments/native_tails_20261005/run.py score
  python experiments/native_tails_20261005/validate.py
  python experiments/native_tails_20261005/run.py verify
  python experiments/native_tails_20261005/report.py

To reproduce without overwriting this completed run, use a separate complete
project copy and a fresh results directory. The protocol contains the actual
local input paths and hashes; checkpoints, market inputs, comparator archives
and the original checkout must be available for its preservation checks.
They are not silently downloaded or replaced. Results already include the
regenerated mixtures and daily native/guard VaR/ES pairs.

This is a local code-and-results delivery. Nothing was added to the dissertation,
and this study has not been pushed to GitHub.
