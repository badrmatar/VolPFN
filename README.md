# VolPFN

Volatility forecasting from a synthetic prior. This repository contains the model, the experiments used in the MSc dissertation, and the associated tests.

**Start with [the model](src/volpfn/model.py), [the synthetic prior](src/volpfn/prior.py), or [the forecasting interface](src/volpfn/forecasting.py).**

## Structure

```text
src/
  volpfn/            model, prior, preprocessing, losses, training and forecasting
  calibeat/          VaR/ES recalibration used in the tail-risk study
experiments/         dissertation comparisons, baselines and analysis procedures
tests/              core regression tests, fixtures and study-specific tests
```

The main code no longer lives in a research/testing folder. Experiment code calls the package; the package does not depend on the experiment folders.

## Model files

| File | Responsibility |
| --- | --- |
| [`model.py`](src/volpfn/model.py) | Transformer blocks and mixture prediction heads. |
| [`prior.py`](src/volpfn/prior.py) | Corrected synthetic prior and reference filters. |
| [`data.py`](src/volpfn/data.py) | Raw synthetic training batches. |
| [`preprocessing.py`](src/volpfn/preprocessing.py) | Causal normalisation, features and target alignment. |
| [`distributions.py`](src/volpfn/distributions.py) | Student-t densities, moments, VaR and ES. |
| [`losses.py`](src/volpfn/losses.py) | Predictive and auxiliary training losses. |
| [`training.py`](src/volpfn/training.py) | Seeded training, validation and checkpoint utilities. |
| [`inference.py`](src/volpfn/inference.py) | Corrected causal forecasts from an in-memory model. |
| [`forecasting.py`](src/volpfn/forecasting.py) | Checkpoint validation and the forecasting command. |
| [`legacy/`](src/volpfn/legacy/) | Explicit support for retained older checkpoints and comparisons. |

The small `_causal.py` module and two experiment import bridges retain the original experiment interfaces without duplicating the implementation.

## Install

Python 3.9 or later is required. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the additional common baseline dependencies, use `python -m pip install -e '.[experiments]'`. The original tested dependency versions remain in `requirements.txt`. Specialist pretrained-model adapters still require their separately specified runtimes and weights.

## Forecast

```sh
volpfn --help
volpfn --ckpt path/to/checkpoint.pt --data returns.csv --out forecasts.csv
```

`python -m volpfn` provides the same command. Input CSVs require increasing, unique `date` values and a finite `return` column. Supply the units expected by the checkpoint; the dissertation's daily target is open-to-close log return in percent units. The command records preprocessing and output provenance. Older checkpoints require `--allow-legacy` explicitly.

For library use:

```python
from volpfn.forecasting import forecast

predictions, provenance = forecast("checkpoint.pt", returns, device="cpu")
```

Checkpoints and market datasets are not bundled. No training or cloud job starts on import or installation.

## Tests

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s tests/tail -p test_fz.py -v
```

The core suite checks causal timing, checkpoint dispatch, simulator corrections, reference calculations and numerical parity with the version before reorganisation. `tests/experiments/` holds additional study-specific checks; some require archived inputs and are intentionally separate from core test discovery.

## Dissertation experiments

See [`experiments/README.md`](experiments/README.md). The scientific algorithms and settings are retained, while imports, test locations and source inventories have been updated. The October duplicate implementation and downloaded third-party source tree have been removed; required implementation code is shared instead.

Historical hash-bound protocols identify their original source versions. Exact historical replay uses the recorded inputs and the pre-reorganisation commit `7cde0f7`, not newly generated hashes presented as old provenance. Existing absolute archive paths and unavailable datasets still limit which historical runners work in a fresh checkout. Reorganisation does not recreate those inputs or certify all archived workflows.

The dissertation itself is submitted separately and is not included in this repository.
