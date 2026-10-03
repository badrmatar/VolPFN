# VolPFN

Code and supporting material for **VolPFN: Volatility Forecasting from a Synthetic Prior**, Bader Matar's MSc dissertation.

## Contents

| Folder | Purpose |
| --- | --- |
| [`dr-vf/`](dr-vf/) | VolPFN, its simulator, causal forecasting interface, dissertation experiments and their tests. |
| [`calibeat-var-es/`](calibeat-var-es/) | The VaR/ES recalibration implementation used in Chapters 6–7. |
| [`dissertation/`](dissertation/) | Active LaTeX source, figures and bibliography. |
| [`docs/`](docs/SCOPE.md) | Chapter-to-code map, provenance, reproduction limits and recorded test results. |

[Find the relevant implementation](docs/SCOPE.md)

The original project folder names are retained because imports, relative paths and the dissertation's evidence references use them. This is a focused source repository; it does not contain the full local research archive.

## Core setup

The recorded core checks use Python 3.9 and the package versions in `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements record the working scientific environment; a fresh installation was not tested during publication. External pretrained-model adapters have their own runtime and model requirements in the relevant research protocols. No cloud jobs are started by setup.

## Forecasting

Use the checkpoint-aware interface, which checks the declared preprocessing and records forecast provenance:

```sh
cd dr-vf
python -m research.forecast_checkpoint --help
```

Forecasting requires a compatible trained checkpoint and an input return series. The dissertation's daily target is the next observed session's open-to-close log return in percent units. The root `volpfn.py` CLI is a retained legacy interface; use `research.forecast_checkpoint` for corrected causal checkpoints.

## Tests

From `dr-vf/`, run the selected self-contained checks:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m unittest \
  research.test_forecast_checkpoint \
  research.test_volpfn_causality \
  research.test_reference \
  research.seed_replication.test_causal \
  research.programme_closure.test_simulator_correction -v
```

From `calibeat-var-es/`, run the tail-score checks:

```sh
python -m unittest discover -s tests -p test_fz.py -v
```

Other historical tests and experiment drivers may require the separately retained data, checkpoints or saved results. They are not all clean-clone smoke tests. No dissertation training or market experiment was rerun for this repository.

## Build the dissertation

With TeX Live or MacTeX and the packages declared in `dissertation/main.tex`:

```sh
sh dissertation/build.sh
```

The LaTeX source corresponds to the final Stage 4 delivery, with the scientific revision dated 2 October 2026 and the editorial revision dated 3 October. The active inputs, bibliography and referenced figures were copied from that delivery without changing their contents. The build helper uses the installed TeX distribution rather than bundling one.

## Scope and reproducibility

The selection follows the dissertation's evidence map and the dependencies of its experiments. It includes the corrected simulator, matched-prior procedure, benchmark and component comparisons, compact-rule analysis, tail-risk study and October follow-ups. The October frozen source bundle is retained because its protocol and import paths explicitly depend on that version.

Datasets, checkpoints, full forecast tables, caches, old manuscript copies, interview material, editorial tooling and unrelated exploratory projects are not included. Compact October result tables and protocol records accompany their code. Some frozen scripts still record original absolute paths and source hashes; the [scope notes](docs/SCOPE.md) explain what is needed before reproducing those runs.
