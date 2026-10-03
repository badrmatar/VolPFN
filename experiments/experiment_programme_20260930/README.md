# Experiment programme — 30 September 2026

This is new, separately identified evidence. Original data, checkpoints, manuscript versions and historical results remain unchanged. Existing empirical panels have already been inspected: these experiments are **retrospective historical confirmations**, not fresh empirical holdouts. Tiny smoke runs and hardware pilots never enter final results.

## Four matched procedures

| Arm | Training and output | What the comparison can establish |
|---|---|---|
| `full` | Corrected broad synthetic prior, four-component return density, Gaussian log-variance auxiliary NLL at0.1 | Matched control |
| `noaux` | Identical synthetic batches, initialization and density objective; auxiliary weight0 | Contribution of auxiliary training within this procedure; unused head retained solely to preserve initialization |
| `historical` | Same no-aux density model; uniformly sampled markets and contiguous pre-2011 return windows | Synthetic versus historical training procedure; diversity, repeated observations and task distribution differ |
| `direct` | Same initial transformer trunk and synthetic batches; positive second-moment output trained with squared-return QLIKE | Point-output procedure versus density procedure; output and objective both change |

Every arm requires seeds8101–8105 and20,000 updates of8,192 input positions. Final checkpoint selection is fixed; synthetic diagnostics do not choose a checkpoint. Native return densities are available only for mixture arms. The issued quantity for squared-return QLIKE is **second moment**, central variance plus mean squared. Auxiliary labels are not exact total conditional moments for every simulator overlay combination; auxiliary benefit is not presumed.

## External comparisons

Three independently tested, explicitly labelled daily-return adaptations are executable: ARCH–LSTM2026 (fixedν author-code branch), AAAI GARCH–LSTM2024 (equation implementation) and GINN2024 (frozen training-only AR/GARCH teacher). Their native papers use different return definitions and/or targets; these implementations forecast the common next observed open-to-close session. This is not an exact replication of every published result.

Training ends2010-12-31; per-market validation selection uses2011–2014; final evaluation uses all50,073 retained28-market rows from2015 through2022-02-25, including zeros. Each market has a registered finite learning-rate/loss-weight grid,50-epoch cap and8-epoch patience. Scaling and teacher parameters are fitted on TRAIN only. All five final seeds are required. The fitted baselines are per-market procedures; VolPFN is pooled/frozen, so this external comparison does not isolate architecture alone.

RECH2022 is unavailable: the public MATLAB source implements a later specification and does not establish the finite posterior moments required by this endpoint. DCC-GNGARCH belongs to a separate graph/information track, needs additional data, and its native test rescaling is anticipative. Neither is replaced silently. See `RECH_SOURCE.md`, `DCC_SOURCE.md`, `ARCH_LSTM_SOURCE.md` and `DAILY_MODEL_SOURCES.md` for primary evidence and exact boundaries.

## Execution and receipts

Use the pinned CUDA2.8.0/cu128 template runtime, float32, deterministic algorithms, noTF32/autocast. Install `requirements.txt` in an isolated environment while retaining templateTorch. Choose hardware from synthetic timing only, then keep the selected stack/GPU type fixed. The new H100 and RTX5090 pilots both projected approximately15 minutes per model; setup, diagnostics, checkpoints, inference, transfer and statistical analysis add time.

1. `ablations freeze --out NEW_COMPONENT_FREEZE --runtime-receipt PILOT_RESULT` freezes corrected source, data copies and runtime.
2. `programme freeze --out NEW_PROGRAMME_FREEZE --ablation-manifest COMPONENT_MANIFEST` binds all method sources, population, dates and masks.
3. `managed bundle --manifest PROGRAMME_MANIFEST --ablation-manifest COMPONENT_MANIFEST --out NEW_TAR` produces a credential-free portable bundle.
4. `worker --manifest PROGRAMME_MANIFEST --ablation-manifest COMPONENT_MANIFEST --arm ARM --seed SEED --out NEW_WORKER_DIR` runs one final training job and its automatic frozen-mask evaluation. Each shard is incomplete evidence until all required seeds have been collected.
5. `daily_runner tune`, `train`, `evaluate` implement each registered market/model comparison. `managed plan/execute` provides a finite sequential alternative; infrastructure/parallel dispatch is controlled separately.
6. Collect complete forecasts, then `statistics --manifest PROGRAMME_MANIFEST --forecasts FORECAST_MAPPING_JSON --out NEW_ANALYSIS` checks exact masks, units, moments and score formulas before synchronized-calendar uncertainty analysis.

All commands above are modules under `python -m research.experiment_programme_20260930.MODULE`. Manifest and output directories must be explicit and fresh. Failures are preserved; no silent seed replacement, fallback estimator, reduced mask or surviving-seed headline. Source or data drift fails verification. Component training can explicitly resume only from a matching checkpoint/design/runtime; a completed seed cannot be retrained in place.

Outputs live under `research/outputs/experiment_programme_20260930/`. The controller must download and verify results before stopping/removing newly created GPUs. A completed Python process alone does not stop Runpod billing. Existing unrelated pods are not altered.

## Interpretation

Primary paired gains are comparatorQLIKE minus referenceQLIKE, equal-weighted across markets and seeds. Use20,000 synchronized circular-calendar resamples, block60 with20/120 sensitivities, fixed RNG2026093001, two-sided centered plus-one probabilities, basic95% intervals and Holm within the registered three-slot component and four-slot external families. Unavailable slots retainp=1. Separate crossed-seed/calendar results are exploratory because five seeds provide limited information about training variability. No multiplicity guarantee spans both families, all secondary endpoints or prior study selection.

Common zero-mean t8 density is an imposed scale adapter. Realised kernel uses decimal²×10,000 and a separately frozen positive finite mask; its equivalence to the open-to-close return second moment remains unverified. Report original outputs, native cap events, seed variability, failures and changed conclusions. Do not claim universal superiority, economic benefit or first-ever novelty from rankings alone. Do not propagate unfinished results into the dissertation.
