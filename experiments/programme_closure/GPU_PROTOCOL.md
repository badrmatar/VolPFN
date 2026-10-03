# Programme closure: corrected matched-prior CUDA amendment

19 September 2026. This is a prospective operational amendment to the hashed
18 September five-pair specification, not an original pre-exposure registration.
The seed8101 historical foundation-model result has now been inspected, as have
the recorded failure and earlier exploratory studies. The previous protocol,
sources, checkpoints, failures, forecasts and reports remain archived unchanged.

## Fixed scope and stopping rule

Close two distinct questions: (1) the recorded volatility forecaster versus the
specified benchmarks; (2) the contribution of substituting the specified broad
training prior for the specified independent Matérn prior in an otherwise matched
forecaster. No architecture changes. No new mechanisms, adaptive extensions,
GARCH-only training, additional populations or further seed search are authorized
by this closure protocol. The previously reserved broad-minus-GARCH superiority
slot remains unavailable with p=1. Stop the research programme after the registered
analyses and claim corrections, whether positive, negative or inconclusive.

## Numerical correction and checkpoint reuse

The archived seed8103 failure is reproducible at update4575: subtracting nearby
large cumulative sums in the stale-price overlay loses a positive local variance.
The correction accumulates each observed interval directly, and combines its
log variances with logaddexp, without clipping, discarding draws or resampling.
The same overlay code applies to both priors. It preserves the intended sampled
law and random draws, but floating-point arrays and later optimization can differ.
The inherited AR/stale auxiliary-label approximation is unchanged and disclosed.

The two completed MPS broad finals remain valid historical trained objects and
the seed8101 benchmark remains its own result. Neither is reused as a member of
the corrected CUDA primary: ten fresh trainings prevent prior arm from being
confounded with numerical implementation or backend. Seeds8101–8105 are retained
as identifiers; fresh training is from their original initialization procedure,
not from old checkpoints. No claim that all experiments are newly unexposed.

## Matched training and timing-only gate

Both arms use the same original base network (width256, six blocks, eight heads,
four Student-t components, eight causal channels, auxiliary weight0.1), 20,000
AdamW updates, 8,192 input positions/update, length512/1024 probabilities0.4/0.6,
and the original learning-rate schedule, normalization, loss, innovations and
overlay probabilities. Fixed final20000 only; no validation selection. Identical
initial weights are generated on CPU within each seed pair and their hashes are
recorded. Both arms run on the same GPU type and numerical software stack.
CUDA float32 without autocast or TF32; deterministic algorithms enabled. These
are implementation choices, not an architectural contribution. No compute/size
change is made after looking at losses. Equal update/position budgets, not equal
wall-clock runtime, define matched compute allocation.

The ordered two-worker, four-prefetch data pipeline assigns every batch by seed
and update; process scheduling does not select or drop samples. Validation uses
the shared broad/GARCH/Matérn panels at initialization and every500 updates for
health only, retaining the within-panel comparisons. CPU generator thread limits
and cached Matérn factors are identical across runs. Record runtime and actual
initial-state hashes. Both length choices must fit within available memory.

Before the full battery, run a separate synthetic timing pilot: seed7103,350
updates per arm at full model/batch size, first50 excluded from throughput.
Time both lengths, generator waits, one full validation pass and checkpoint save.
Pilot weights are never reused. Estimate the ten-run wall time using the fixed
length probabilities, all validation/checkpoint costs, and a stated contingency;
separately disclose setup/export and market-evaluation uncertainty. Record the
live hourly GPU price. Report this estimate to the user before full paid launch.
The benchmark does not automatically start the battery. Its provisional paid
allocation is one existing RTX4090 at its live quote for at most30 minutes; stop
on completion or failure. If it cannot finish in that allocation, report partial
timing rather than silently extending it. Never terminate existing pods.

The timing runtime receipt must match every full run (Python, NumPy, SciPy,
PyTorch, CUDA and GPU type); source hashes are rechecked. The concrete CUDA stack
is recorded before benchmarking and frozen before full training. A stack change
requires an explicit operational record, not an unnoticed substitution.

Run paired arms in fixed alternating order: broad then Matérn for8101, reverse
for8102, and alternate through8105. Preserve every failure. No automatic retry,
replacement seed, result-dependent early stop or survivor-only headline.
Intermediate checkpoints every2500 updates permit a separately documented
technical-interruption continuation; a numerical training failure is investigated
and retained, not retried until success. Full-launch tooling requires a receipt
recording approval after the timing/cost report and an operational spend/deadline
plan. It does not claim that a local receipt itself limits cloud billing.

## Data and inference: unchanged scientific endpoints

The prescribed primary remains daily open-to-close returns on eligible members
of the fixed28-index population, 2022-03-01 through2026-08-31. Acquire and hash
the source identity evidence, raw OHLC, QC, complete causal context, row mask and
all known earlier data contact before model scoring. Keep the original mechanical
QC and minimum20 eligible markets; no ETFs, index substitutions, close-to-close
fallback or relaxed thresholds. Data unavailability is reported, not repaired by
promoting the exposed historical secondary. The amended GPU procedure is not
claimed to have been frozen before the seed8101 benchmark was seen.

Use the existing common-t8 log-score contrast computed from each trained mixture's
second moment, L1024/stride512 causal inference and the original frozen rules:
five equally weighted seed effects and equal market weighting; 5,000 synchronized
circular calendar bootstrap draws, block60, seed9182601; two-slot Holm with the
unrun GARCH slot p=1; materiality0.005 nats, positive95% interval and the original
breadth threshold ceil(20*N/28). Report all gates separately, every seed and range,
and conditional calendar uncertainty. No seed-population inference is implied.
Separate equivalence endpoint: TOST and90% interval within ±0.003 nats. A negative
superiority result does not establish equality. QLIKE/native densities and the
inspected2015–2022 block remain supporting/descriptive as originally specified.

## Existing benchmark and minimum procedure confirmation

Preserve the complete April2015–June2016 foundation-model study, its forecasts,
full-sample statistical report and interim-exposure disclosure. Simulator repair
does not change an already trained checkpoint's inference or erase those results.
Do not rerun expensive foundation-model inference merely because training changed.

For the corrected procedure, evaluate all five broad finals against the saved
competitor forecasts on exactly the existing8,774 rows, using the frontier's
per-origin1024 context (not the matched-prior stride512 adapter). Refit each
VolPFN scalar using exactly the archived development-only three-market72-origin
calibration rule; freeze the rule and origins before scoring. Competitor scales
stay unchanged. Report every seed and equal-seed mean score performance, not a
new density ensemble. Primary procedure-sensitivity metric is fixed-scale QLIKE;
three comparisons, Holm3, date-block10 with5/20 sensitivity, same confidence
construction as the original analysis. This is exposed historical confirmation
conditional on the five trained models, not a new untouched superiority study.
Keep the original and corrected-procedure results in separate labelled rows;
never replace the original with whichever wins. No architecture or causal
financial-mechanism claim follows from either comparison.

## Writing commitment

After analysis, produce claim/result/limitation/dissertation-wording evidence
rows and complete the writing checklist. Part II is closed by matching claims
to executed estimators and proved assumptions; unimplemented guarantees are
narrowed or removed. No new theory is required to finish. A positive prior result
supports this specified training-prior substitution; a negative or inconclusive
result narrows the conclusion without triggering another mechanism search.
