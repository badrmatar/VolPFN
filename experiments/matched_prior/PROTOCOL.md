# Matched simulator comparison — engineering protocol, 18 September 2026

## Question and scope

Hold the VolPFN network, eight inputs, density and auxiliary heads, causal
preprocessing, optimizer, update/position budgets and checkpoint rule fixed.
Change the simulation distribution. This tests a particular prior choice;
it does not isolate return feedback from every other distributional difference.
The first run is a small, synthetic-only engineering pilot. It cannot establish
market gains, novelty, or the adequacy of full-scale optimization.

Historical sources and the active five-seed replication are read-only. This
package imports their model and causal preparation without changing them.

## Three distributions

1. **Broad:** existing seven-family probabilities, unchanged.
2. **GARCH:** the existing random-parameter GARCH family with weight one.
3. **Independent Matérn latent scale:** stationary zero-mean Gaussian log
   variance, smoothness uniformly chosen from {0.1, 0.5, 1.5}, length scale
   uniformly chosen from {30, 120, 500} observations, and log-variance standard
   deviation uniform on [0.3, 1.5]. Draw the finite path from its covariance by
   Cholesky decomposition. No jitter, eigenvalue truncation, spectral embedding,
   or per-path centring/rescaling is used. Numerical failure is recorded.

The third arm spans rough, exponential-covariance and smoother paths. Its
parameters are fixed from the existing simulator ranges and this declared
smoothness extension, before its named pilot. They were not fitted to markets.
It is an alternative latent-scale prior, not a proxy for every general-purpose
foundation model or a faithful reproduction of ForecastPFN/KernelSynth.
The broad prior already contains independent rough paths and some independent
SV paths. The alternative's unconditional variance level, volatility dynamics
and initialization are not matched to all broad components. In particular a
zero-mean Gaussian log variance implies mean variance greater than one; this
matches the convention of the existing rough component, not GARCH's level.

All arms use the same innovation-law distribution and observation-overlay
probabilities. Random draws need not be identical across different generators.
Overlays are not disabled for GARCH. The shared overlay code supplies an
auxiliary variance label; with combined AR and stale-price overlays it omits
cross-day covariance. Consequently this label is not advertised as exact
observed-return conditional variance. Clean Matérn returns have conditional
variance exp(logv) given their latent scale and parameters. No exact observed-
history posterior predictive or Bayes-error certificate is claimed here.

## Pilot allocation and matching

Two initialization seeds, 9101 and 9102, each run once under all three arms.
Width64, two blocks, four attention heads, four Student-t components. Each run
gets 200 updates of 512 input positions. Length64/128 is selected with equal
probability from a deterministic schedule shared across arms within each seed.
Every arm starts from identical weights within a seed; the state hash is logged.
AdamW, lr0.0003, betas(0.9,0.98), weight decay0.01, epsilon1e-8, warmup20,
cosine decay to10%, gradient clipping1, auxiliary weight0.1; float32, CPU,
one torch thread and one BLAS thread. There are no data-loader workers.

Use corrected within-window prefix normalization from seed_replication.causal.
The target's value never enters its own forecast units. Scored masks and the
raw-density scale Jacobian are retained. Position counts match exactly;
scored-target counts can differ because of stale observations.

The **final update is the sole checkpoint**. There is no fallback to a best
validation checkpoint after divergence. Shared synthetic panels for all three
arms use independent seeds 99101/99102/99103, 12 tasks each, length128, and
burn32. All six models see these exact panels at initialization and after
100/200 updates, as diagnostics only. Report scores separately by generating
arm and their equal-arm average. They select neither model nor checkpoint.
Per-seed arm order rotates, but two seeds do not balance three positions.
Runtime is descriptive while other research is running, not an efficiency claim.

## Recording, tests and engineering gate

Before named training: snapshot and hash local package code/protocol, shared
model/prior/causal/training code and the complete configuration. Record runtime
versions. Refuse overwrite of named runs. Verify hashes before every run and
after the battery. Record failures and interruptions without replacement or
automatic retry. Any failed run prevents a complete-battery headline.

Tests cover covariance reconstruction, innovation variance, deterministic
simulation, matching of initialization and sequence budgets, overlay retention,
prefix causality, masked target alignment, checkpoint reload, failure retention,
and the fixed-final rule. The engineering gate requires six complete finite
runs, matched within-seed initialization and allocation, reloadable final
checkpoints and causal probes. A fall in validation loss is reported but is not
a scientific success test. Include synthetic distribution diagnostics (levels,
log-variance dispersion, persistence, roughness and overlay/law frequencies).
These diagnostics reveal differences; they do not certify perfectly matched
priors or authorize tuning against market scores.

## Full-scale study: not launched by this pilot

The runner can freeze a separately labelled full training battery: three fresh
seeds 9201–9203, base network (width256, six blocks, eight heads, four components),
20,000 updates, 8,192 positions/update, lengths512/1024 with probabilities0.4/0.6,
warmup300 and the same fixed-final rule. This would be nine runs. Full training
must be a separate explicit invocation; a pilot never automatically launches it.
It does not reuse selected checkpoints from the active replication, whose
validation-based selector differs. Full hardware/runtime must be measured;
short CPU pilot throughput is not an estimate of base MPS runtime.

Before market scoring, freeze a separate market protocol and exposure inventory:
primary broad-minus-Matérn and supporting broad-minus-GARCH contrasts, identical
dates/units/masks, common-t8 second-moment scoring, QLIKE and native densities in
their stated roles, materiality and multiplicity rules, failures, and uncertainty
over paired training seeds and synchronized calendar blocks. The proposed
0.005-nat/20-of-28 criteria are not activated by this engineering protocol.
With only three paired seeds, training-variability inference is limited; do not
treat thousands of dates as thousands of training replications. Already examined
2011–2014 and 2015–February2022 periods are not untouched confirmation.

If broad wins, the conclusion concerns these particular simulator substitutions.
If it does not, that is not proof that all priors are equivalent. External-model
benchmarks answer a different question and remain separate work.
