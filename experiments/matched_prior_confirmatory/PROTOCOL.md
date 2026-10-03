# Five paired prior substitutions — prospective design lock, 18 September 2026

This specification must be hashed before any designated replication market
output exists or is viewed in this task, and before any Matérn training starts.
It succeeds the dated full-study addendum and replaces its nine-fresh-run draft.
The engineering pilot and the original replication retain their own protocols.
No historical preregistration of this new contrast is claimed.

## Allocation, objects and chronology

The primary contrast is five broad-minus-Matérn pairs labelled by seeds8101–8105.
The broad objects are the existing replication's completed **final.pt at exactly
20,000 updates**, not validation-selected best.pt. The five new Matérn counterparts
use the same seed, initialization procedure, architecture, optimizer, objective,
normalization, precision, length schedule and update/input-position budgets.
The prior distribution is the specified substitution. No new label correction
is introduced: the inherited AR/stale auxiliary-label approximation is held fixed
and disclosed. It is not an exact observed-history variance certificate.

The audit compared300 full-size seed8101 batches from the actual two-worker,
four-prefetch generator against the new ordered generator: all seven arrays
match byte-for-byte, over2,457,600 input positions. All20,000 schedule entries
match. Frozen-source and completed-final provenance checks passed. This is a
data/configuration compatibility result, not replay of full optimization.
No archived step-zero weights exist. Initial-state hashes are reconstructed
from frozen code, seed and configuration; record this qualification for every pair.
Each broad final must independently pass completion/step/configuration/checksum
checks before reuse. Rehash every relevant code path before each Matérn launch
and after each run. Source drift stops the battery; do not reinterpret it silently.

All five original seeds are included in advance. Missing/failed counterparts
make the complete five-pair primary endpoint unavailable; do not average survivors
or substitute seeds. Individual completed runs remain descriptive. If the original
broad battery cannot provide all five valid finals, stop before the expensive
Matérn battery and record the unmet prerequisite. Broad market-evaluation failures
alone do not invalidate otherwise valid trained finals.

The broad run is already in progress. Do not interrupt it or falsely describe
the resulting chronology as alternating. The new job waits for the original
replication process to reach its terminal state, checks all broad finals, then
trains Matérn8101,8102,8103,8104,8105 in order. Training order is consequently
blocked by arm; no causal runtime comparison is claimed. Pairing is by matched
seed/schedule, not simultaneous execution.

GARCH-only counterparts8101–8103 are a predeclared optional supporting extension,
not part of this launch. Its reserved superiority slot is p=1 unless all three
complete with matched settings on the same data mask. Availability of resources,
not observed gains, may activate it in a dated operational record. No automatic
GARCH training or outcome-dependent addition of seeds occurs.

## Matched training

Base VolPFN: width256, six layers, eight attention heads, four Student-t components,
eight causal inputs and auxiliary log-scale head with weight0.1. Each new model
receives20,000 updates and8,192 positions/update; lengths512/1024 have probabilities
0.4/0.6 from the identical seed-derived schedule. AdamW lr0.0003, warmup300,
cosine decay to10%, betas(0.9,0.98), weight decay0.01, epsilon1e-8, clipping1;
MPS float32 without autocast. No backend substitution, retry, best-step fallback
or outcome-dependent early stopping. Final update only.

The Matérn bank is unchanged: independent stationary Gaussian log variance,
mean0, standard deviation uniform[0.3,1.5], smoothness uniform{0.1,0.5,1.5},
length scale uniform{30,120,500}; Cholesky sampling without jitter or eigenvalue
clipping. Existing skew-t/Gaussian innovation mixture and all overlay probabilities
are identical to the broad procedure. This arm has observation jumps and skewed
innovations. The experiment does not isolate feedback, breadth or financial
knowledge from other differences between these specified priors.

Shared synthetic diagnostic panels: independent seeds99101/99102/99103 for
broad/GARCH/Matérn,48 paths per panel, length1024, target burn100. Retain and hash
the generated arrays at design freeze. Validate Matérn at initialization and every
500 updates, without selection. Score broad finals on the same panels only after
the lock. Retain the full matrix when rows are available; unavailable GARCH row
remains unavailable. Synthetic secondaries are within-column breadth costs:
NLL(broad on j) minus NLL(j-specialist on j), for j=GARCH and Matérn. Do not compare
raw diagonal NLL values across different data-generating distributions as gains.
Per-seed values and conditional whole-task uncertainty remain separate from the
market primary. No synthetic outcome changes the locked priors or allocation.

Record per-arm distributions of sample squared-return ACF at lags1,5,22 and
log-variance dispersion/persistence; use full training lengths and distinguish
finite-path uncertainty from population claims. Some tasks lack finite fourth
moments, so add log-transformed squared-return diagnostics and treat raw squared-
return ACF as descriptive. Coverage overlap is not equality of prior distributions.

## Fixed primary data block and data-only eligibility

Primary dates are **2022-03-01 through2026-08-31**, inclusive. The candidate
population is the exact28 retained index identities in the hashed exposure
inventory. Target is daily open-to-close return,100*log(close/open), in percent.
There is no close-to-close fallback or ETF/related-index substitution. Confirm
each vendor instrument's exact index identity and date/price conventions before
acquisition. Unresolved identities are unavailable, not replaced by a convenient
symbol. Seal the mapping and evidence before fetching that index's new snapshot.

Later raw OHLC snapshots do not yet exist in this project inventory. This is a
fixed prospective design, not a claim that later data have passed their checks.
Earlier Yahoo whole-history audits touched at least four index histories. Neither
a new download nor the date2022 makes a period untouched. Record all known prior
contact and any additional contact found during acquisition.

The executable data_gate.py fixes these pragmatic rules before scoring:

- Unique increasing local session dates; positive finite OHLC and internal
  low/open/close/high consistency. No forward fills or silent deduplication.
- At least800 valid target observations in the primary block and100 valid prior
  observations for causal context. Invalid OHLC rows may be removed solely by
  the fixed rule and listed; more than1% invalid provided primary rows excludes
  the whole index. Missing dates remain absent.
- Exclude an index if more than20% of primary open/previous-raw-close pairs are
  equal within rtol1e-10,atol1e-12. Report the denominator, per-year fractions and
  open-equals-same-day-close fractions. The threshold is conservative research
  QC, not proof of fabricated opens or a universally validated cutoff.
- Retain every excluded index and reason. Fewer than20 eligible indices means
  the primary is unavailable. For eligible population N in[20,28], the breadth
  requirement is at least ceil(20*N/28) strictly positive market mean gains.

Hash acquired raw data, metadata, source mappings, QC results, exact population,
context histories and common scoring mask in a separate data lock **before any
new-model scoring on that block**. Data locking cannot change dates, quality
thresholds, target, contrasts, margins, seeds or decision rules. No market result
may resolve a mapping ambiguity. A negative or unavailable primary never promotes
the historical secondary to primary.

The exposed historical secondary is2015-01-01 through2022-02-25 on the archived
50,073-row mask across28 indices. It can include the aligned realised-kernel
proxy. It is never called untouched confirmation. External-model benchmarks will
use this same exposure inventory and, if eligible, the same prospective data lock;
their future model-specific conversions require their own pre-scoring lock.

## Forecasts, scores and uncertainty

Forecast using the unchanged causal inference convention: window1024, stride512,
minimum history100, prefix RMS and EWMA state reset within each window. Every
forecast uses only earlier returns. Reject nonfinite or missing predictions on
the locked mask; do not silently drop difficult market/date rows.

Primary gain: broad common-t8 log density minus Matérn common-t8 log density,
using each trained mixture's predictive second moment. This standardizes the
scoring law; it does not remove the trained head. Average paired daily gains
within each market and seed, then weight markets and the five seeds equally.
It is average score performance, not the score of an ensemble density.

Use5,000 synchronized circular calendar-block bootstrap draws, block length60
observed union dates, RNG9182601. The same calendar draws apply across markets,
all fixed seeds and both contrasts. Missing market sessions remain missing and
each market is renormalized on available sampled dates. Report centered/basic
95% intervals and centered one-sided superiority tests with plus-one Monte Carlo
correction. No seed resampling: inference is **conditional on the trained
checkpoints and fixed population**, not a population-of-training-seeds test.
Show every seed estimate and its range. There is no survivor-only inference.

The prospective primary-block superiority family has two slots: broad-minus-
Matérn (five pairs) and broad-minus-GARCH (three optional pairs). Holm controls
this family at0.05; unavailable GARCH contributes p=1. The material-gain gate
requires mean>=0.005 nats, a positive95% interval, Holm significance and the
declared breadth requirement. Report each condition separately. An observed
effect need not have its entire interval above0.005 to clear the point-materiality
criterion; an interval wholly above it is an additional descriptive tier.

**Equivalence is a separate single primary endpoint**, not part of that Holm
family: broad-minus-Matérn within plus/minus0.003 nats, TOST alpha0.05, matching
centered boundary tests and basic90% interval, both required to pass. Report no
joint familywise guarantee across superiority and equivalence. The margin is a
prospective pragmatic choice, not justified simply because another study used it.
Failure of superiority is not equivalence. A small supported positive effect can
also be practically equivalent. For an effect below0.005 say:
**"Statistically supported but below the declared materiality bar."**
Other missed gates and inconclusive results retain their numerical estimates;
do not compress them into a generic success/failure label.

QLIKE and native log density are descriptive/supporting; historical, realised-
kernel and synthetic analyses are explicitly secondary and do not rescue the
primary. No multiple-testing claim spans every endpoint in the dissertation.

## Exposure guard, launch and interpretation

The original runner evaluates best.pt only after all broad training is terminal.
Although these are not the fixed-final objects, their outcomes could inform
allocation. At design freeze inspect artifact existence only, not scores, and
fail if any original base-market evaluation artifact or seed_mean_results/REPORT
exists. Synthetic-only pilot evaluation artifacts are not market contact.
The dated manifest records this evidence before and after creating the lock.
It cannot certify what another person or task has viewed elsewhere.

The queued job waits for the original replication to finish, validates all broad
finals and launch-time source hashes, then trains the five Matérn counterparts.
It may read original run status/completion metadata, but never original market
scores to choose a run. It produces synthetic training artifacts only. Market
inference remains blocked until the prospective data lock exists and verifies.

A positive result supports this broad-prior substitution over the declared
alternative on the declared data. It does not establish that feedback or
financial structure alone caused the gain. A null without supported equivalence
remains inconclusive. No novelty, publication, dissertation mark or market
improvement is guaranteed by passing the engineering or compatibility gates.
