# Cross-market information helper: prospective recipe for a retrospective panel

The user authorized the new cross-market XGBoost helper, rather than a blend
borrowing GNNHAR's forecast. The aim is to provide the own-market intraday and
foreign-market realised-risk information absent from the frozen daily-return
VolPFN. No VolPFN or GNNHAR retraining is authorized here. This is a new follow-up
on an already exposed panel, not independent confirmation or a paper replication.

The feature specification was clarified BEFORE any scientific fitting: use
each foreign market's individual HAR features, not only a cross-market average.
No six-feature scientific fit or GNNHAR-blend experiment was run.

## Data, chronology and fixed population

Use the 28 authoritative Oxford-Man archives and the identical September 26
mask: 22,026 market/date keys, 827 union dates, 2019-01-01–2022-02-25. The original
17 target-defined exclusions remain unchanged. Never intersect on forecast
success, change calendars or silently drop a seed. RK is converted from decimal
variance to percent-squared once by the existing verified loader. Saved PFN/GJR
forecasts already have native percent-squared units and must not be rescaled.

For target day t, own HAR features are risk over the last 1, 5 and 22 completed
own sessions. For EACH other market, build the same session-ending three HAR
features using only source sessions strictly BEFORE the target market's own
previous-session origin date. This conservative convention avoids pretending
same-date foreign closes are simultaneously available. Record actual latest
positive RK inputs, own origin, maximum foreign input date and foreign counts.
Calendar dates identify availability; they are not model predictors. Historical
RK publication latency/timezone vintages remain unverified.

Invalid historical RK inputs may be carried forward using strictly earlier
positive observations; no future/backward fill is allowed. Invalid target labels
are never filled. Native NaNs are permitted for foreign sources without 22
earlier sessions. TRAIN needs at least 10 complete foreign source triplets;
every evaluation key must have all 27. The feature order is own 1/5/22, followed
by individual foreign 1/5/22 triplets in a frozen lexical market order: 84
numerical features per market. No target-market category or date feature is added.

## Independent helper, fixed before pilot and evaluation

Fit a separate small XGBoost helper for each target market. All valid positive
own RK TRAIN labels through 2017 are eligible once own features and the foreign
minimum are available. Each market's positive TRAIN-only RK mean, computed
through 2017, normalizes its inputs and target. Foreign columns are divided by
their respective source market's TRAIN-only mean. TRAIN transforms are batch
estimates; all normalization constants are frozen before 2018/2019 forecasting.
Only the target market's RK is a fitted label; foreign RK enters as a predictor.

Five complete panel repetitions use seeds 8101–8105 (140 fitted helpers). Pin
the isolated XGBoost 1.7.1 runtime. Fixed parameters: reg:gamma positive log-link
objective, 40 trees, depth 3, learning rate 0.1, subsample 0.75, column subsample
1, L2 penalty 1, minimum child weight 1, split gamma 0, base score 1, exact CPU
tree method and one thread per fit; seed-per-iteration is true. No grid search,
validation checkpoint selection, early stopping or evaluation-period refitting.
Whole 2018 is diagnostic validation only; its losses do not change the recipe.

Each market/seed must produce finite positive predictions and TRAIN mean QLIKE
no worse than its eligible TRAIN-label empirical constant-mean forecast within
1e-10. The normalizer uses all positive TRAIN labels, so the eligible-row mean
need not equal one; also record the fixed constant-one diagnostic. Preserve every
failed attempt and diagnose it; do not replace seeds or tune using evaluation.
No forecast floors/negative repairs are allowed: reject invalid predictions.
Undo target normalization exactly once to obtain RK in percent-squared units.
Record TRAIN/validation diagnostics, checkpoints, actual settings, sources and
dependency hashes. Validate coverage and replay saved predictions before scoring.
Record tree splits/importances on own and foreign features without using them
to select a method. Access to foreign predictors does not prove they help.

A timing pilot fits a maximum of two markets and one registered seed. It may
inspect TRAIN and 2018 diagnostics and issue a few forecasts for coverage/timing,
but may not score 2019–2022 outcomes or select a method. Freeze the final sources
and design before the pilot. Unit tests use small synthetic examples, not the
scientific evaluation panel.

## Fixed blend/control comparison

Reference: **full Sep 30 VolPFN + cross-market XGBoost helper**. Both stochastic
components use the same fixed seed ID in each panel repetition; no forecast
ensemble or seed-pair permutation search is performed.

Six comparators, all five fixed seeds retained:

1. Raw helper.
2. Helper-only log-linear calibration: log h = a + b log h_helper.
3. GJR-GARCH-t + helper (broadcast the saved deterministic GJR recipe to each
   helper seed explicitly).
4. Saved full VolPFN + TTM, unchanged from the prior extension.
5. GNNHAR, unchanged from the prior extension.
6. Transfer-XGBoost, unchanged from the prior extension.

The two-component blend preserves the prior algorithm:
log h = a + w log h_A + (1-w) log h_helper, with unconstrained w.
Both calibration and blends fit two free parameters, separately by proxy and
seed, using pooled market/day QLIKE on strictly earlier completed union dates.
Initial 60 union dates use a=0,w=0.5 for blends and a=0,b=1 for helper-only
calibration. Refit every 20 union dates. Nelder-Mead starts at those same fixed
initial parameters; xatol=1e-6, fatol=1e-10, maxiter=2000. Reject optimizer
failure/nonfinite outputs and preserve failures. Fits must be no worse than
their starting objective within 1e-8. No late/outcome-based starts or tuning.

These controls test whether PFN adds information beyond a cross-market helper
and beyond calibration, and whether it contributes more than another daily
forecaster. They do not prove PFN architecture superiority. GNNHAR additionally
uses a return-derived graph, pooled multi-market fitting and 2018 checkpoint
selection; supplying these RK inputs does not match those design choices. Helper and blend
training/fitting opportunities differ from other pipelines; access to the same
RK series does not imply identical representations or equally effective use.

Squared-return QLIKE uses saved PFN/GJR second moments; RK QLIKE uses their
central variance. The helper issues a native RK point used against each proxy,
without inventing a return density. Proxy-specific blends/calibration can differ
and do not form one jointly coherent density. RK equality with open-to-close
return risk remains a documented proxy assumption.

## Scoring and conditional uncertainty

Score y/h + log h. Average dates within each market/seed, then markets and seed
losses equally. Never average forecasts before scoring. Rank all seven complete
procedures descriptively, never omit a superior competitor. No success-only mask.

The new reference has six comparators and two proxies: a NEW fixed 12-slot
family, separate from the earlier 38-slot family. Use 20,000 synchronized circular
calendar resamples, seed 2026093005, primary block 60, sensitivities 20/120.
Two-sided centered plus-one p-values, joint Holm, ordinary 95% and Bonferroni
family 95% basic intervals. Require both proxies, adjusted p<.05, family interval
direction and all five corresponding fixed seed-difference directions for joint
support. Matching seed IDs couples the shared helper where applicable; it does
not establish paired training randomness for unrelated architectures.

Inference is conditional on frozen forecasts/models. It excludes helper training,
online refitting, pretraining and research-selection uncertainty. Choosing this
helper after the earlier results makes this retrospective, even with causal
forecast fitting. Sensitivities cannot replace the primary result. Preserve the
old table, family, archives and conclusions; do not splice new p-values into it.

## Preservation and execution

All outputs use fresh exclusive directories under research/outputs. Freeze code,
protocol, design, dependency identities and input hashes before scientific fits.
Run meaningful chronology/unit/seed/positivity tests and a timing pilot; then
complete all 28 markets/five seeds, independent quality verification, comparison,
uncertainty and qualified report. Local CPU is adequate; do not provision a new
GPU just for this small helper. No prose-style manuscript rewrite is started.
