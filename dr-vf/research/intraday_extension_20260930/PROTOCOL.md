# Intraday-information extension, 30 September 2026

This is a new, retrospective extension authorized after the completed daily
programme. The objective is a **simple frozen VolPFN + saved TTM-RK blend**,
compared with established and feasible recent realized-risk models. There is no
new VolPFN training or intraday-input transformer. Every historical dataset,
checkpoint, source, result, failure and calibrated table remains untouched.

The user subsequently confirmed explicitly: **do not train a new intraday
VolPFN**. The finite extension continues only its existing blends, registered
benchmark fits, scoring, uncertainty checks and report. No new intraday-input
model was started, and hypothetical future tests do not authorize one.

## Locked population and units

Use exactly the September 26 primary mask: 22,026 market/date keys, 28 markets,
827 union dates, 2019-01-01 through 2022-02-25. The same 17 target-defined kernel
exclusions apply to both endpoints. Never intersect on successful forecasts or
synchronized market holidays. Native market calendars are retained.

The mask and saved forecasts come from
`research/outputs/harmonized_leaderboard_20260930_v1`. Its verified units are
percent returns and percent-squared risk quantities. Decimal Oxford-Man returns
were multiplied by 100 once and kernels by 10,000 once. Saved issued forecasts
are already back in those units; do not transform them again.

For saved density/return-moment models, squared-return QLIKE uses issued M2 and
kernel QLIKE uses issued central variance. Native realized-risk models produce
an RK forecast, not a return density. Their same issued risk-level prediction is
scored against each proxy, explicitly without inventing a mean or claiming a
full return distribution. RK equivalence to open-to-close return risk remains a
proxy assumption, not a proved identity.

Loss is y/h + log(h), lower is better. First score each seed, average dates within
market, then markets and seeds equally. Never ensemble forecasts before scoring.
The five fixed VolPFN seeds are 8101–8105. Deterministic older controls remain one
recipe rather than five artificial repetitions.

## Blend fixed before this extension's scoring

Reference: **Sep 30 full VolPFN + TTM-RK**. Sep 26 VolPFN stays separately named.
Reuse the historical log-linear blend algorithm on **raw issued components**:

    log h = a + w log h_A + (1-w) log h_B

Fit a and unconstrained w independently per proxy and per retained PFN seed.
Initial 60 union dates use a=0,w=0.5. Refit every 20 union dates on all completed
dates strictly before the block starts. Objective is pooled market-day mean
QLIKE; final scores weight markets equally. Nelder-Mead starts at [0,0.5], with
xatol=1e-6, fatol=1e-10, maxiter=2000; failed fits reject the run. Weights are
log-linear coefficients, not bounded percentages. The intercept performs
online calibration even though components are raw.

Both proxies receive different causal fits. These are two risk forecasts, not
one coherent predictive density. Do not splice historical calibrated scores
into this raw-component ranking. Preserve their exact table separately.

Controls have the identical fitting opportunity: GJR+TTM, GJR+Log-HAR and
MS-HAR+Log-HAR. Retain new PFN+Log-HAR, new PFN+MS-HAR and old PFN+TTM for
comparison, without selecting a new reference from observed results.

## Complete literature/comparator register

All five requested sources remain registered, including ones whose complete
data or faithful replication cannot currently be supplied:

1. Zhang et al. (2025), GNNHAR. Author architecture/graph/loss must be checked.
   A causal global-calendar adapter is necessary for these 28 index markets.
2. Teller et al. (2026), multi-source transfer XGBoost. New-issue/spin-off
   population and additional predictors differ. An index-only adaptation must
   be labelled as such; it is not automatically the published experiment.
3. Kong et al. (2025), M2VN. Historical point-in-time news, market predictors
   and a different native target are required. Never fill absent news with
   fabricated data or call a market-only network the complete M2VN model.
4. Puke & Schweikert (2026), QLIKE-trained HAR. Verify exact objective,
   constraints, horizon, fitting window and positivity handling from author code.
5. Brini (2026), foundation-model realized-volatility study. This is a study,
   not an additional model; TTM-RK is the retained runnable comparator. Its
   existing adapter is not claimed to replicate that study's 50-asset experiment.

DeepRGARCH remains an additional previously discussed feasibility slot. It uses
Bayesian SMC; an ordinary neural/MLE fit must not silently replace that method.
Author-code bugs or full-sample transformations require documented causal fixes.
Do not claim a method was defeated unless every registered seed covers the whole
common mask with accepted finite positive forecasts and provenance.

The fixed roster in `run_blends.py` has 20 procedures (reference plus 19
comparators); ten reference-comparator proxy slots belong to the five new
frontier/feasibility methods. Missing slots stay unverified with p=1 if inference
is subsequently run; they cannot disappear to shrink the testing family.

## Execution and evidence

Saved blends run locally. Frontier models must pass unit/leakage/coverage checks
and a timed pilot before any complete fitting or paid deployment. Pre-2019
fitting and validation dates, seed sets, source pins, adaptation details and
budgets must be saved before full forecasts. Neither test QLIKE nor favourable
seed screening may choose a model. Record causal refit opportunities rather than
claiming every model had identical information or compute.

The same historical panel has already been exposed. Results are retrospective
pipeline comparisons, not an untouched holdout or universal frontier claim.
Original frozen TTM forecasts also inherit pretraining-overlap limitations.
If an uncertainty analysis is run, use 20,000 synchronized circular calendar
draws, seed 2026093004, blocks 60/20/120 and the full 38-slot joint family; retain
unavailable slots. Ordinary and family intervals must be distinguished. Such
conditional intervals do not remove selection or online-fitting uncertainty.

Outputs use fresh, exclusive directories. Record before/after source hashes,
weights, row forecasts/losses, coverage, seed/market summaries and failures. New
stages produce new versions rather than modifying completed earlier results.
