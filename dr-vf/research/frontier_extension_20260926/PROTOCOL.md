# Historical frontier extension, 26 September 2026

This protocol declares a finite rerun of the existing models and three new
comparators on a longer historical panel. It is an extension on **previously
exposed historical data**, not an untouched confirmation experiment. Freeze the
final source, origin/target bundle, model roster, checkpoint hashes, adapters,
runtime and execution settings before producing or inspecting evaluation losses.
The machine-readable frozen bundle and source manifest identify the actual run;
this document does not by itself certify that freezing or execution has occurred.

## Population and timing

- Reuse the 28 canonical market files from `combined_leaderboard_v2`, checking
  their hashes against that archive. Request target dates from **2019-01-01
  through 2022-02-25**, inclusive. No observations after that endpoint are needed.
- A forecast at market index `t` predicts that observed market session using only
  indices strictly smaller than `t`. This is one next observed session, not a
  fixed 24-hour horizon. Preserve each market's trading-session calendar.
- Returns are the archived percentage open-to-close log returns. Targets are
  `r2 = r**2` and the archived Parzen realized kernel in percent-squared units
  (`10000 * RK` in unscaled return units). Do not change the raw target arrays.
- `S_all` is every requested observed session with its finite squared return.
  The primary matched population `S` is defined solely by finite, strictly
  positive target RK. The same `S` is used for both primary proxies and every
  model/seed. Input availability, forecast success, predicted losses and model
  ranking cannot alter it.
- The read-only target audit gives **22,043 `S_all` rows; 22,026 `S` rows; 28
  markets; 827 union dates**. The 17 exclusions are 4 BSESN, 12 NSEI and 1
  STOXX50E target rows. Preparation must reproduce these counts. Different
  market holidays remain missing cells on the union calendar, not fabricated
  observations.
- Retain the exact archived **72 calibration origins**: 24 each in FTSE, N225
  and SPX, within 2012-01-04 to 2013-12-02. Copy their exact keys and targets from
  the hashed old bundle; matching only the date bounds and counts is insufficient.
  Calibration and evaluation do not overlap.

## Fixed roster and information

The reference is the existing VolPFN checkpoints for training seeds
`8101, 8102, 8103, 8104, 8105`. Every seed remains required. The 16 primary
comparators are the 13 old controls plus `ttm_rk`, `ms_har`, and `log_rhar`:

| Group | Primary models | Information |
| --- | --- | --- |
| V | VolPFN, five separate seeds | Existing corrected checkpoints; independent 1,024-return contexts |
| E | hist250, ewma, garch_t, gjr_garch_t, egarch_t, aparch_t, compact9, univol | Existing return-information adapters |
| F | chronos2, moirai2, tabpfn_ts | Existing squared-return foundation adapters |
| R | har, loghar, ttm_rk, ms_har, log_rhar | Past realized-kernel information |

`loghar_meanlog` and the separately named TTM square-root-input recipe are
diagnostics. They have no primary ranks, no hypothesis slots, and cannot replace
the primary model after results are seen. R-model rankings are displayed
separately from V/E/F rankings because their information differs. All 16
VolPFN comparisons nevertheless share one multiplicity family.

Rerun all old models on the new origins without retuning their adapters, training
weights or archived fitted objects. New scores must not be spliced into the old
period's scores. Prior cached forecasts are reusable only when origin, history,
checkpoint, adapter, runtime and numerical-bridge identities all match; a reused
calibration forecast must retain the same identity checks. Parameter-state
methods receive the same full calibration/evaluation origin union per market so
that the first required origin anchors their refit clock. They process
intervening sessions, including sessions not scored.

## New realized-information adapters

**TTM R2.1:** pin IBM `ibm-granite/granite-timeseries-ttm-r2`, branch
`512-96-ft-r2.1`, revision
`3442406f106122f414a11f72b4ebfe4878bec89d`, and the four asset hashes in `ttm.py`.
Use 512 preceding market-session RK inputs, univariate MSE point output, daily
frequency token 8 and the first of the checkpoint's 96 output horizons. The
primary recipe feeds RK directly. Per-context population standardization and
its inverse use that context alone; CPU float32 is the verified initial runtime.
The square-root recipe is diagnostic: squaring its point forecast is
`E[sqrt(RK)]**2`, not `E[RK]`, even when its original paper motivates that recipe.
Both are zero-return-mean variance-proxy adapters, not native return densities.

Invalid RK **inputs** are forward filled from the last finite positive value in
the past, including a value preceding the 512-row window if available. No
backfill is allowed; an unfillable context fails. Count imputations. The output
floor is `1e-8 * max(mean(context_RK), 1e-12)` in variance units; record the native
point and every floor. Never square a negative square-root-output point into an
apparently valid positive variance. A nonfinite model output fails the run's
coverage requirement rather than causing a dropped origin.

**MS-HAR:** two-state Gaussian HAR on square-root RK. Only the intercept, daily
slope and emission variance switch; weekly and monthly slopes are common, as in
the Fed paper's Appendix B.3.1, equation 31. Predict the full mixture second
moment `sum(p_s * (mu_s**2 + sigma_s**2))` to target RK. This is a variance-target
adaptation of the paper's volatility point, not a literal replication of equation
34. The number of regimes is fixed; the paper's recursive linearity diagnostic
is not performed. Report implied negative Gaussian volatility mass as an audit.

**log-Rough-HAR:** regress log RK on the daily value and weekly/monthly means of
log RK, with a Gaussian MA(1) error. This follows the paper's HAR+MA(1) equation
2.9. The implementation uses stationary initialization of the MA error conditional
on the observed lagged design. It does not reproduce or claim the paper's joint
stationary ARMA(22,1) initial likelihood or AR-root constraints. Forecast RK with
`exp(log_mean + 0.5 * one_step_log_variance)`, including filtering uncertainty.
The nested `theta=0` mean-of-logs HAR diagnostic uses Gaussian MLE variance
`SSE/n`; it does not replace the old `loghar` (logarithms of arithmetic RK lag
means, residual correction `SSE/(n-4)`).

Both new fitted models use **1,000 valid training targets** and refit every **20
actual market sessions**, anchored at the first required origin. This window
differs deliberately from the old models' 1,000-calendar-session window. Missing
emissions between the selected valid targets remain in the likelihood/filter
clock. Predictors are constructed from causally forward-filled inputs; missing
targets have likelihood one and receive prediction-only state updates.

Econometric optimization is finite: 300 iterations maximum, fixed three starts,
no stochastic search or evaluation-based choice. Accept only converged finite
fits with valid positive variance and interior transition/MA parameters; choose
among accepted starts by training likelihood. The MA boundary policy rejects
and carries, rather than reproducing the paper's boundary cap. Failed refits
carry accepted parameters while continuing filtering. Before any accepted fit,
use the mean of positive RK in the last 250 sessions. Invalid daily forecasts
carry the last valid value, or that causal RK mean if none exists. The log models
replace predictions outside the fitted raw-RK training range with that training
mean. Preserve every fit failure, carried-state episode, range replacement and
other substitution. These are part of the declared forecasting algorithms.

## Calibration and estimand

Preserve the old primary convention. Separately for every model/seed, compute
`c = mean_over_3_markets(mean_over_24_origins(r2 / raw_second_moment))` using only
the exact 72 calibration rows. Use that **same** scalar for both proxies:
`c * second_moment` predicts squared returns and `c * variance` predicts RK.
The R and F point adapters declare zero return mean and set variance equal to
second moment. Native return models retain distinct moments where applicable.

Report calibrated primary and unscaled secondary scores side by side, including
each scalar and the direction/size of raw-versus-calibrated gains. Here “raw”
means **after the frozen numerical adapter/floor/fallback but before scalar
calibration**, not an unchecked raw neural output. Calibration can materially
change rankings; a primary gain does not imply an uncalibrated gain. Provide
squared-return `S_all` results as an explicitly secondary population.

Loss is the zero-safe QLIKE form `log(h) + y/h`. Average time within each market,
then give each market equal weight. For VolPFN, average the losses of the five
separately calibrated seeds; do **not** average forecasts and then score an
ensemble. Show every seed score, range and comparator gain. The gain sign is
`comparator loss - VolPFN loss`, so positive favors VolPFN.

## Uncertainty and reporting

Use **20,000 synchronized circular calendar-block bootstrap draws**, seed
`2026092601`, with **60 union-calendar dates per block** as the main analysis.
Resample the same date-count vector for every market, model, seed and proxy.
Within each draw renormalize by the available sampled dates of each market,
then average markets equally. Never resample markets independently. A draw that
omits a whole market fails rather than silently losing that market.

All 16 primary comparators against VolPFN across both proxies form **32 joint
comparisons**. Compute centered, one-sided positive-gain bootstrap p-values
with plus-one correction and Holm adjustment over all 32 slots at alpha 0.05.
Also report basic two-sided 95% pointwise intervals and Bonferroni simultaneous
family intervals (each marginal coverage `1 - 0.05/32`). The family interval is
an additional conservative requirement, not a substitute for Holm.

A comparison supports positive VolPFN improvement only if, on **both** proxies
at block 60, the gain is positive, its family interval lower bound is positive,
Holm-adjusted one-sided p is below 0.05, and the gain is positive for **every**
retained VolPFN seed against the comparator seed-mean loss. Failure to meet the
rule is not proof of equality. Negative estimated gains and family intervals
wholly below zero must be disclosed; the positive-direction one-sided p-value
must not be repurposed as a reverse-direction significance test.

Repeat the same joint family at block lengths **20 and 120** as named sensitivity
analyses. They cannot replace the main block or create a success claim absent
at block 60. These intervals condition on the fixed forecasts, markets, adapters,
seeds and calibration; they do not include training-population, seed-distribution,
calibration-estimation, model-selection or pretraining-overlap uncertainty.
Calendar-block validity is an approximation, especially through a changing
market regime; the historical panel and 827 dates do not establish universal
superiority.

Require every declared model/seed on every frozen calibration and `S_all` origin.
Duplicates, extra keys, target changes, missing/failed seeds, or invalid moments
stop complete analysis. Publish coverage, per-market row counts, substitutions,
input imputations, floors and seed failures even for an incomplete attempt.
Ranks are descriptive. Recent foundation checkpoints may have seen overlapping
historical data, and were not available at these historical dates. Causal
test-time windows do not remove that limitation.

The authorized operation is bounded by $10 and approximately eight total hours.
The native pod credential returned HTTP 403 on authenticated self-read, so a
pod-local billing-stop guard is **not verified or armed**. The authorized fallback
is an external, Mac-dependent monitor: a 15-minute heartbeat plus the explicit
original operation deadline and any earlier per-pod budget deadline. The Mac
must remain awake with its lid open, and have a working network/account connection
for those checks and stop requests.
A heartbeat can detect or act after a threshold has passed; it is not a
provider-enforced dollar cap. Frozen approval/runtime receipts must state this
fallback accurately and must not claim that `cloud_guard.py` is armed merely
because that module exists.

Local and remote worker deadlines bound computation, not cloud billing: ending
or killing a forecasting process does not stop its rented pod. Bind resume and
supervisor commands to the original absolute operation deadline; do not restart
the eight-hour clock or override an earlier per-pod deadline. Stop the pod
externally on completion, an unrecoverable failure or the budget/deadline boundary.
Preserve volume artifacts; do not
terminate/delete the pod. Use finite resume-safe work units. Do not extend
budget, silently omit slow models/seeds, or select a shorter favorable period
to finish. If the declared run cannot complete, preserve artifacts and report
it as incomplete. A new untouched period or changed primary recipe requires a
separate protocol, not relabeling this extension as confirmatory evidence.
