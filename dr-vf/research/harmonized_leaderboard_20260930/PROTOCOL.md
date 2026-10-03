# Harmonized saved-forecast leaderboard, 30 September 2026

This is a new retrospective analysis, authorized to combine the older slide
comparison with the completed overnight daily models. No model is trained,
refitted, selected or calibrated here. Originals remain authoritative for their
original procedures. This protocol is written before inspecting harmonized
scores; the underlying studies and historical data have already been seen.

## Fixed scope

Use the **exact 22,026 primary market/date keys** in the September 26 bundle:
28 markets, 827 union dates, 2019-01-01 through 2022-02-25. Retain market holidays
and the original 17 target-based RK exclusions. Do not intersect on successful
forecasts or merely choose date bounds. Every declared seed must cover every key.

The main daily table has 15 procedures: September 30 full VolPFN (reference),
September 26 VolPFN (separate version), ARCH-LSTM, GARCH-LSTM, GINN, hist250,
EWMA, GARCH-t, GJR-GARCH-t, EGARCH-t, APARCH-t, UniVol, Chronos-2, Moirai-2 and
TabPFN-TS. Keep full as reference regardless of noaux results. Preserve all five
seeds of each PFN version and each new neural comparator. Older deterministic
controls retain their one issued seed/recipe, not five artificial repetitions.

Five lagged-RK procedures (HAR, log-HAR, TTM-RK, MS-HAR and log-Rough-HAR) have a
separate descriptive ranking because their information differs. No blend is
constructed or substituted. Keep compact9, loghar_meanlog, ttm_sqrt_rk and the
new noaux/direct/historical arms as unranked supplementary results. This retains
compact evidence while respecting its removal from the main presentation.
GP-Vol, RECH and DCC/GN-GARCH lack complete compatible forecasts here; no defeat
or rank is invented. Future nonlinear intraday models are not part of this run.

## Units and forecast functionals

All issued forecast means are percentage points, and variances, second moments,
squared returns and kernels are percentage points squared. Verify the old
already-scaled source arrays against the new decimal-return arrays ×100 and
decimal RK ×10,000 on every key. Verify frozen hashes, targets, positive finite
moments, M2 = variance + mean², causal saved context endpoints and complete seeds.
New external outputs already inverse-transform TRAIN-only standardization:
mu_raw = center + RMS*mu_z; variance_raw = RMS²*variance_z;
M2_raw = variance_raw + mu_raw². Never standardize those outputs again.

Calculate L(y,h) = y/h + log(h) from the **unscaled issued moments after their
original adapters/floors/carries**, before posthoc scalar calibration:

- squared-return QLIKE: y = r², h = issued second moment;
- kernel QLIKE: y = RK, h = issued central variance.

The same functional and units apply to every row of each metric. The old
2012–2013 calibration cannot be reused as independent calibration for new models
selected on 2011–2014 validation. Do not mix calibrated old scores with raw new
scores. Preserve the older calibrated leaderboard and all overnight scores.
Overnight saved kernel QLIKE used M2; **recompute** the new central-variance
kernel score, do not copy that column. This is an explicitly changed secondary
estimand, not a correction silently imposed on the historical experiment.

The direct arm provides only M2, without a native mean or central variance.
Its common RK/variance score is unavailable; retain its M2-based kernel score
as a distinctly labelled diagnostic. Do not invent a zero mean to give it a rank.
Kernel alignment with the open-to-close predictive variance is still an empirical
proxy limitation, even when its numerical units are correct.

## Averaging and uncertainty

Score each seed separately, average time within market, then markets and seeds
equally. Never average forecasts before scoring or pool all dates across markets.
Save row scores, per-market/per-seed scores, counts and original fallback flags.

For the 14 main daily comparators versus September 30 full, both proxies form
one new **28-slot** family. Recalculate 20,000 synchronized circular calendar
block draws, seed 2026093003, main block 60 and sensitivities 20 and 120. Preserve
missing dates; renormalize within each market on each draw. Use centered
two-sided plus-one p-values, joint Holm and basic 95% intervals plus Bonferroni
family intervals (marginal coverage 1-0.05/28). A positive support flag additionally
requires positive gain for every fixed reference seed and positive family lower
bound on both proxies at block 60. Report negative differences explicitly.
No archived p-values are copied. Secondary/variant/intraday scores have no
significance flags. These retrospective conditional tests do not account for
research selection, calibration uncertainty, training-seed populations or
pretraining overlap, and do not establish universal superiority.

## Preservation and interpretation

Write only to a new output directory, refusing overwrite. Record source and input
hashes before analysis and verify them again after. Copy old calibrated summaries
only with their original labels; preserve all original data, checkpoints, masks,
sources, results and failed attempts. No GPU resources are needed.

This compares issued forecasting pipelines. Contexts, fitting opportunities,
training information/budgets and pretraining exposure differ. Old accepted-fit
checks support forecast reuse, but APARCH/EGARCH carries and extreme forecasts,
HAR/Moirai floor events and independent-adaptation limitations remain material.
Do not claim architecture isolation, a fresh holdout or author-code replication.
