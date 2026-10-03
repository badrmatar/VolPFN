# Fixed overnight extension: April 2015–June 2016

Specified after viewing January–March 2015 forecasts and their exploratory
uncertainty analysis, and before generating this extension's forecasts. The
endpoint is selected to fit the user's approximately ten-hour compute window,
not selected from extension results. This is a historical follow-up: the market
data and earlier econometric/VolPFN evaluations have already been inspected.
It is not an untouched market test or proof of prior causality. Foundation-model
pretraining overlap is unknown. A frozen specification does not remove this
research-selection history.

## Fixed sample and forecasts

Primary follow-up period: 1 April 2015 through 30 June 2016, all 8,774 archived
market/date rows in the original mask, all 28 indices, 327 union-calendar dates.
Three original bridge origins are engineering checks and excluded from scores.
Reuse all 1,730 January–March predictions unchanged for an 18-month descriptive
table (10,504 rows). Inferential conclusions use the extension only. No interim
comparative scores or p-values, outcome-dependent stopping, changed endpoint,
series removal, or selective promotion of a subperiod.

Reuse the four pinned checkpoints, adapters, output semantics, eight-member
TabPFN ensemble, 1,024-observation causal contexts and original development
scale multipliers. Retain negative-output floor events and extreme returns.
VolPFN is fixed-final seed 8101. Scale calibration is the archived 72-observation,
three-market 2012–2013 estimate, not a full-panel fit. No new fitting or tuning.
Mean/median output semantics remain different. This is neither a native-density
comparison nor a seed-mean result. Moirai's floor-dominated QLIKE must be disclosed.

## Statistical rules

Main question: does VolPFN beat TabPFN-TS on fixed-scale squared-return QLIKE?
Retain all nine comparisons (three competitors by native return pinball,
native squared-return QLIKE and fixed-scale squared-return QLIKE) in the same
Holm family. Positive gain is competitor loss minus VolPFN loss. Average dates
within each market, then weight all markets equally.

Use 20,000 circular calendar-block bootstrap draws, seed 2026091902. Main block
length is 10 observed union-calendar dates; report 5 and 20 as sensitivities,
without selecting a preferred result afterwards. Resample dates jointly across
markets and contrasts, preserving absent sessions and renormalizing each market.
Fail rather than redraw if a sampled market has no observations. Report basic
95% pointwise intervals, two-sided centered p-values with plus-one correction,
Holm-nine within each setting, and conservative Holm-27 across settings.
Also report basic Bonferroni family intervals at individual coverage 1-.05/9.
As in the preceding analysis, these intervals and centered tests are not exact
inversions under skewness. Approximate coverage depends on time-series regularity.

A supported positive (negative) comparison requires Holm-nine p <= .05 and a
positive (negative) estimate. For the stronger headline 'supported superiority',
also require that the main-setting Bonferroni interval lies wholly above zero;
'supported inferiority' requires it wholly below zero. Otherwise the stronger
headline is inconclusive. Report all criteria separately. A non-significant
result is not equivalence. No materiality threshold is newly introduced; report
the absolute gain and uncertainty rather than claiming practical importance
from significance. Sensitivity disagreement must qualify the headline.

Uncertainty is conditional on fixed forecasts, calibration, selected markets and
checkpoint; it excludes calibration estimation, training and research-selection
uncertainty. Market counts are descriptive, not independent replications. Even
a successful test supports only the stated task, period and compared procedures.

## Integrity and operations

Hash data, mask, source, models, calibration and prior predictions before launch.
All cells must be complete and aligned before comparison. Preserve every failed
attempt and all prior studies. No result table from a timeout-truncated sample.
Test boundary/overlap handling and decision rules; independently verify score
means, bootstrap draws and Holm adjustment. Store draws, counts and hashes.

Measured throughput is 3.55179 seconds per origin on the previous RTX 4090:
8,777 predictions including bridges imply 8.66 GPU forecasting hours. Transfer,
startup and export take additional time. Existing pod rate is $0.74/hour;
ten hours costs $7.40 compute plus storage. Verify sufficient credit before
launch. Stop on completion, failure or the fixed ten-hour pod-runtime deadline;
never terminate or delete a saved pod. A timeout is not a billing stop. The
current MCP stop monitor requires the Mac awake and connected; do not claim an
independent cloud cap. If the user needs to close the laptop, resolve shutdown
separately before an unattended launch. No automatic further extension.
