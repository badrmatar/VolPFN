# Calibeating the (VaR, ES) pair — falsification sandbox protocol

Written 7 September 2026. A pipeline smoke run at small scale (`outputs/sandbox_smoke`, 2 seeds,
T <= 16k, 2 levels) was executed before this file was frozen, to check the code end to end; it is
recorded in STATUS.md and its numbers are not the study. Two changes were made after it and before
this freeze: sliding-window variants of the ratio calibeater were added because the
stochastic-approximation tracker was too noisy at its first step size, and the report guards a
division by a vanishing miscalibration. The thresholds below apply to the full run
(`outputs/sandbox_v1`) and are not to be revised after it.

## Claims under test

The object is an online, deterministic wrapper that takes any black-box (VaR, ES) forecaster's
output (v_t, e_t) and past outcomes, and emits a recalibrated pair before y_t is seen. Scoring is
the strictly consistent FZ0 loss of Patton, Ziegel and Chen (2019).

* **C1 (hindsight no-harm).** For a forecaster whose pair is location-scale with a fixed tail law
  (e_t = kappa v_t, kappa constant — every GARCH/RiskMetrics/historical-simulation-style model in
  practice), replacing the pair by the empirical (VaR, ES) of the standardised outcomes y_t/|v_t|,
  mapped back through |v_t|, never increases the average FZ0. The gain is the forecaster's
  miscalibration MCB with respect to that class. (Exact, by strict consistency of FZ0 and its
  degree-zero homogeneity; the sandbox checks the implementation, not the theorem.)
* **C2 (online no-harm, stationary).** The online follow-the-leader version (`ratio_ftl`) has
  average FZ0 at most the hindsight value plus a regret term that is sublinear in T, so it is no
  worse than the original forecaster up to o(1), and captures the forecaster's MCB asymptotically.
* **C3 (strictness).** For a mis-specified black box the online gain converges to MCB > 0, so the
  wrapper strictly beats every mis-specified location-scale forecaster.
* **C4 (model-free binning pays resolution).** Foster-Hart style binning on the level of v_t, which
  discards the forecaster's within-bin dynamics, can be worse than the original for a forecaster
  with good dynamics and a wrong tail; the scale-structured wrapper does not have this cost.
* **C5 (drift).** Under a break in the tail law, a windowed variant closes more of the gap to the
  oracle than the full-history version.

## Construction

DGPs with an exactly known conditional (VaR, ES): GARCH(1,1) with unit-variance t(5) innovations;
a two-regime Markov volatility with t(5) innovations; the GARCH with a tail-shape break (nu 8 -> 3
at T/2). Black boxes, all causal: the oracle; the oracle's sigma with a Gaussian tail; the oracle
scaled by 0.8; RiskMetrics EWMA with a Gaussian tail; EWMA with a t(5) tail; a rolling 250-day
historical simulation. Calibeaters: `ratio_ftl`, `ratio_win1000`, `ratio_win2000`, `ratio_sa`
(eta 0.002), `bins1/5/10`. Warm-up 250 observations (pass-through), scoring from t = 250.
Benchmarks: the oracle (floor) and the hindsight recalibration for each grouping (defines MCB).
Grid: alpha in {0.05, 0.025, 0.01}; T in {1k, 2k, 4k, 8k, 16k, 32k}; 5 seeds; 1,620 cells.

## Decision rules (evaluated by `calibeat.sandbox`, per DGP and level)

* **D1, no harm on a calibrated forecaster.** Oracle + `ratio_ftl` minus oracle at T = 32k is at
  most one tenth of the EWMA-Gaussian forecaster's MCB at the same T, and the excess decays in T
  with a fitted exponent <= -0.4 (or is non-positive). Prediction: pass on the two stationary DGPs,
  fail on the break DGP (FTL is not meant to track a break).
* **D2, the gain captures the miscalibration.** For the EWMA-Gaussian forecaster, online gain /
  MCB >= 0.8 at T = 32k. Prediction: pass at alpha 0.05 and 0.025; open at 0.01 (about 320 tail
  events at T = 32k, which is the practical limit of ES recalibration and worth knowing).
* **D3, sublinear regret.** T x (online - hindsight) for the EWMA-Gaussian forecaster grows like
  T^b with b < 0.8. Prediction: pass, b between 0.2 and 0.5.
* **D3b, drift.** On the break DGP the best of the windowed/SA variants gains more than `ratio_ftl`
  against the original. Prediction: pass for the 1000-window at alpha 0.05; open at 0.025.
* **C4 is read descriptively** from the oracle and oracle-Gaussian rows: the bins should show large
  negative gains on the oracle where the ratio wrapper shows near zero.

## What failure looks like

D1 failing on a stationary DGP means the wrapper harms a calibrated forecaster by more than a
tenth of what it earns on a mis-specified one at practical sample sizes; the no-harm theorem would
then be true only asymptotically and the paper must say so. D2 failing at alpha 0.025 means ES
recalibration is data-starved at the Basel level with a decade of daily data, and the paper
becomes a VaR-calibeating result with ES reported. D3 failing means no online guarantee is
available for follow-the-leader and the theorem needs a different procedure (exponential weights
over a grid, at a cost in rates). Nothing here touches market data; that comes only after the
theorem is stated, on the 28-index development panel with a reserved block, as in dr-vf.
