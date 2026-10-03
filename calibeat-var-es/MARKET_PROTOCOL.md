# Calibeating the (VaR, ES) pair on the Oxford-Man panel — market protocol

Written and frozen 7 September 2026, after the sandbox (PROTOCOL.md, `outputs/sandbox_v1`, 25/27
pre-registered decisions passed) and before any development-panel run beyond the pipeline smokes
disclosed below. Thresholds are not to be revised after `outputs/market_dev_v1`.

## What is tested

The online wrapper of `calibeat.calibeater` (standardise outcomes by |v_t|, follow-the-leader on the
empirical (VaR, ES) of the standardised history, map back) around real black-box (VaR, ES)
forecasters, on the 28 equity-index return series of dr-vf's development panel (Oxford-Man realized
library via Barde's replication release; calendar-corrected copy `calendar_development_v2`; daily
log returns in percent; 2000-01 to 2014-12). Scoring: FZ0 (Patton-Ziegel-Chen 2019), averaged over
the development window `split == 1` (2011-01 to 2014-12, about 1,006 days per market), panel mean
over markets. The reserved block (2015+) is not read by the development run.

Black boxes (all causal; parameters refitted on an expanding window every 250 days from day 750):
`ewma_gauss` (RiskMetrics, lambda .94, Gaussian tail), `ewma_t5` (same variance, unit-variance t5),
`hist250` (rolling 250-day historical simulation), `garch_t` (GARCH(1,1)-t by MLE),
`garch_gauss` (the garch_t variance path with a Gaussian tail — right dynamics, wrong tail law),
`gas_fz0` (PZC's one-factor GAS fitted by FZ0 minimisation, one model per level), `volpfn_t8` (the
dr-vf network's extracted variance path with the unit-variance t8 law). Levels alpha in
{0.05, 0.025, 0.01}. Wrappers run online from day 0 with a 250-day pass-through, so they hold about
2,750 days of standardised history when scoring starts.

Wrappers: the pure ratio wrappers `ratio_ftl`, `ratio_win1000`, `ratio_win2000`, `ratio_sa`; the
unguarded switch `switch_*` (follow-the-leader over the two experts {original, wrapped} on
cumulative FZ0 since day 250, original for the first 250 comparisons); the guarded switch `guard_*`
(the same comparison over the last 1,000 days only, and the wrapped pair is used only if it beat the
original by more than a margin per day — half the regret budget below); Foster-Hart style `bins5`
for contrast. The binding variant for the development rules is **`guard_ftl`**.

Benchmarks: `hindsight_full` = the constant-ratio recalibration fitted on the whole 2000-2014
standardised sample, evaluated on the window (defines MCB_full); `hindsight_window` = the same
fitted on the window itself (the in-sample optimum of the class on the window).

Statistics: per (forecaster, level, wrapper), the panel mean difference wrapper minus original
with the calendar-block bootstrap of dr-vf (60-day blocks, 2,000 resamples, seed 927601, dates
resampled jointly across markets), a two-sided bootstrap p, and the number of markets improved.

## Regret budget

The finite-sample cost of the wrapper around a perfectly calibrated forecaster was measured in the
sandbox in this panel's exact geometry (T = 3,763, scored on the last 1,006 days, GARCH-t and
regime DGPs, 10 seeds): mean oracle excess +0.0006 / +0.0018 / +0.0002 FZ0 units at alpha =
0.05 / 0.025 / 0.01, single-window standard deviations 0.004 / 0.0065 / 0.020. The budget is the
mean plus two standard errors of a ten-effective-market panel: **B = 0.0035, 0.006, 0.013**.

## Decision rules (evaluated by `calibeat.market summarise --variant guard_ftl`)

* **M1, gain on a mis-specified black box (binding for `ewma_gauss`, `garch_gauss`).** Panel mean
  FZ0 of the guarded wrapper below the original, bootstrap p < 0.05, at least 20 of 28 markets
  improved. Prediction: pass at all three levels for both. `hist250` is reported under M1 as
  informational only: its ES/VaR ratio is not constant, so it lies outside the class the wrapper
  recalibrates within (its own hindsight recalibration is worse than the original on the smoke
  markets); prediction: the guard stays with the original, gain about zero.
* **M2, no harm on a specialist (binding for `garch_t`, `gas_fz0`, `volpfn_t8`).** Excess of the
  guarded wrapper over the original at most B(alpha). Prediction: pass for `garch_t` and `gas_fz0`
  at all levels; for `volpfn_t8` pass at 0.01, open at 0.025 and 0.05 (see the smoke disclosure —
  the network's residual tail drifted across the 2008 regime, and a 1,000-day guard lags such a
  flip by up to two years of the four-year window).
* **M3, capture of the miscalibration (binding at 0.05 and 0.025 for `ewma_gauss`, `garch_gauss`;
  informational at 0.01).** (original − guarded) / MCB_full >= 0.5. Prediction: pass.
* **M4, variant choice (no threshold).** The ratio variant with the largest total gain over all
  forecasters and levels is recorded; the reserved-block rules are evaluated on `guard_<that
  variant>`. Prediction: open between `ratio_ftl` and `ratio_win1000`.
* **M5, binning contrast (descriptive).** `bins5` against the specialists, next to the ratio and
  guarded wrappers. Prediction: bins hurt the specialists by an order of magnitude more than the
  guarded wrapper, as in the sandbox (claim C4).
* **M6, hit rates (descriptive).** Markets whose scored hit rate lies inside the exact binomial 95%
  band, original vs wrapped. Prediction: the wrapper moves `ewma_gauss` and `ewma_t5` from mostly
  outside the band to mostly inside at 0.05.
* The pure wrapper `ratio_ftl` and the unguarded `switch_ftl` are reported under the same rules,
  marked informational: the pure wrapper's M2 excess on a good model is the price of recalibration
  under drift, and the difference to the guard is what the guard buys.

## Reserved block (M7, one pass, after the development rules are read)

Prepare the full-series copies (`calendar_full_v1` for the same 28 markets, the network's
extract on the full series), run `calibeat.market run --reserved` once, and evaluate M1-M3 with the
same thresholds on `guard_<M4 choice>` over `split == 2` (2015-01 to 2022-02, about 1,790 days,
including March 2020); the wrappers and refits continue online from 2000 without reset.
Predictions: M1 pass; M2 open for every specialist (a tail shock inside the window); M3 pass at
0.05 and 0.025.

## Pipeline smokes disclosed

Three runs on SPX and FTSE at alpha 0.05 only (`outputs/market_smoke`, `_smoke2`, `_smoke3`) were
made to check the pipeline on real inputs before this freeze. They led to three design changes,
all made before any other market or level was scored: (1) `hist250` moved from binding to
informational under M1, for the reason above; (2) the switch layer was added after the pure wrapper
raised `volpfn_t8`'s FZ0 by +0.005 (SPX) and +0.011 (FTSE) — its t8 residual tail under-covered in
2008-2010 (hit rate 6.9% at the 5% level) and covered correctly in 2011-2014 (4.7%), so the
constants learned through the crisis were too wide afterwards; (3) the cumulative switch inherited
that harm (+0.001 / +0.011), so the guarded, windowed, margined switch became the binding variant
(+0.002 / +0.006 on the two markets — still above B at 0.05 on FTSE, which is why M2 for
`volpfn_t8` at 0.05 and 0.025 is pre-registered as open rather than predicted to pass). The smoke
numbers are not the study and are not reused.

## Failure semantics

M1 or M3 failing on the panel means the wrapper does not capture real tail-law misspecification
and the paper's practical claim falls. M2 failing for a specialist with the pure wrapper but not
the guard is the expected finding (drift). M2 failing with the guard for `garch_t` or `gas_fz0`
means the no-harm layer is not good enough as built and needs a tracking (fixed-share) design
before the reserved block is touched.
