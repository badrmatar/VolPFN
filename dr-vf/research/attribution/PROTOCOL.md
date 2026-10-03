# D1 attribution study: pre-registered protocol

Written 6 September 2026, before any result of this study was computed. Thresholds below are
fixed in advance. If a threshold is not met, the stated conclusion is the negative one; the
threshold is not to be revised after seeing results.

## Question

The saved `out_volpfn/volpfn_base_best.pt` checkpoint holds a development advantage on the
28-index Oxford-Man panel of +0.01210983 nats/return over UniVol (27/28 markets) and
+0.01714039 over rolling refit GARCH-t (28/28). Under a common zero-mean Student-t8 law imposed
on both methods' second moments the advantage is +0.01025356 (27/28), so most of it is carried
by the risk-scale forecast rather than the output density shape. **No component of the design has
been shown to produce that scale advantage.** This study asks whether the advantage is
*compressible*: whether the network's variance updating can be reproduced by an explicit rule
with few parameters, and whether that rule stands on its own as a forecaster.

This is an attribution study on development data. It is not a confirmatory evaluation. The
reserved 2015+ Oxford-Man outcomes are not opened by any script in this directory.

## Data and scoring, frozen from the previous study

* Markets: the 28 development indices in `research/prior_adaptation/experiment.py:GROUPS`.
* Returns: `research/data/oxfordman_barde/calendar_development_v2/<name>.npz`, loaded by
  `research.prior_adaptation.experiment.load_market` (percent returns, initial scale from the
  first 100 observations only).
* Scoring dates: the `length_512` mask in `research/outputs/prior_adaptation_v1/scores/`,
  2011-01-01 to 2014-12-31, 26,293 scored observations.
* Uncertainty: `research.prior_adaptation.experiment.paired_calendar_interval`, unchanged
  (60-day calendar blocks, 2000 resamples, seed 927601). These intervals are exploratory and
  conditional on the fixed checkpoint and market panel.
* Comparators: UniVol and rolling refit GARCH-t exactly as scored in that study.

Reusing the frozen harness is deliberate: it makes every number here directly comparable to the
+0.01210983 already reported, and removes the freedom to re-specify scoring after seeing results.

## Leg C: gain surgery

The checkpoint's mixture second moment `h_t` is extracted causally with `min_history=100`. The
quantity studied is the network's own log-variance update

    g_t = log h_{t+1} - log h_t

which is the response to observing `r_t`. Transitions where `h_t` and `h_{t+1}` are owned by
different sliding windows are **excluded**, because at a window boundary the change reflects a
context switch rather than a response to the shock. Window ownership is recomputed from the same
rule `predict_series` uses.

Candidate regressors are causal at time t: the standardised shock `z_t = r_t/sqrt(h_t)`, the
score term `z_t^2 - 1`, the absolute term `|z_t| - E|z_t|`, a signed leverage term, the level
`log h_t`, the disagreement between the 5/20/100-day EWMA channels the network is given, and a
zero-return indicator.

Three nested specifications are fitted, in increasing order:

1. **constant gain** — `g_t = a + b (z_t^2 - 1) + c (log h_t - mean)`. This is the GARCH/GAS form.
2. **asymmetric gain** — adds the absolute and signed leverage terms. This is Beta-t-EGARCH form.
3. **state-dependent gain** — the coefficient on the score term is itself linear in the state:
   `g_t = a(s_t) + b(s_t)(z_t^2 - 1)`, `a`, `b` affine in the EWMA disagreement and level.
   This is the only specification that is not already a standard model.

Fold structure: coefficients are fitted on three of the four geographic groups and evaluated on
the held-out group, rotating. No market contributes to the fit of its own evaluation.

### Thresholds

The reported quantity is **gain recovery**: the fraction of the checkpoint's +0.01025356
common-t8 advantage over UniVol that a standalone filter driven by the fitted rule retains, on
held-out markets, under identical common-t8 scoring.

* **Compressible** if a specification with at most 10 free parameters recovers **>= 50%**.
* **Partially compressible** if the best such specification recovers **30% to 50%**.
* **Not compressible** if the best recovers **< 30%**. In that case the conclusion of this study
  is that the checkpoint's advantage is opaque in-context computation, the network is reported as
  an empirical result without a mechanism, and leg C ends. No further specification is fitted.

Two filters are run, and both are reported whichever way the threshold falls:

* **distilled** — parameters taken from the regression that matches the network. Tests whether
  the network is compressible.
* **likelihood-fit** — the same functional form, parameters fitted by maximum likelihood on the
  2011-2014 development returns of the fitting groups only. Tests whether the *form* is a good
  model independently of the network. The likelihood-fit filter is additionally required to beat
  refit GARCH-t by **>= +0.004 nats/return on >= 20/28 markets** with a paired calendar-block
  interval excluding zero before it is described as a forecasting method rather than a diagnostic.

## Leg B: matched-budget prior ablation

Paired training runs, identical in architecture, token budget, step count, length mixture,
learning-rate schedule, contamination settings and seed, differing only in the family weights
passed to `--prior`:

* **broad** — `volpfn_prior.PRIOR_WEIGHTS` unchanged.
* **garch-only** — `garch=1.0`.

At least three seed pairs. Both arms scored on the panel with the frozen harness.

**Positive control, checked before the contrast is interpreted:** the broad arm must itself beat
UniVol by **>= +0.005 nats/return** on the panel. A small model at a reduced budget may simply be
too weak to display the effect being attributed; if the broad arm does not clear that bar, the
budget is declared insufficient, the contrast is *not* interpreted in either direction, and the
next action is a larger budget rather than a conclusion.

If the control passes, the reported quantity is the broad-minus-garch-only difference with its
paired interval, and the number of seed pairs in which the sign agrees.

## What this study cannot establish

An advantage attributed to a design component on this panel is attributed on development data
that has been inspected many times across this project's history. It is not a confirmatory
result and does not license a claim about untouched markets. A compressed rule that reproduces
the network is evidence about the network, not about the market's true conditional density,
which is unavailable. The simulator's exact filters certify approximation quality with respect to
the simulator's own generative assumptions and carry no guarantee about market data.

## Search history, stated

This checkpoint was retained after a search over at least nine mechanisms (Rao-Blackwellised
training, conditional-target training, learned finite priors, broad-model history channels,
learned chunk-posterior memory, an auditable controller, PSD filtering, circuit analysis,
weak-moment filters), all of which failed their own advancement rules. The advantage studied here
therefore carries selection exposure that no interval below adjusts for. This is recorded so that
the exposure appears in the paper rather than being discovered by a referee.

## Amendment, 6 September 2026, before any result of this study was computed

The pre-registration above places the fit of both filters on 2011-2014 data of the fitting groups.
Both fits are moved to the **pre-2011 history** of the fitting groups instead, with evaluation
unchanged on the held-out groups' 2011-2014 mask. This is strictly more conservative: the fitted
rule then shares neither markets nor dates with its evaluation, whereas the original wording shared
dates. The 2011-2014 fit is retained as a reported sensitivity. Both the distilled and the
likelihood-fit filter use the same fitting window so that the two are comparable. Thresholds are
unchanged. Recorded here rather than silently applied because the protocol is the object that makes
the search history auditable.
