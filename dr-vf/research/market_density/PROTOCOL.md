# Development density diagnostic — frozen before market scores

This is an exploratory comparison on the repeatedly examined 2011–2014 development period, not a new-method or confirmatory publication experiment. Read only `research/data/oxfordman_barde/development_v1`. All 28 previously eligible markets and both RV5 and Parzen RK are included. The existing data preparation retained valid records; horizons below count retained observations, not guaranteed consecutive exchange sessions. Do not open the reserved 2015+ files.

## Forecast experiment

For each measure separately, forecast the sum of its next 1, 5 and 22 observations using a single recursively generated daily path. Origins start at the first development observation and recur every five retained observations. Require all 22 future labels to lie within the development period; use identical origins across methods and proxies. Fit on the last 1,000 regression responses plus 22 preceding lags, refitting at the first eligible origin at least 22 observations after the last fit (therefore every 25 observations on this grid). Update GARCH residual variance with every intervening observation. No future regressors are supplied; each simulated path generates its own HAR features.

Five controls, with no tuning against development scores:

1. Bayesian log-HAR: normal/inverse-gamma posterior, prior precision 0.001 I, coefficient mean (0, .3, .3, .3), IG(shape=2, scale=.0625). Centre log measure by the current training mean. Draw coefficients and innovation variance once per entire future path, then independent Gaussian innovations conditional on those draws.
2. Bootstrap log-HAR: OLS coefficients; independent draws from centred in-sample residuals. Parameter uncertainty omitted.
3. Gaussian log-HAR-GARCH(1,1): joint Gaussian maximum likelihood using arch 7.2.0, HAR features (last, last-five mean, last-22 mean), conditional residual variance h_next=omega+alpha*residual²+beta*h.
4. Bootstrap log-HAR-GARCH: same fitted parameters and variance recursion, drawing centred, unit-second-moment standardized training residuals. This is a practical resampling control, not a claim to reproduce the paper's NIG likelihood.
5. Gaussian square-root HAR-GARCH: the published Gaussian HAR-GARCH functional form on square-root realized variance, with jointly estimated mean and GARCH coefficients. Use an affine training standardization for conditioning. Simulate the signed linear square-root variable and square it to recover variance; retain and report negative-root paths. No rejection or truncation of the published Gaussian support.

The fifth model follows the Gaussian specification in Corsi et al., *The Volatility of Realized Volatility*, CFS 2005/33 (subsequently Econometric Reviews 2008). Its paper uses different data and empirical estimation settings: this implements the model class, not a replication of its tables. Models 3–4 apply the dynamics to log variance for positive paths. These controls do not replace a later comparison with SHARP/SHARP-SV, SHAR/SHARK, or the published NIG variant. We do not call a fixed-coefficient GARCH model a dynamic-coefficient SHARP replication.

Source: https://www.econstor.eu/bitstream/10419/25467/1/515328057.PDF (sections 3.1–3.2); published DOI https://doi.org/10.1080/07474930701853616. Implementation reference: https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.GARCH.html and LS documentation. Known density and heteroskedasticity motivation is prior work, not our novelty.

## Scores and uncertainty

Use 4,096 Monte Carlo paths, fixed reproducible seeds keyed by market/measure/origin/model. Save all marginal quantiles at probabilities .01, .025, .5, .975, .99 and scores, plus fitted parameters and state metadata. Do not save all billions of draws; source and seeds reproduce them. No clipping of explosive paths: abort on nonfinite or zero variance paths. For one in twenty origins, independently repeat 4,096 draws to quantify simulation sensitivity without replacing primary forecasts.

Primary descriptive accuracy: equal-weight mean of the five quantile pinball losses, divided by horizon times the mean observed measure in that origin's fit window. Report each horizon separately. This evaluates these quantiles, not the full density. Report central-95% coverage, lower-1% and upper-1% violations, and normalized interval width. No assertion of finite predictive means is needed; in particular, exponentiated Bayesian Student predictive tails need not have finite means. Do not score sample means or claim return VaR/ES calibration from variance quantiles.

Report panel scores and per-market scores, and chronological halves 2011–2012 / 2013–2014. Define high-volatility origins using information already available: mean of last five observed measures exceeds the 90th percentile of five-observation averages in the fit window. This is a volatility-level subset, not an identified structural-break experiment. Evaluate each model against its own measure; agreement across separately fitted RV5/RK studies is sensitivity evidence, not observed latent truth.

For pooled diagnostics and paired method differences, resample contiguous blocks of 60 distinct forecast calendar dates, jointly across all markets, with 2,000 circular moving-block replicates. Use 20 and 120 date blocks as sensitivity checks. Compute observation-weighted ratios of resampled totals, preserving the point-estimate weighting. Dependence intervals are exploratory, unadjusted for multiple comparisons and conditional on this historical panel. Contemporaneous markets and overlapping horizons are not independent replications. Report market wins and counts without binomial claims. Report Monte Carlo repeat discrepancies separately.

## Execution and decision

Freeze source/protocol/input hashes before forecasting. Fail on GARCH nonconvergence after a deterministic retry, invalid data, or nonfinite draws; preserve failures. Run synthetic tests for time ordering, recursive variance and prediction agreement with arch's one-step forecast, scale equivariance, posterior draw persistence, and horizon aggregation before the panel. Audit the original 413-file manifest at completion.

Decision: if ordinary controls largely resolve the previously observed weaknesses, close the generic coherence/uncertainty novelty claim. Persistent errors can justify a narrowly specified next diagnostic only if they occur across measures, markets and time periods and are not explained by Monte Carlo error or a weak baseline. They do not establish a novel method or justify immediate large neural training. Produce a short decision report with all controls and limitations, including missing contemporary comparators.
