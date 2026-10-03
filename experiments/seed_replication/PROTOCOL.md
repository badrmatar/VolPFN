# Five-seed causal VolPFN replication

This specification precedes the named pilot and base runs. Source and evaluation-input hashes are frozen before either battery begins. Synthetic unit checks may exercise tiny models before that freeze; they provide implementation checks, not research comparisons. All earlier studies and the retained checkpoint remain unchanged.

## Question and headline choice

Does the newly specified causal VolPFN procedure reproduce the retained model's forecasting and transfer findings across training seeds? This is a procedural replication with changed preprocessing, not an exact historical replay or a paired experiment that isolates the effect of normalization. There is one new training arm, not ten historical/new paired-arm runs.

The headline is the arithmetic mean of the performance of **five independently trained base models**, seeds **8101–8105**, with every seed shown. It is not a mixture of their predictive densities and is not the best seed. The retained checkpoint remains the originally selected model in a separate historical row. No market outcome can select a checkpoint, alter a seed list, set a normalization option or stop the base battery for lack of gains.

## Causal input and target units

At observed position j within a task/window, s[j] is the RMS of nonzero observations through j, calculated using stable log arithmetic. An all-zero prefix has scale one. At the first observation, s[0]=abs(r[0]) when nonzero, otherwise one. The first scored next-return target is r[1]/s[0]. More generally, position j predicts r[j+1]/s[j]; the target's own observation and every later observation are excluded from s[j].

The existing eight feature meanings are retained: asinh standardized return, log squared standardized return, zero indicator, sign, three log EWMAs and log position. EWMAs of raw squared returns use half-lives 5/20/100, initialize at the first observed nonzero squared return, and skip zero updates. Before a nonzero observation, their fallback is one. Their values are divided by s[j]^2 to express them in current output units. Running scale, EWMAs and position reset inside every inference window, matching synthetic-task preparation; there is no added full-history memory. Window length 1024, stride 512, minimum returned history 100, and ownership remain fixed for market inference.

The predictive objective is raw-return negative log density: standardized NLL plus log s[j]. The auxiliary target is raw log latent variance at j+1 minus 2 log s[j]. Its weight remains 0.1. Masked stale targets retain the simulator's mask. Synthetic generation calls the original family simulators and overlays before legacy first-100 normalization; it does not reconstruct raw data from already rounded normalized observations.

This alters both training and inference, including EWMA initialization. It fixes the target-scale information violation by construction but does not identify how much any performance change comes from each altered preprocessing detail. Tests perturb future values, truncate prefixes, cover initial zeros and window boundaries, and check the density Jacobian and training/inference preparation agreement.

## Training budget, selection and failure policy

Base architecture: width256, 6 layers, 8 heads, 4 components, as in the retained base network. Each base seed receives 20,000 updates and 8,192 input positions per update; sequence lengths are 512 with probability0.4 and 1024 with probability0.6. Total allocated base input budget is 819,200,000 positions across five seeds. The scored-target count is recorded separately and need not equal that budget.

AdamW uses learning rate0.0003, betas(0.9,0.98), weight decay0.01, epsilon1e-8, warmup300, cosine decay to10%, auxiliary weight0.1, and gradient norm clipping at1. Training uses explicit MPS float32 without autocast, two CPU threads and two deterministic synthetic-data workers. The fixed simulator weights are GARCH .20, GJR .15, EGARCH .10, SV .20, roughSV .10, HMM .12, lattice .13; overlays are jumps .30, AR .30, drift .50, stale .25. There is no silent backend substitution.

The checkpoint with the smallest broad-prior raw-return validation NLL is selected, with first occurrence winning a tie. Validation uses48 fixed synthetic tasks of length1024, seed777, scored from target index100, at updates500,1000,...,20000. Selection uses the equal-series mean and never market data. This validation's raw-unit scores are not interchangeable with the legacy standardized validation scores.

All batches, losses, outputs, gradients, updated parameters and validation results must be finite. Failure records its cause, consumed updates, input positions and elapsed time; its seed is not replaced. A failed seed makes the five-seed headline unavailable; completed-seed individual results remain visible. A later fully separate repair protocol may be proposed, preserving the failed battery. Interrupted partial seeds are recorded as interrupted and not restarted as if uninterrupted. Infrastructure interruption may continue with not-yet-started frozen seeds, without selecting among completed outcomes.

## Small-model engineering pilot

Seeds7101 and7102 use the existing small architecture, 200 updates each, 1,024 input positions per update, lengths128/256 with equal probabilities, warmup20, and otherwise the same normalization/objective/optimizer. Validation uses8 synthetic tasks of length256 at updates100 and200, with the same seed777 and burn100. These runs check machinery and measure throughput; they are not tests of base-model performance.

Both must finish with finite results, selected checkpoints must reload under the normalization guard, causal probes must pass, and the evaluation adapter must pass its three-study **synthetic-only** integration route. No market predictions or scores from pilot models enter model design. Pilot failure stops escalation. Pilot success starts all five declared base seeds sequentially. Actual wall-clock estimates are reported from measured runs; a one-hour pilot or a few-day base battery is not assumed in advance. Small-model throughput is not an accurate base-model runtime estimator.

## Frozen evaluation replay

All base training and synthetic checkpoint selection finish before market evaluation starts. Forecast caches identify seed, checkpoint hash, causal specification and frozen inputs. All dates, return units, baseline records and feature variants are reused from audited earlier studies; no new vendor download or changed data-quality filter occurs here.

1. **28-index confirmation:** repeat C1–C4 native-density comparisons against GARCH/UniVol/EWMA and the corrected common-t8 comparison against GARCH on the audited mask. The historical common-t8 error is not reproduced. A four-slot Holm family is newly imposed on these replication claims; the older confirmation lacked that joint adjustment. Supporting QLIKE uses the audited strengthening study's twelve-slot family, retaining unavailable comparators. This is not a replay of every comparator in the later twenty-two-slot expanded study.
2. **Calendar screen:** keep both calendar scenarios, the same timing/feature variants and masks. Refit only the specified pooled density-scale coefficient on2011–2014 separately for each new checkpoint, then freeze it. Retain the twelve channel slots and eight timing/adaptation slots across scenarios, including unimplemented adaptive slots. The calendars remain modeled and exploratory; replication cannot turn them into verified session records.
3. **Population transfer:** retain all20 stocks and20 FX,2011–2014 development and2015–25 February2022 evaluation. Reuse the frozen pooled and per-market compact fits, including failures. Regenerate network forecasts with new causal inference. Supporting log-score families have four slots each for stocks andFX; stock equivalence at±0.003 nats is separate. The historical per-market numerical qualification and instability remain disclosed. FX remains a joint population/target change.

Available prior-checkpoint results stay in their historical rows; they do not enter the new seed mean. Identical scoring laws and masks are necessary for comparisons. Any unavailable slot stays in its declared family at p=1. All source paths, hashes and explicit comparison/mask metadata are captured in the evaluation inventory.

## Seed and sampling uncertainty; survival

For each declared contrast, compute per-market mean paired daily gains within each seed, then average markets and seeds equally. Primary intervals resample whole seeds with replacement and synchronized circular blocks of observed calendar dates, using2,000 replicates, block length60 and RNG seed927601. The sorted union of dates across all44 frozen masks is the common calendar, including FX-only dates. Missing market dates remain absent and each market's resampled mean is renormalized on its available dates. Consequently,60 means union-observed dates, not60 sessions in each individual study. This seed-aggregation calculation is new and is not numerically identical to legacy per-study intervals. Dates are synchronized across markets and seeds; market rows are not treated as independent daily replications. Conditional date-block intervals of the realized five-seed mean are a sensitivity analysis. With only five seeds, the crossed bootstrap is an approximate assessment of training variability, not a strong guarantee about unseen seeds or the original model search.

A positive claim survives the operational replication criterion only if all five individual seed means are positive, the crossed seed-mean95% interval excludes zero on the positive side, and the corresponding Holm-adjusted p-value is at most0.05. Preserve separate declared families; do not claim one overall familywise bound across every study and endpoint. Report point estimates and intervals when the conservative survival gate is not met; a missed gate is not proof of no effect.

For stock equivalence, use both TOST boundary tests and the crossed90% interval strictly inside±0.003. This is a new seed-mean equivalence question; the historical checkpoint's equivalence was never established. Missing a seed or a fixed scoring row invalidates the affected aggregate rather than averaging survivors. Stock per-market comparator failure retains its unavailable slot.

## Interpretation and later work

The transfer study's observed90% interval had width about0.00781, exceeding the0.006 width of its equivalence band. Say **realized precision was insufficient for the declared margin**. That does not establish prospective statistical power under every possible effect or data-generating process.

HKD's concentration makes level mismatch a plausible hypothesis; AMD/RRC also concentrate stock gains. Descriptive leave-one-market-out means are provided separately without refitting or changing primary membership. A level-only correction is a future protocol and cannot automatically label its remaining gap as dynamics. One fitted coefficient does not itself guarantee a stable recursive model. A post-hoc bounded scale correction is one candidate for a clean level diagnostic, to be specified before that fit.

After replication: separately specify the level-only comparator, then audit exposure and endpoints before extending stock dates beyond February2022, then consider a stabilized per-market reference. Later dates may improve precision but do not guarantee equivalence, are not automatically untouched, and may change the estimand through regime changes. No positive result is promised, and no new mechanism is inferred from seed robustness alone.
