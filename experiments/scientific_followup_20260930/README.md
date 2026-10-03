# Scientific correction and supported numerical repairs, 30 September 2026

This is a follow-up to the completed repository audit. The starting manuscript is
`research/dissertation_audited_20260929`; the separate corrected source is
`research/dissertation_scientifically_corrected_20260930`. This is a scientific
correction, not the requested later prose-style rewrite. Original data,
checkpoints, frozen implementations and outputs are preserved.

The delivery report and issue-by-issue table are under
`research/outputs/scientific_followup_20260930`. Its preservation manifest records
2,175 inputs at the start of this follow-up. The final verification compares
content hashes, rather than inferring preservation from an already dirty Git tree.

## Method versions and scope

- The corrected final five-pair VolPFN evidence retains the v2 training source,
  v3 plans, seeds and final checkpoints. No PFN retraining occurred here.
- Historical attribution and tail experiments retain their earlier checkpoint
  and preprocessing identity. The original expanding-window GARCH/GAS tail fits
  accepted failed optimizers. The convergence repair is a separate same-model,
  same-calendar sensitivity; read `OPTIMIZER_PROTOCOL.md` before using it.
- The native density GARCH comparator and later wider leaderboards use different
  `arch` fitting windows and acceptance/fallback policies. The expanding tail
  optimizer repair must not be substituted into those experiments.
- GP likelihood version 1 fixes Gaussian floating-point algebra only. Its
  nine bounded tests and actual 200-particle prefix do not complete the prescribed
  two-seed 28-market GP comparison. A full replay remains unverified.
- The theorem describes a projected residual estimator under its stated
  assumptions. The archived market safeguards implement a different estimator.
  No numerical optimizer repair removes that distinction.

## Supported local checks

From this repository root, using the retained scientific environment:

```sh
"/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python" -m research.scientific_followup_20260930.verify_supported
python3 -m research.scientific_followup_20260930.supported_checks
python3 -m research.scientific_followup_20260930.preservation verify
```

The local path above is available on the audited host. It is not a portable
installation recipe or a claim of full GPU reproduction. The observed runtime
inventory is saved separately; Moirai and the newer frontier stack use distinct
environments. Existing exact assertions are not weakened to hide backend changes.

The historical attribution import discrepancy is resolved through a supported
module launcher with a required new output directory:

```sh
python -m research.scientific_followup_20260930.attribution --help
python -m research.scientific_followup_20260930.attribution --out NEW_DIRECTORY
```

The second command executes an exploratory attribution experiment with its
original fit budget. It is not part of the checks run here and does not repair
failed driver comparisons by merely giving them a new filename.

## Experiments needed only for stronger claims

The corrected manuscript narrows these claims instead of silently changing the
method: full zero-inclusive predictive optimality; incremental auxiliary-head
benefit; superiority of optimized driver classes; a literal theorem implementation
on markets; complete GP rankings; VolPFN-specific blend synergy; worldwide novelty;
and portable fresh-clone/full GPU reproduction. The resolution table states the
specific evidence required. Replaying an inspected period is not a new holdout.
