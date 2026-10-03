# Historical tail input validation

`strict_tail_inputs.py` is an additive, read-only support loader for the retained
tail cohorts. It checks exact market/extract calendar and percentage-return
alignment, unique increasing calendar dates, lengths, finite returns and valid
split labels. It rejects a missing extract and every nonpositive or nonfinite
neural variance on the explicitly chosen scored split. Invalid unscored neural
variance is filled using the frozen evaluator's EWMA convention, and counts and
source hashes are reported. It neither changes a forecast nor substitutes an
alternative forecast on scored dates.

From the dr-vf root, validate one retained later-period cell:

```sh
python -m research.audit_support.strict_tail_inputs \
  --market-file research/data/oxfordman_barde/calendar_full_v1/SPX.npz \
  --extract-file research/outputs/attribution_extract_full_v1/SPX.npz \
  --scored-split 2
```

Use the configured project environment's Python. Add `--report NEW_PATH.json` to
save a new receipt; an existing receipt is never replaced. For development use
`calendar_development_v2`, `attribution_extract_v1` and `--scored-split 1`.
Run the small bad-input checks with:

```sh
python -m unittest research.audit_support.test_strict_tail_inputs -v
```

The Python API returns `TailInputs`: validated calendar, percentage returns,
split mask, filled variance path, unscored-fill mask and an audit receipt. Arrays
are read-only. This module has **not** been connected to or monkeypatched into
the frozen sibling `calibeat-var-es/calibeat/market.py`; that evaluator's loader
remains unchanged. Running this command is a preflight check, not proof that
the subsequent historical evaluator enforces the check itself. A future
versioned evaluator can explicitly call this loader; it should retain the
receipt and version new outputs rather than overwrite original results.
