"""Publish the new follow-up without replacing any earlier comparison."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from research.cross_market_blend_20260930.run_blends import ROOT, REFERENCE, ROSTER, SEEDS, sha, summary, validate_mask, write_json
from research.cross_market_blend_20260930.statistics import validate_design, load_losses

GRAND = ROOT / 'research/outputs/intraday_extension_20260930_grand_table_final_v1'
NEW = (REFERENCE, 'crossmarket_xgb_calibrated', 'gjr+crossmarket_xgb', 'crossmarket_xgb')
LABELS = {REFERENCE: 'VolPFN + cross-market XGBoost',
          'crossmarket_xgb_calibrated': 'Cross-market XGBoost, calibrated',
          'gjr+crossmarket_xgb': 'GJR + cross-market XGBoost',
          'crossmarket_xgb': 'Cross-market XGBoost',
          'volpfn_20260930_full+ttm_rk': 'VolPFN + TTM-RK',
          'gnnhar': 'GNNHAR adaptation', 'transfer_xgboost': 'Transfer-XGBoost adaptation'}


def run(helper, comparison, statistics, out):
    helper, comparison, statistics, out = map(lambda p: Path(p).resolve(), (helper, comparison, statistics, out))
    out.mkdir(parents=True, exist_ok=False)
    pins = {}
    def pin(p):
        p = Path(p).resolve(); digest = sha(p)
        if str(p) in pins and pins[str(p)] != digest:
            raise ValueError('Input changed: ' + str(p))
        pins[str(p)] = digest
        return p
    def receipt(folder):
        for relative, digest in json.loads(pin(folder / 'ARTIFACT_RECEIPT.json').read_text()).items():
            p = (folder / relative).resolve()
            if not p.is_relative_to(folder) or sha(pin(p)) != digest:
                raise ValueError('Invalid report input artifact: ' + str(p))
    for folder in (helper, comparison, statistics, GRAND):
        receipt(folder)
    quality = json.loads(pin(helper / 'QUALITY_VERIFICATION.json').read_text())
    if quality['status'] != 'passed' or quality['checkpoints'] != 140 or quality['evaluation_rows'] != 110130:
        raise ValueError('Full helper verification required')
    for folder in (comparison, statistics, GRAND):
        provenance = json.loads(pin(folder / 'provenance.json').read_text())
        for path, digest in provenance['source_hashes'].items():
            if sha(pin(path)) != digest:
                raise ValueError('Historical or follow-up source changed: ' + path)
    mask = pd.read_csv(pin(comparison / 'common_mask.csv'), dtype={'market': str, 'date': str})
    validate_mask(mask)
    pd.testing.assert_frame_equal(mask, pd.read_csv(pin(GRAND / 'common_mask.csv'), dtype={'market': str, 'date': str}),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    design = json.loads(pin(comparison / 'REGISTERED_DESIGN.json').read_text()); validate_design(design)
    if Path(design['helper']).resolve() != helper:
        raise ValueError('Comparison and supplied helper differ')
    load_losses(comparison, mask, pin)
    table = pd.read_csv(pin(comparison / 'leaderboard.csv'))
    if set(table.model) != set(ROSTER) or len(table) != 7:
        raise ValueError('Exact complete comparison required')
    for _, row in table.iterrows():
        values = summary(pd.read_csv(pin(comparison / 'row_forecasts' / (row.model + '.csv.gz')),
                                     dtype={'market': str, 'date': str, 'seed': str}))
        for name, value in values.items():
            np.testing.assert_allclose(row[name], value, rtol=1e-12, atol=1e-12)
    for proxy in ('r2', 'rk'):
        np.testing.assert_array_equal(table[proxy + '_rank'], table[proxy + '_qlike'].rank(method='min'))
    stat = json.loads(pin(statistics / 'statistics.json').read_text())
    expected_contrasts = {(block, name, proxy) for block in (60, 20, 120)
                          for name in ROSTER if name != REFERENCE for proxy in ('r2', 'rk')}
    actual_contrasts = {(r['block'], r['comparator'], r['proxy']) for r in stat['contrasts']}
    if (stat['reference'] != REFERENCE or stat['family_slots'] != 12 or len(stat['contrasts']) != 36
            or actual_contrasts != expected_contrasts or stat['seed'] != 2026093005
            or stat['resamples'] != 20000 or Path(stat['source_comparison']).resolve() != comparison):
        raise ValueError('Complete new-family uncertainty required')
    primary = [r for r in stat['contrasts'] if r['block'] == 60]
    old = pd.read_csv(pin(GRAND / 'leaderboard.csv'))
    if old.model.duplicated().any() or old.status.eq('complete').sum() != 30:
        raise ValueError('Preserved complete historical table required')
    for name in ROSTER:
        if name not in NEW:
            prior, current = old.loc[old.model.eq(name)].iloc[0], table.loc[table.model.eq(name)].iloc[0]
            for proxy in ('r2', 'rk'):
                np.testing.assert_allclose(prior[proxy + '_qlike'], current[proxy + '_qlike'], rtol=1e-12, atol=1e-12)
    entries = []
    for name in NEW:
        row = table.loc[table.model.eq(name)].iloc[0].to_dict()
        row.update(label=LABELS[name],
                   information_group='Own and 27 foreign lagged RK' + (' + daily-return forecast' if '+' in name else ''),
                   r2_forecast_functional='Proxy-specific risk blend/calibration' if name != 'crossmarket_xgb' else 'Native RK point as r² proxy',
                   rk_forecast_functional='Proxy-specific risk blend/calibration' if name != 'crossmarket_xgb' else 'Native RK point',
                   fitting_opportunity='Fixed pre2018 helper; causal pooled prior-date calibration every20dates' if name != 'crossmarket_xgb' else 'Fixed pre2018 helper, no evaluation refits',
                   output_adapter='No return density; new retrospective procedure, not architecture isolation',
                   intraday_archive=str(comparison), daily_archive='', pathology_notes='')
        entries.append(row)
    merged = pd.concat([old, pd.DataFrame(entries)], ignore_index=True)
    for proxy in ('r2', 'rk'):
        merged[proxy + '_rank'] = merged[proxy + '_qlike'].rank(method='min')
    merged = merged.sort_values('rk_rank', na_position='last')
    merged.to_csv(out / 'extended_leaderboard.csv', index=False)
    table.sort_values('rk_rank').to_csv(out / 'focused_leaderboard.csv', index=False)
    pd.DataFrame(primary).to_csv(out / 'primary_contrasts.csv', index=False)
    receipt_data = json.loads(pin(helper / 'RECEIPT.json').read_text())
    fit_seconds = sum(t['seconds'] for t in receipt_data['trials'])
    lines = ['# Cross-market input helper follow-up', '',
             'All 140 helper fits and all 110,130 helper forecasts completed on the unchanged 28-market panel. '
             'The existing VolPFN checkpoints were not retrained. All older outputs remain preserved.', '',
             '## What this tests', '',
             'The helper supplies own-market intraday risk and separate lagged HAR1/5/22 inputs from the other 27 markets '
             '(84 predictors). Each market helper uses only its own TRAIN labels through 2017; 2018 is diagnostic, '
             'not a tuning set. The frozen daily-return VolPFN is combined with this helper using causal prior-date '
             'QLIKE fitting. Calibration-only and GJR controls use the same number of free parameters (two) '
             'and fitting dates.', '',
             'This addresses access to foreign realised-risk histories; it does not establish full GNNHAR input/training '
             'parity. GNNHAR also has a return-derived graph, pooled fitting and validation selection. Dates only enforce '
             'availability; they are not predictors. A separate no-foreign ablation would be needed to measure the '
             'specific benefit of foreign inputs.', '',
             '## Results: lower QLIKE is better', '',
             '| Procedure | Squared-return QLIKE | RK QLIKE | r² rank | RK rank |',
             '|---|---:|---:|---:|---:|']
    for _, row in table.sort_values('rk_rank').iterrows():
        lines.append(f"| {LABELS[row.model]} | {row.r2_qlike:.6f} | {row.rk_qlike:.6f} | {int(row.r2_rank)} | {int(row.rk_rank)} |")
    lines.extend(['', 'Scores average dates within each market, markets equally, then the five seed losses; '
                  'forecasts are not ensembled. Dates: 2019-01-01–2022-02-25, 22,026 market/date keys, '
                  '827 union dates. The original 17 target-defined exclusions remain. All risk forecasts are '
                  'percent-squared, with the Oxford RK conversion applied once.', '',
                  '## Registered primary uncertainty', '',
                  'Gain means comparator QLIKE minus VolPFN+helper QLIKE: positive favours VolPFN+helper. '
                  'The new 12-slot family is separate from the preserved earlier 38-slot family.', '',
                  '| Comparator | Proxy | Gain | Family 95% interval | Holm p | Seeds favouring reference | Joint conclusion |',
                  '|---|---|---:|---|---:|---:|---|'])
    for row in primary:
        conclusion = 'Reference favoured on both proxies' if row['positive_support'] else 'Comparator favoured on both proxies' if row['negative_support'] else 'Not resolved by joint rule'
        interval = row['family_interval95']
        lines.append(f"| {LABELS[row['comparator']]} | {row['proxy']} | {row['gain']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | {row['p_holm']:.6f} | {row['matched_seeds_positive']}/5 | {conclusion} |")
    lines.extend(['', '20,000 synchronized circular calendar resamples, primary block60; block20/120 sensitivities '
                  'are saved separately. Two-sided centered plus-one p-values, joint Holm and Bonferroni basic '
                  'family intervals. Joint support requires both proxies, adjusted p<.05, family interval direction '
                  'and all five corresponding seed directions. Unresolved differences are not equivalence.', '',
                  '## Verification and limits', '',
                  f"All 140 saved checkpoints replayed all 110,130 issued helper predictions. All TRAIN gates passed. "
                  f"Foreign features were used in {quality['models_using_foreign_features']}/140 models "
                  f"({quality['foreign_feature_splits_total']:.0f} tree splits); this is diagnostic, not an ablation.", '',
                  f"Helper full-run wall time: {receipt_data['wall_seconds']:.2f}s; accumulated individual fitting "
                  f"time: {fit_seconds:.2f}s. Local CPU execution; no new GPU was provisioned for this experiment.", '',
                  'The panel was already exposed when this helper was chosen. Inference is retrospective and '
                  'conditional on the saved forecasts; it excludes model training, online weight-refitting, '
                  'pretraining and research-selection uncertainty. Risk publication latency/timezone vintages '
                  'remain unverified. RK versus return risk remains a proxy assumption. PFN/GJR use second '
                  'moments for squared returns and central variance for RK; separate proxy blends do not '
                  'produce a jointly coherent density.', '',
                  'The extended leaderboard preserves all 30 earlier complete procedures and adds four new ones '
                  '(34 complete). Missing M2VN news inputs and unavailable exact Deep-RGARCH remain explicit. '
                  'Historical architecture, attribution and tail results are not replaced. No manuscript rewrite '
                  'or architecture-superiority claim is made.', '',
                  '## Files', '',
                  f'- Helper checkpoints and replay: `{helper}`',
                  f'- New causal comparison: `{comparison}`',
                  f'- New 12-slot uncertainty: `{statistics}`',
                  '- `extended_leaderboard.csv`: descriptive combined table, including input/fitting labels.',
                  '- `focused_leaderboard.csv`, `primary_contrasts.csv`: new comparison and primary uncertainty.'])
    (out / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    pin(Path(__file__)); shutil.copy2(Path(__file__), out / 'report_frozen.py')
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Report input drift: ' + path)
    write_json(out / 'provenance.json', {'source_hashes': pins, 'originals_preserved': True})
    write_json(out / 'COMPLETION.json', {'status': 'complete', 'helper_models': 140, 'helper_predictions': 110130,
               'new_procedures': 4, 'focused_procedures': 7, 'extended_complete_procedures': 34,
               'new_volpfn_training': False, 'new_gpu_resources': 0, 'old_results_preserved': True})
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(table.sort_values('rk_rank').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('helper', 'comparison', 'statistics', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); run(args.helper, args.comparison, args.statistics, args.out)
