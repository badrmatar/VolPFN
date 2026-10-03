"""Qualified final account of the intraday extension; no manuscript replacement."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from research.intraday_extension_20260930.run_blends import REFERENCE, sha, write_json


def run(source, statistics, grand, out):
    source, statistics, grand, out = map(lambda p: Path(p).resolve(),
                                        (source, statistics, grand, out))
    out.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(source / 'leaderboard.csv')
    stats = json.loads((statistics / 'statistics.json').read_text())
    if stats['reference'] != REFERENCE or stats['family_slots'] != 38:
        raise ValueError('Report requires the fixed registered comparison')
    for name in ('gnnhar', 'qlike_har', 'transfer_xgboost'):
        if table.set_index('model').loc[name, 'status'] != 'complete':
            raise ValueError('Feasible frontier run is not complete: ' + name)
    complete = table.loc[table.status == 'complete'].set_index('model')
    ref = complete.loc[REFERENCE]
    primary = [r for r in stats['contrasts'] if r['block'] == 60]
    if len(primary) != 38:
        raise ValueError('Report requires all registered primary comparison slots')
    stats_pins = json.loads((statistics / 'provenance.json').read_text())['source_hashes']
    for path in (source / 'REGISTERED_DESIGN.json', source / 'common_mask.csv'):
        if stats_pins.get(str(path)) != sha(path):
            raise ValueError('Statistics came from a different comparison version')
    for row in primary:
        name, proxy = row['comparator'], row['proxy']
        if name in complete.index:
            path = source / 'row_forecasts' / (name + '.csv.gz')
            if stats_pins.get(str(path)) != sha(path):
                raise ValueError('Statistics forecast hashes do not match: ' + name)
            expected_gain = complete.loc[name, proxy + '_qlike'] - ref[proxy + '_qlike']
            if row['status'] != 'complete' or not math.isclose(row['gain'], expected_gain,
                                                              rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Statistical gain does not match leaderboard: ' + name)
        elif row['status'] != 'unverified/unavailable':
            raise ValueError('Missing procedure has a reported statistical result')
    reference_rows = source / 'row_forecasts' / (REFERENCE + '.csv.gz')
    if stats_pins.get(str(reference_rows)) != sha(reference_rows):
        raise ValueError('Reference forecast hashes do not match statistics')
    positive = sorted({r['comparator'] for r in primary if r['positive_support']})
    negative = sorted({r['comparator'] for r in primary if r['negative_support']})
    lookup = {(r['comparator'], r['proxy']): r for r in primary}
    comparison_files = [source / 'leaderboard.csv', source / 'REGISTERED_DESIGN.json',
                        source / 'verification.json', statistics / 'statistics.json',
                        statistics / 'provenance.json',
                        grand / 'leaderboard.csv', grand / 'verification.json']
    root = Path(__file__).resolve().parents[2]
    literature = Path(__file__).with_name('LITERATURE_REGISTER.md')
    pins = {str(p): sha(p) for p in comparison_files + [Path(__file__).resolve(), literature]}
    lines = [
        '# Intraday-information extension: final qualified report', '',
        f'**{len(complete)} completed procedures** on the same 22,026 market/date keys, '
        '28 markets and 827 union dates, 2019-01-01–2022-02-25. '
        'All five registered stochastic seeds are retained; deterministic recipes have one run.', '',
        f'The fixed VolPFN–TTM reference ranks **{int(ref.r2_rank)} on squared-return QLIKE** '
        f'and **{int(ref.rk_rank)} on RK QLIKE** in this extension. '
        f'Its losses are **{ref.r2_qlike:.6f} / {ref.rk_qlike:.6f}**. '
        'Ranks are descriptive and are separate from the conditional inference below.', '',
        'These rank **complete forecasting procedures**, not architectures. They establish '
        'neither VolPFN architecture superiority nor inferiority. The reference is a frozen '
        'daily-return PFN combined with RK-fed TTM. Seeds are scored separately, then dates '
        'within market, markets and seed losses are averaged equally; no forecast ensemble is formed.', '',
        f'Lowest squared-return loss: **{complete.r2_qlike.idxmin()}**; '
        f'lowest RK loss: **{complete.rk_qlike.idxmin()}**. '
        'A valid stronger comparator must remain in the account, even when it changes the headline. '
        'These results do not support a claim that VolPFN defeats every recent intraday method.', '',
        '## Focused results', '',
        '| Procedure | r² QLIKE | RK QLIKE | Seeds | r² rank | RK rank |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in complete.sort_values('rk_qlike').iterrows():
        lines.append(f'| {name} | {row.r2_qlike:.6f} | {row.rk_qlike:.6f} | '
                     f'{int(row.seeds)} | {int(row.r2_rank)} | {int(row.rk_rank)} |')
    lines += ['', 'Lower is better. Loss differences are **not percentages of forecast error**; '
              'QLIKE levels change by a common additive constant under unit changes. '
              'No cross-paper published percentage is transplanted into this table.', '',
              f'[One combined descriptive leaderboard]({grand / "leaderboard.csv"}) '
              'also includes the existing daily-only primary models on identical rows. '
              'Its information-group labels must accompany any combined ranking; '
              'the older compact/ablation/failed historical arms remain in their separate archives.', '',
              '## Conditional uncertainty', '',
              'Primary block: 60 union dates. 20,000 synchronized circular calendar resamples, '
              'seed 2026093004, a fixed joint 38-slot family with unavailable models retained at p=1. '
              'Gains are comparator loss minus reference loss; positive favours the reference. '
              'Ordinary 95% intervals and Bonferroni family intervals are distinct. '
              '20/120-date results are sensitivities, not alternatives from which to select a favourable test.', '',
              '| Comparator | Proxy | Gain | Ordinary 95% interval | Family 95% interval | Holm p |',
              '|---|---|---:|---|---|---:|']
    for name in ('gnnhar', 'qlike_har', 'transfer_xgboost', 'ttm_rk',
                 'gjr_garch_t+ttm_rk', 'volpfn_20260926+ttm_rk'):
        for proxy in ('r2', 'rk'):
            row = lookup[name, proxy]
            if row['status'] != 'complete':
                raise ValueError('Incomplete primary contrast: ' + name)
            interval = lambda v: f'[{v[0]:.6f}, {v[1]:.6f}]'
            lines.append(f'| {name} | {proxy} | {row["gain"]:.6f} | '
                         f'{interval(row["interval95"])} | '
                         f'{interval(row["family_interval95"])} | {row["p_holm"]:.6g} |')
    lines += ['', 'Support requires both proxies, the adjusted test, the family interval and '
              'all five fixed reference-seed directions to agree.', '',
              'Comparators with supported higher loss than the reference: ' +
              (', '.join(positive) or 'none') + '.', '',
              'Comparators with supported lower loss than the reference: ' +
              (', '.join(negative) or 'none') + '.', '',
              'These statements are **conditional on the saved forecasts and fixed trained models**. '
              'They exclude training-population, online refit, pretraining, calibration and research-selection '
              'uncertainty. The panel has already been exposed; this is not an untouched confirmation sample. '
              'The combined grand leaderboard does not create a larger inferential claim or silently '
              'change this registered family.', '',
              '## What was implemented and checked', '',
              '- Reused frozen September 30 full and September 26 VolPFN forecasts, with distinct names. '
              'No VolPFN or intraday-input transformer was retrained.',
              '- Rebuilt raw-component log-linear blends. Proxy-specific weights and intercepts use '
              'only completed earlier dates, refitted every 20 union dates after a 60-date burn-in. '
              'Fits pool earlier market-day labels across all 28 markets. Matched GJR blends '
              'receive the same fitting opportunity. The intercept is online calibration.',
              '- GNNHAR: author graph/neural/loss core, TRAIN through 2017, whole-2018 checkpoint selection, '
              'then frozen weights/graph. Five seeds; exact checkpoint reconstruction and chronology verified. '
              'The original seed-8104 failed fit remains preserved. Its same training trajectory was '
              'continued under a full-TRAIN constant-baseline qualification gate; no seed was replaced.',
              '- QLIKE-HAR: deterministic, native RK-level target, daily rolling 1,000 prior target-session '
              'fits; all 22,026 fits converged and were no worse than their TRAIN constant '
              'objective within the declared tolerance. '
              'Strict positivity and retry handling are checked. The repair produced identical v1/v2 forecasts.',
              '- Transfer-XGBoost: pinned author standard HAR-input branch; exact DTW selects prior '
              'source segments from the other 27 markets. Five seeds, refit every five own-market sessions, '
              'pinned XGBoost 1.7.1 and 40 exact-method trees. Source labels must precede each forecast block. '
              'Internal author scaling is undone once before scoring.',
              '- Exact masks, native calendars, all declared seeds, positive forecasts, loss recomputation '
              'and equal-market/seed summaries are checked. Missing RK targets never shrink model-specific coverage. '
              'Original decimal returns are converted ×100 once, RK ×10,000 once; saved native forecasts '
              'are already in percent-squared units.', '',
              'The runnable frontier procedures are **documented Oxford-Man/RK adaptations**, '
              'not replicas of each paper’s original assets, fitting schedule and headline margins. '
              'GNNHAR and transfer-XGBoost can use other markets’ histories; own-market TTM/HAR '
              'components and blends with pooled weight fitting have different information. '
              'Session-date chronology does not establish historical RK publication latency or '
              'vintage availability. Fixed-fit versus rolling/online fitting '
              'opportunities also differ. This comparison does not isolate architecture.', '',
              'A blend-versus-raw-TTM gain alone cannot isolate the value of VolPFN information '
              'from the fitted intercept/scale adjustment. The matched GJR–TTM control is informative '
              'about the choice of added daily forecaster, but a TTM-only causal calibration control '
              'would directly address calibration-only gains. That control was not in this fixed roster.', '',
              'Saved return-moment models use M2 for squared-return QLIKE and central variance for RK QLIKE. '
              'Native RK models use their same issued risk prediction for both proxy scores, without '
              'inventing a return mean or density. RK equality with open-to-close return risk remains '
              'an unverified proxy assumption. Separately fitted blend forecasts do not form one coherent density.', '',
              'Existing HAR/Moirai floor events and extreme EGARCH outputs remain part of their retained '
              'issued procedures. Their huge losses cannot establish that those model families are '
              'intrinsically poor. The new QLIKE-HAR and qualified graph comparator are stronger useful checks.', '',
              '## All requested literature retained', '',
              f'[Complete literature register]({literature}) records all five requested sources: '
              'GNNHAR, transfer-XGBoost, M2VN, QLIKE-trained HAR and Brini’s foundation-model study. '
              'The latter motivates the TTM comparator; existing TTM-RK forecasts are reused, '
              'and its original 50-asset study is not replicated.', '',
              '**M2VN remains unverified.** Complete timestamped news embeddings and required market '
              'predictors are absent for these 28 indices, and its published standard-deviation target '
              'and stock/calendar design differ. No news was fabricated or incomplete surrogate relabelled.', '',
              '**DeepRGARCH remains unverified.** Its additional feasibility slot requires a repaired '
              'causal Bayesian SMC adapter and a timed native-particle pilot. MLE/neural training cannot '
              'silently replace that method. Its unavailable slot remains in the multiplicity family.', '',
              '## Remaining decisions', '',
              '**Scope confirmed by the user: no new intraday VolPFN training.** None was started. '
              'The current work finishes the existing blends, benchmark comparison and report.', '',
              'The following remain documented possibilities outside this finite run:', '',
              '1. Whether to obtain compatible point-in-time news/predictor data for full M2VN or '
              'use a separately labelled stock population. That would be a new comparison.',
              '2. Whether to implement and time the exact repaired DeepRGARCH SMC procedure.',
              '3. Whether to add a newly registered blend with a stronger helper. Do not choose '
              'it on this exposed test panel and call the result independent confirmation.', '',
              '4. Whether to register a TTM-only causal calibration control to isolate the '
              'incremental-information claim more directly.', '',
              '## Evidence and reproduction', '',
              f'- [Complete scored rows and causal blend weights]({source / "REPORT.md"}).',
              f'- [All registered statistical contrasts]({statistics / "contrasts.csv"}).',
              f'- [Combined descriptive table and verification]({grand / "verification.json"}).',
              f'- [Protocol]({Path(__file__).with_name("PROTOCOL.md")}).',
              '- Source/output hashes and frozen scripts accompany each new output. '
              'Existing data, checkpoints, original results, failed fits and earlier receipts were not overwritten.',
              '- All new fitting and scoring in this extension ran locally on CPU. No new paid GPU '
              'was provisioned; this is not a statement about earlier overnight programme charges.',
              '- Reproduce using the pinned frontier_v1 runtime, run_blends with the final extras '
              'manifest, statistics, grand_table and this report module, each with a fresh --out path. '
              'Frozen source copies are authoritative for their original result version.']
    (out / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    shutil.copy2(literature, out / literature.name)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Report input changed: ' + path)
    write_json(out / 'PROVENANCE.json', {'source_hashes': pins, 'originals_preserved': True,
                                        'manuscript_style_rewrite': False})
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    print(out / 'FINAL_REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'statistics', 'grand', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.source, args.statistics, args.grand, args.out)
