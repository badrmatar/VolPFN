"""Study 1 of PROTOCOL_AMENDMENT_20260907.md: does the gate want a slope or a level?

Same recursion, core, fitting window, fold rotation, held-out mask and common Student-t8 scoring as
gain_surgery.py; only the state variable driving the gate changes across specifications. Every
specification is fitted three ways (distilled OLS on the network's updates, free-running
distillation, likelihood on data) and run forward as a standalone filter on the held-out groups.
Decision rules R1-R6 of the amendment are evaluated on the likelihood variant and written to
decisions.json; nothing here opens reserved outcomes.

usage: PYTHONPATH=. python3 experiments/attribution/driver_swap.py [--specs all] [--maxfev 800]
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.prior_adaptation.experiment import GROUPS, paired_calendar_interval, save, sha
from experiments.prior_adaptation.model import log_law
from experiments.attribution.gain_surgery import BURN, NU_SCORE
import experiments.attribution.drivers as D

STUDY_SPECS = [s for s in D.SPECS if s != 'readout']
FIT_END = '2011-01-01'

# Pre-registered comparisons (candidate, comparator, family). Evaluated on the likelihood variant.
COMPARISONS = [
    ('gate_d2', 'gate_level_h', 'R1'), ('gate_d2', 'gate_level_l100', 'R1'), ('gate_d2', 'gate_level_l20', 'R1'),
    ('state_dep_plus_l100', 'state_dep_gain', 'R3'), ('state_dep_plus_lev', 'state_dep_gain', 'R3'),
    ('gate_levels_5_20_100', 'gate_only', 'R3'),
    ('gate_only', 'asymmetric_gain', 'R4'), ('drift_only', 'asymmetric_gain', 'R4'),
    ('state_dep_gain', 'asymmetric_gain', 'R4'),
    ('gate_d2', 'asymmetric_gain', 'context'), ('gate_d1', 'asymmetric_gain', 'context'),
    ('gate_level_h', 'asymmetric_gain', 'context'), ('gate_level_l100', 'asymmetric_gain', 'context'),
    ('gate_level_h_pooled', 'gate_level_h', 'context'),
] + [(s, 'gate_level_l20', 'R6') for s in D.SWEEP if s != 'gate_level_l20']


def fit_and_score(frame, markets, net_h, mask_by_market, specs, maxfev, do_freerun, log, do_likelihood=True):
    rows = []; fold_detail = {}
    for held_out in GROUPS:
        train = frame[(frame.group != held_out) & (frame.next_date < FIT_END)]
        held = frame[(frame.group == held_out) & frame.in_mask]
        norm = D.norm_from(train)
        fit_packed = D.stack([D.prepare(markets[n], norm,
                                        (markets[n]['date'] < FIT_END) & (np.arange(len(markets[n]['r'])) >= BURN),
                                        net_h[n])
                              for g, ns in GROUPS.items() if g != held_out for n in sorted(ns)])
        eval_names = sorted(GROUPS[held_out])
        eval_packed = D.stack([D.prepare(markets[n], norm, np.isin(markets[n]['date'], mask_by_market[n]), net_h[n])
                               for n in eval_names])
        detail = {}
        for spec in specs:
            t0 = time.perf_counter()
            fit = D.fit_spec(train, fit_packed, spec, norm, maxfev, do_freerun=do_freerun, do_likelihood=do_likelihood)
            Xh = D.design_matrix(held, spec, norm); yh = held.g.to_numpy()
            out_r2 = 1 - np.var(yh - Xh @ fit['distilled']) / np.var(yh)
            for label in ('distilled', 'freerun', 'likelihood'):
                H = D.run_batch(eval_packed, fit[label], spec, norm)
                for i, name in enumerate(eval_names):
                    m = len(markets[name]['r']); kk = eval_packed['keep'][i, :m]
                    rows.append(pd.DataFrame({'date': markets[name]['date'][kk], 'market': name, 'group': held_out,
                                              'spec': spec, 'variant': label,
                                              'filter_common_t8': log_law(markets[name]['r'][kk], H[i, :m][kk], NU_SCORE)}))
            names = D.param_names(spec)
            detail[spec] = {
                'free_parameters': len(names),
                'coefficients_distilled': dict(zip(names, fit['distilled'].tolist())),
                'coefficients_likelihood': dict(zip(names, fit['likelihood'].tolist())),
                'coefficients_freerun': dict(zip(names, fit['freerun'].tolist())),
                'distillation_r2_in_fold': fit['distillation_r2_in_fold'], 'distillation_r2_held_out': float(out_r2),
                'likelihood_converged': fit['likelihood_converged'], 'likelihood_evaluations': fit['likelihood_evaluations'],
                'likelihood_mean_nll': fit['likelihood_mean_nll'],
                'freerun_converged': fit['freerun_converged'], 'freerun_evaluations': fit['freerun_evaluations'],
                'freerun_mean_squared_log_gap': fit['freerun_mean_squared_log_gap'],
                'freerun_mean_squared_log_gap_at_distilled': fit['freerun_mean_squared_log_gap_at_distilled'],
                'normalisation': norm, 'seconds': time.perf_counter() - t0}
            nll = fit['likelihood_mean_nll']
            log(f'  {held_out:13s} {spec:22s} p={len(names):2d}  R2 in {fit["distillation_r2_in_fold"]:6.3f} '
                f'out {out_r2:6.3f}  nll {nll if nll is not None else float("nan"):.5f}  {time.perf_counter() - t0:5.0f}s')
        fold_detail[held_out] = detail
    return pd.concat(rows, ignore_index=True), fold_detail


def decide(scored, fold_detail, reference, specs, log):
    """The pre-registered rules of the amendment, evaluated on the likelihood variant."""
    merged = scored.merge(reference[['date', 'market', 'vol_common_t8', 'uni_common_t8']],
                          on=['date', 'market'], how='inner', validate='many_to_one')
    if len(merged) != len(scored):
        raise RuntimeError('scored dates did not align one-to-one with the reference mask')
    denominator = float((reference.groupby('market').vol_common_t8.mean()
                         - reference.groupby('market').uni_common_t8.mean()).mean())
    context = {}
    for (spec, variant), block in merged.groupby(['spec', 'variant']):
        vs_uni = paired_calendar_interval(block, 'filter_common_t8', 'uni_common_t8')
        vs_net = paired_calendar_interval(block, 'filter_common_t8', 'vol_common_t8')
        context[f'{spec}|{variant}'] = {'free_parameters': D.n_params(spec),
                                        'gain_recovery': vs_uni['mean_difference'] / denominator,
                                        'mean_common_t8': float(block.groupby('market').filter_common_t8.mean().mean()),
                                        'versus_univol_common_t8': vs_uni, 'versus_network_common_t8': vs_net}
    comparisons = {}
    for variant in ('likelihood', 'distilled', 'freerun'):
        w = D.wide(scored, variant)
        for cand, comp, family in COMPARISONS:
            if cand in w.columns and comp in w.columns:
                comparisons[f'{variant}|{cand}-{comp}'] = dict(D.paired_block(w, cand, comp), family=family,
                                                                candidate=cand, comparator=comp, variant=variant)
    r1 = [k for k, v in comparisons.items() if v['variant'] == 'likelihood' and v['family'] == 'R1']
    if r1:
        for k, a in zip(r1, D.holm([comparisons[k]['bootstrap_p_two_sided'] for k in r1])):
            comparisons[k]['holm_adjusted_p'] = a

    def c(cand, comp, variant='likelihood'):
        return comparisons.get(f'{variant}|{cand}-{comp}')

    def mean_of(spec, variant='likelihood'):
        e = context.get(f'{spec}|{variant}'); return None if e is None else e['mean_common_t8']

    have = lambda *ss: all(s in specs for s in ss)
    rules = {}
    if have('gate_d2', 'gate_level_h', 'gate_level_l100', 'gate_level_l20'):
        parts = []
        for comp in ('gate_level_h', 'gate_level_l100'):
            x = c('gate_d2', comp)
            parts.append(x['mean_difference'] > 0 and x['calendar_block_interval95'][0] > 0
                         and x.get('holm_adjusted_p', 1.) < .05 and x['markets_improved'] >= 17)
        rules['R1_slope_beats_level'] = {'pass': bool(all(parts)),
                                         'detail': {comp: c('gate_d2', comp) for comp in ('gate_level_h', 'gate_level_l100', 'gate_level_l20')}}
    if have('gate_d2', 'gate_level_h', 'gate_level_l100', 'gate_level_l20', 'state_dep_gain', 'asymmetric_gain'):
        increment = mean_of('state_dep_gain') - mean_of('asymmetric_gain')
        margin = mean_of('gate_d2') - max(mean_of('gate_level_h'), mean_of('gate_level_l100'), mean_of('gate_level_l20'))
        rules['R2_material'] = {'pass': bool(margin >= .25 * increment), 'margin': float(margin),
                                'state_dependence_increment_B_minus_A': float(increment), 'threshold': .25}
    if have('state_dep_plus_l100', 'state_dep_plus_lev', 'state_dep_gain', 'gate_levels_5_20_100', 'gate_only'):
        j = c('state_dep_plus_l100', 'state_dep_gain'); k = c('state_dep_plus_lev', 'state_dep_gain')
        i = c('gate_levels_5_20_100', 'gate_only')
        covers0 = lambda x: x['calendar_block_interval95'][0] <= 0 <= x['calendar_block_interval95'][1]
        nothing = lambda x: covers0(x) or x['mean_difference'] < .0005
        rules['R3_level_adds_nothing'] = {'pass': bool(nothing(j) and nothing(k)), 'J_minus_B': j, 'K_minus_B': k}
        rules['R3_sum_to_zero_constraint_free'] = {'pass': bool(covers0(i)), 'I_minus_D': i}
    if have('gate_only', 'drift_only', 'state_dep_gain', 'asymmetric_gain'):
        a = mean_of('asymmetric_gain'); inc = mean_of('state_dep_gain') - a
        rules['R4_gate_not_drift'] = {'pass': bool(mean_of('gate_only') - a >= .5 * inc),
                                      'D_minus_A': float(mean_of('gate_only') - a),
                                      'C_minus_A': float(mean_of('drift_only') - a), 'B_minus_A': float(inc)}
    if have('gate_d2', 'gate_level_h', 'gate_level_l100'):
        fid = {}
        for s in ('gate_d2', 'gate_level_h', 'gate_level_l100'):
            gaps = [f[s]['freerun_mean_squared_log_gap'] for f in fold_detail.values()]
            fid[s] = {'freerun_gap_mean': None if any(g is None for g in gaps) else float(np.mean(gaps)),
                      'freerun_gap_at_distilled_mean': float(np.mean([f[s]['freerun_mean_squared_log_gap_at_distilled']
                                                                      for f in fold_detail.values()])),
                      'distillation_r2_held_out_mean': float(np.mean([f[s]['distillation_r2_held_out'] for f in fold_detail.values()]))}
        rules['R5_network_fidelity'] = {'best_by_freerun_gap': min(fid, key=lambda s: fid[s]['freerun_gap_at_distilled_mean']),
                                        'best_by_held_out_r2': max(fid, key=lambda s: fid[s]['distillation_r2_held_out_mean']),
                                        'detail': fid}
    if all(s in specs for s in D.SWEEP):
        curve = {str(h): mean_of(s) for h, s in zip(D.SWEEP_H, D.SWEEP)}
        finite = [s for s in D.SWEEP if s != 'gate_level_l20']
        best = max(finite, key=mean_of); x = c(best, 'gate_level_l20')
        rules['R6_finite_horizon'] = {'pass': bool(x['mean_difference'] > 0 and x['calendar_block_interval95'][0] > 0),
                                      'best_finite': best, 'curve_mean_common_t8_by_H': curve, 'best_versus_infinite': x}
    verdict = 'not evaluated'
    if 'R1_slope_beats_level' in rules and 'R2_material' in rules:
        verdict = ('slope beats level, materially' if rules['R1_slope_beats_level']['pass'] and rules['R2_material']['pass'] else
                   'slope beats level, not materially' if rules['R1_slope_beats_level']['pass'] else
                   'slope does not beat level')
    log('')
    for k, v in rules.items():
        log(f'  {k:32s} {"PASS" if v.get("pass") else ("FAIL" if "pass" in v else "-")}')
    log(f'\nVERDICT (likelihood variant, held-out groups): {verdict}')
    return {'denominator_checkpoint_advantage': denominator, 'context': context, 'comparisons': comparisons,
            'rules': rules, 'verdict': verdict}


def write_summary_md(root, result, fold_detail, specs):
    lines = ['# Driver swap: held-out common-t8 log score by specification', '',
             f'Checkpoint advantage over UniVol (denominator): {result["denominator_checkpoint_advantage"]:+.8f}', '',
             '| spec | p | variant | mean log score | recovery | vs UniVol | vs network |',
             '|---|---:|---|---:|---:|---:|---:|']
    for spec in specs:
        for variant in ('likelihood', 'distilled', 'freerun'):
            e = result['context'].get(f'{spec}|{variant}')
            if e:
                u, n = e['versus_univol_common_t8'], e['versus_network_common_t8']
                lines.append(f'| {spec} | {e["free_parameters"]} | {variant} | {e["mean_common_t8"]:+.6f} | '
                             f'{e["gain_recovery"]:+.1%} | {u["mean_difference"]:+.6f} ({u["markets_improved"]}/{u["markets"]}) | '
                             f'{n["mean_difference"]:+.6f} ({n["markets_improved"]}/{n["markets"]}) |')
    lines += ['', '## Pre-registered comparisons (likelihood variant)', '',
              '| family | candidate - comparator | difference | 95% block interval | markets | boot p | Holm p | Wilcoxon p |',
              '|---|---|---:|---|---:|---:|---:|---:|']
    for k, v in result['comparisons'].items():
        if v['variant'] != 'likelihood':
            continue
        lines.append(f'| {v["family"]} | {v["candidate"]} - {v["comparator"]} | {v["mean_difference"]:+.6f} | '
                     f'[{v["calendar_block_interval95"][0]:+.6f}, {v["calendar_block_interval95"][1]:+.6f}] | '
                     f'{v["markets_improved"]}/{v["markets"]} | {v["bootstrap_p_two_sided"]:.3f} | '
                     f'{v.get("holm_adjusted_p", float("nan")):.3f} | {v["wilcoxon_p_two_sided"]:.3f} |')
    lines += ['', '## Rules', '']
    for k, v in result['rules'].items():
        lines.append(f'* {k}: **{"PASS" if v.get("pass") else ("FAIL" if "pass" in v else "descriptive")}**')
    lines += ['', f'**Verdict:** {result["verdict"]}', '', '## Gate coefficients by fold (likelihood variant)', '']
    for fold, d in fold_detail.items():
        for spec in specs:
            gates = {k: v for k, v in d[spec]['coefficients_likelihood'].items() if k.startswith('score_x_')}
            if gates:
                lines.append(f'* {fold} {spec}: ' + ', '.join(f'{k}={v:+.4f}' for k, v in gates.items()))
    (Path(root) / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def main(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    logf = open(root / 'run.log', 'a')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    specs = STUDY_SPECS if args.specs == 'all' else [s.strip() for s in args.specs.split(',')]
    unknown = [s for s in specs if s not in D.SPECS]
    if unknown:
        raise SystemExit(f'unknown specifications: {unknown}')
    reference = pd.read_csv(args.reference, dtype={'date': str})
    mask_by_market = {m: np.sort(g.date.unique()) for m, g in reference.groupby('market')}
    here = Path(__file__).resolve().parent
    hashes = {str(p): sha(Path(p)) for p in [__file__, here / 'drivers.py', here / 'gain_surgery.py', here / 'PROTOCOL.md',
                                             here / 'PROTOCOL_AMENDMENT_20260907.md', args.reference]}
    save(root / 'protocol.json', {
        'scope': 'Study 1 of the 7 September amendment: driver swap on development data. No training, no checkpoint '
                 'selection, no reserved outcomes opened.',
        'inputs': hashes, 'specifications': {s: D.param_names(s) for s in specs}, 'comparisons': COMPARISONS,
        'nu_score': NU_SCORE, 'burn': BURN, 'fit_end': FIT_END, 'maxfev': args.maxfev, 'freerun': not args.no_freerun and not args.distilled_only,
        'distilled_only': args.distilled_only,
        'fit_window': 'pre-2011 history of the fitting groups; evaluation on held-out groups on the 2011-2014 mask'})
    started = time.perf_counter()
    markets = {n: D.load_market(D._E.DATA / (n + '.npz')) for n in sorted(n for g in GROUPS.values() for n in g)}
    frame = D.build_frame(args.extract, mask_by_market, markets)
    _, net_h = D.load_panel(frame, args.extract)
    log(f'{len(frame)} usable transitions over {frame.market.nunique()} markets; {int(frame.in_mask.sum())} on scored '
        f'dates; {len(specs)} specifications')
    scored, fold_detail = fit_and_score(frame, markets, net_h, mask_by_market, specs, args.maxfev,
                                        not args.no_freerun and not args.distilled_only, log,
                                        do_likelihood=not args.distilled_only)
    result = decide(scored, fold_detail, reference, specs, log)
    scored.to_csv(root / 'scores.csv', index=False)
    save(root / 'folds.json', fold_detail)
    save(root / 'summary.json', dict(result, elapsed_seconds=time.perf_counter() - started))
    save(root / 'decisions.json', {'rules': result['rules'], 'verdict': result['verdict']})
    write_summary_md(root, result, fold_detail, specs)
    save(root / 'status.json', {'status': 'complete', 'verdict': result['verdict'],
                                'elapsed_seconds': time.perf_counter() - started})
    log(f'done in {(time.perf_counter() - started) / 60:.1f} min; outputs in {root}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extract', default='experiments/outputs/attribution_extract_v1')
    ap.add_argument('--reference', default='experiments/outputs/prior_adaptation_mechanism_v1/scores.csv')
    ap.add_argument('--out', default='experiments/outputs/attribution_driver_swap_v1')
    ap.add_argument('--specs', default='all', help='comma list of specifications, or all')
    ap.add_argument('--maxfev', type=int, default=800, help='objective evaluations per Powell refit')
    ap.add_argument('--no-freerun', action='store_true', help='skip the free-running distillation refit (halves wall time)')
    ap.add_argument('--distilled-only', action='store_true', help='no Powell refits at all; tests and smoke runs')
    main(ap.parse_args())
