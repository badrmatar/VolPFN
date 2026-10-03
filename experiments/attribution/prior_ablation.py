"""Study 2 of PROTOCOL_AMENDMENT_20260907.md: which training priors produce the term-structure gate?

Leg B of PROTOCOL.md (broad versus garch-only, matched budget, >= 3 seed pairs, positive control)
plus the mechanism readout of the amendment: every arm's checkpoint is extracted, its own
log-variance updates are regressed on the core and the slope, level and gate terms, and each
coefficient carries a market-cluster bootstrap interval.

Subcommands (run from the dr-vf root, with PYTHONPATH=.):
  plan        write the matched arm list to <out>/plan.json
  run         train -> extract -> gain surgery (distilled) per arm; resumable, skips finished stages
  summarise   mechanism readout, panel advantage, positive control, contrasts -> summary.json, SUMMARY.md

Source edits this study relies on (originals in code_backups/20260907_driver_swap_ablation/):
  volpfn_prior.py   + sim_garch_fixed and SIMULATORS['garch_fixed'] (PRIOR_WEIGHTS/FAMILIES untouched)
  volpfn.py         + `--overlay` option of `train`, default None = current behaviour
  extract.py        torch-dependent imports moved inside run(), so window_owner imports without torch
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.prior_adaptation.experiment import GROUPS, paired_calendar_interval, save, sha
from experiments.prior_adaptation.model import log_law
from experiments.attribution.gain_surgery import NU_SCORE
import experiments.attribution.drivers as D

NO_OVERLAY = 'jumps=0,ar=0,drift=0,stale=0'
FAMILIES_ALONE = ['gjr', 'egarch', 'sv', 'hmm', 'lattice', 'rough']
READOUT_TERMS = ['score_x_d1', 'score_x_d2', 'score_x_lev', 'score_x_l100', 'd1', 'd2']
BOOT_SEED, BOOT_DRAWS = 20260907, 500


def make_plan(args):
    common = dict(size=args.size, steps=args.steps, tokens=args.tokens, lengths=args.lengths, lr=args.lr,
                  warmup=args.warmup, aux=args.aux, eval_every=args.eval_every, workers=args.workers, device=args.device)
    arms = []
    for s in args.seeds:
        arms += [dict(name=f'broad_s{s}', tier=1, family='broad', prior=None, overlay=None, seed=s),
                 dict(name=f'garch_s{s}', tier=1, family='garch', prior='garch=1.0', overlay=None, seed=s),
                 dict(name=f'garch_fixed_s{s}', tier=1, family='garch_fixed', prior='garch_fixed=1.0', overlay=None, seed=s)]
    s = args.seeds[0]
    arms += [dict(name=f'{f}_s{s}', tier=2, family=f, prior=f'{f}=1.0', overlay=None, seed=s) for f in FAMILIES_ALONE]
    arms += [dict(name=f'broad_clean_s{s}', tier=3, family='broad_clean', prior=None, overlay=NO_OVERLAY, seed=s),
             dict(name=f'garch_clean_s{s}', tier=3, family='garch_clean', prior='garch=1.0', overlay=NO_OVERLAY, seed=s)]
    return {'common': common, 'arms': arms, 'written': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'scope': 'Matched-budget prior-family ablation with mechanism readout (amendment of 7 Sep 2026). '
                     'Development panel only; no reserved outcomes.'}


def train_cmd(arm, common, ckpt_dir):
    cmd = [sys.executable, 'src/volpfn/model.py', 'train', '--size', common['size'], '--steps', str(common['steps']),
           '--tokens', str(common['tokens']), '--lengths', common['lengths'], '--lr', str(common['lr']),
           '--warmup', str(common['warmup']), '--aux', str(common['aux']), '--eval-every', str(common['eval_every']),
           '--workers', str(common['workers']), '--device', common['device'], '--seed', str(arm['seed']),
           '--out', str(ckpt_dir), '--name', arm['name']]
    if arm['prior']:
        cmd += ['--prior', arm['prior']]
    if arm['overlay']:
        cmd += ['--overlay', arm['overlay']]
    return cmd


def training_health(log_jsonl):
    """Best and last validation nll from the trainer's jsonl log; flags a NaN blow-up."""
    if not Path(log_jsonl).exists():
        return {'evals': 0}
    rows = [json.loads(l) for l in Path(log_jsonl).read_text().splitlines() if l.strip()]
    vals = [r['all']['nll'] for r in rows]
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return {'evals': len(rows), 'best_val_nll': min(finite) if finite else None,
            'last_val_nll': vals[-1] if vals else None, 'nan_seen': any(v is None or not np.isfinite(v) for v in vals)}


def run_arms(args):
    root = Path(args.out); plan = json.loads((root / 'plan.json').read_text()); common = plan['common']
    ckpt_dir = root / 'ckpt'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    tiers = {int(t) for t in args.tiers.split(',')}
    wanted = [a for a in plan['arms'] if a['tier'] in tiers and (not args.arms or a['name'] in args.arms.split(','))]
    logf = open(root / 'run.log', 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()

    log(f'{len(wanted)} arms in tiers {sorted(tiers)}: ' + ', '.join(a['name'] for a in wanted))
    for arm in wanted:
        name = arm['name']; best = ckpt_dir / f'{name}_best.pt'; last = ckpt_dir / f'{name}.pt'; arm_dir = root / 'arms' / name
        arm_dir.mkdir(parents=True, exist_ok=True); status_path = arm_dir / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not best.exists() or status.get('train') != 'complete':
            log(f'{name}: training ({common["size"]}, {common["steps"]} steps, seed {arm["seed"]}, prior {arm["prior"]}, overlay {arm["overlay"]})')
            t0 = time.perf_counter()
            with open(arm_dir / 'train.log', 'w') as f:
                rc = subprocess.call(train_cmd(arm, common, ckpt_dir), stdout=f, stderr=subprocess.STDOUT)
            health = training_health(ckpt_dir / f'{name}_log.jsonl')
            if rc != 0 or not best.exists():
                log(f'{name}: TRAINING FAILED (rc {rc}); see {arm_dir / "train.log"}'); status['train'] = 'failed'
                save(status_path, status); continue
            # Matched arms are compared at the end of the same schedule (the last checkpoint), because
            # selecting the best step on the broad validation set would favour the broad arm; a run whose
            # final evaluation is NaN falls back to its best checkpoint and is flagged.
            use_last = args.checkpoint_rule == 'last' and last.exists() and not (health.get('last_val_nll') is None or not np.isfinite(health['last_val_nll']))
            chosen = last if use_last else best
            status.update(train='complete', train_seconds=time.perf_counter() - t0, health=health, checkpoint=str(chosen),
                          checkpoint_rule=args.checkpoint_rule, fell_back_to_best=bool(args.checkpoint_rule == 'last' and not use_last))
            save(status_path, status); log(f'{name}: trained in {(time.perf_counter() - t0) / 60:.1f} min; {health}; using {chosen.name}')
        ext = arm_dir / 'extract'
        if status.get('extract') != 'complete':
            from experiments.attribution.extract import run as extract_run
            if ext.exists():
                shutil.rmtree(ext)
            t0 = time.perf_counter(); extract_run(argparse.Namespace(checkpoint=status.get('checkpoint', str(best)), out=str(ext)))
            status.update(extract='complete', extract_seconds=time.perf_counter() - t0); save(status_path, status)
            log(f'{name}: extracted in {time.perf_counter() - t0:.0f}s')
        gs = arm_dir / 'gain_surgery'
        if status.get('surgery') != 'complete':
            from experiments.attribution.gain_surgery import main as surgery_main
            if gs.exists():
                shutil.rmtree(gs)
            t0 = time.perf_counter()
            surgery_main(argparse.Namespace(extract=str(ext), reference=args.reference, out=str(gs),
                                            garch_common=args.garch_common, maxfev=800, skip_likelihood=True))
            status.update(surgery='complete', surgery_seconds=time.perf_counter() - t0); save(status_path, status)
            log(f'{name}: gain surgery (distilled) in {time.perf_counter() - t0:.0f}s')
    log('run finished')


# ----------------------------------------------------------------------------- summarise
def readout(frame, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """OLS of an arm's own log-variance updates on the readout specification (core + d1, d2 drift +
    score x {d1, d2, lev, l100} gates), with a market-cluster bootstrap: markets are resampled with
    replacement and the regression refitted, so the interval reflects between-market variation."""
    norm = D.norm_from(frame)
    X = D.design_matrix(frame, 'readout', norm); y = frame.g.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); names = D.param_names('readout')
    r2 = 1 - np.var(y - X @ beta) / np.var(y)
    markets = frame.market.to_numpy(); uniq = np.unique(markets)
    idx = {m: np.flatnonzero(markets == m) for m in uniq}
    rng = np.random.default_rng(seed); boot = np.empty((draws, len(beta)))
    for b in range(draws):
        rows = np.concatenate([idx[m] for m in rng.choice(uniq, size=len(uniq), replace=True)])
        boot[b], *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
    coef = {}
    for j, n in enumerate(names):
        lo, hi = np.quantile(boot[:, j], [.025, .975]); agree = float(np.mean(np.sign(boot[:, j]) == np.sign(beta[j])))
        coef[n] = {'estimate': float(beta[j]), 'interval95': [float(lo), float(hi)], 'sign_agreement': agree,
                   'loads': bool((lo > 0 or hi < 0) and agree >= .95)}
    return {'coefficients': coef, 'r2': float(r2), 'transitions': int(len(y)), 'markets': int(len(uniq)),
            'normalisation': {k: v for k, v in norm.items() if k.endswith('_sd') or k == 'logh_mean'},
            'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'markets'}}


def panel_scores(extract_root, reference, markets):
    """The arm's own common-t8 log score on every scored (date, market) of the reference mask."""
    mask = {m: np.sort(g.date.unique()) for m, g in reference.groupby('market')}
    parts = []
    for name in markets:
        a = np.load(Path(extract_root) / f'{name}.npz', allow_pickle=False)
        date, r, h = a['date'].astype('U10'), a['r'], a['h']
        sel = np.isin(date, mask[name]) & np.isfinite(h) & (h > 0)
        parts.append(pd.DataFrame({'date': date[sel], 'market': name, 'arm_common_t8': log_law(r[sel], h[sel], NU_SCORE)}))
    frame = pd.concat(parts, ignore_index=True)
    merged = frame.merge(reference[['date', 'market', 'uni_common_t8', 'vol_common_t8']], on=['date', 'market'],
                         how='inner', validate='one_to_one')
    return merged, int(len(reference) - len(merged))


def summarise(args):
    root = Path(args.out); plan = json.loads((root / 'plan.json').read_text())
    reference = pd.read_csv(args.reference, dtype={'date': str})
    names = sorted(n for g in GROUPS.values() for n in g)
    markets = {n: D.load_market(D._E.DATA / (n + '.npz')) for n in names}
    mask = {m: np.sort(g.date.unique()) for m, g in reference.groupby('market')}
    arms = []
    if args.include_base and Path(args.base_extract).exists():
        arms.append({'name': 'base_checkpoint', 'tier': 0, 'family': 'broad (4.75M, 20k steps)', 'seed': None,
                     'extract': args.base_extract, 'status': {'train': 'complete', 'extract': 'complete'}})
    for arm in plan['arms']:
        sp = root / 'arms' / arm['name'] / 'status.json'
        if sp.exists():
            st = json.loads(sp.read_text())
            if st.get('extract') == 'complete':
                arms.append(dict(arm, extract=str(root / 'arms' / arm['name'] / 'extract'), status=st))
    if not arms:
        raise SystemExit('no completed arms to summarise')
    results = {}; panel = {}
    for arm in arms:
        name = arm['name']; t0 = time.perf_counter()
        frame = D.build_frame(arm['extract'], mask, markets)
        rd = readout(frame)
        scores, missing = panel_scores(arm['extract'], reference, names)
        panel[name] = scores
        adv = paired_calendar_interval(scores, 'arm_common_t8', 'uni_common_t8')
        vs_base = paired_calendar_interval(scores, 'arm_common_t8', 'vol_common_t8')
        entry = {'family': arm['family'], 'tier': arm['tier'], 'seed': arm['seed'], 'health': arm['status'].get('health'),
                 'checkpoint': arm['status'].get('checkpoint'), 'fell_back_to_best': arm['status'].get('fell_back_to_best'),
                 'advantage_over_univol_common_t8': adv, 'versus_base_network_common_t8': vs_base,
                 'scored_rows_missing': missing, 'readout': rd}
        gs = root / 'arms' / name / 'gain_surgery' / 'summary.json'
        if gs.exists():
            g = json.loads(gs.read_text())['results']
            for key in ('state_dep_gain|distilled', 'asymmetric_gain|distilled', 'constant_gain|distilled'):
                if key in g:
                    d = g[key]['versus_univol_common_t8']['mean_difference']
                    entry[f'standalone_{key.split("|")[0]}_vs_univol'] = d
                    entry[f'standalone_{key.split("|")[0]}_recovery_of_own_advantage'] = (d / adv['mean_difference']
                                                                                          if adv['mean_difference'] > 0 else None)
        results[name] = entry
        c = rd['coefficients']
        print(f'{name:22s} adv {adv["mean_difference"]:+.6f} ({adv["markets_improved"]}/28)  '
              + '  '.join(f'{t}={c[t]["estimate"]:+.4f}{"*" if c[t]["loads"] else " "}' for t in READOUT_TERMS)
              + f'  R2 {rd["r2"]:.3f}  {time.perf_counter() - t0:.0f}s', flush=True)
    # positive control and contrasts (leg B of PROTOCOL.md)
    broad = [n for n, e in results.items() if e['family'] == 'broad']
    control_mean = float(np.mean([results[n]['advantage_over_univol_common_t8']['mean_difference'] for n in broad])) if broad else None
    control = {'broad_arms': broad, 'mean_advantage': control_mean, 'threshold': .005,
               'pass': bool(control_mean is not None and control_mean >= .005)}
    contrasts = {}
    for a_fam, b_fam in (('broad', 'garch'), ('broad', 'garch_fixed'), ('garch', 'garch_fixed'), ('broad_clean', 'garch_clean'),
                         ('broad', 'broad_clean'), ('garch', 'garch_clean')):
        per_seed = []
        for s in sorted({e['seed'] for e in results.values() if e['seed'] is not None}):
            an = f'{a_fam}_s{s}'; bn = f'{b_fam}_s{s}'
            if an in panel and bn in panel:
                m = panel[an][['date', 'market', 'arm_common_t8']].merge(
                    panel[bn][['date', 'market', 'arm_common_t8']], on=['date', 'market'], suffixes=('_a', '_b'))
                per_seed.append(dict(paired_calendar_interval(m, 'arm_common_t8_a', 'arm_common_t8_b'), seed=s))
        if per_seed:
            signs = [x['mean_difference'] > 0 for x in per_seed]
            contrasts[f'{a_fam}_minus_{b_fam}'] = {'per_seed': per_seed, 'mean_over_seeds': float(np.mean([x['mean_difference'] for x in per_seed])),
                                                   'seed_pairs': len(per_seed), 'sign_agreement': int(max(sum(signs), len(signs) - sum(signs))),
                                                   'interpreted': control['pass']}
    # predictions of the amendment
    def loads(name, term='score_x_d2'):
        return results[name]['readout']['coefficients'][term]['loads']
    def sign(name, term='score_x_d2'):
        return float(np.sign(results[name]['readout']['coefficients'][term]['estimate']))
    fam = lambda f: [n for n, e in results.items() if e['family'] == f]
    predictions = {}
    # drift predictions (a')-(d') of the 7 Sep addendum, written after Study 1 decided R4 for the drift
    if fam('garch_fixed'):
        predictions['a_prime_pinned_garch_no_slope_drift'] = {'pass': not any(loads(n, 'd1') or loads(n, 'd2') for n in fam('garch_fixed')),
                                                              'detail': {n: {t: (loads(n, t), sign(n, t)) for t in ('d1', 'd2')} for n in fam('garch_fixed')}}
    if fam('garch'):
        predictions['b_prime_random_garch_drift_loads'] = {'pass': all(loads(n, 'd1') and sign(n, 'd1') < 0 for n in fam('garch')),
                                                           'detail': {n: {t: (loads(n, t), sign(n, t)) for t in ('d1', 'd2')} for n in fam('garch')}}
    if broad:
        predictions['d_prime_broad_drift_loads'] = {'pass': all(loads(n, 'd1') and sign(n, 'd1') < 0 for n in broad),
                                                    'detail': {n: {t: (loads(n, t), sign(n, t)) for t in ('d1', 'd2')} for n in broad}}
    if fam('garch_fixed'):
        predictions['a_pinned_garch_no_slope_gate'] = {'pass': not any(loads(n) for n in fam('garch_fixed')),
                                                       'level_gate_loads': {n: loads(n, 'score_x_lev') for n in fam('garch_fixed')}}
    if fam('garch'):
        predictions['b_random_garch_slope_gate'] = {'loads': {n: loads(n) for n in fam('garch')}, 'either_way': True}
    regime = [n for f in ('hmm', 'lattice', 'rough', 'sv') for n in fam(f)]
    if regime and broad:
        ref_sign = sign(broad[0])
        predictions['c_regime_families_slope_gate'] = {'pass': sum(loads(n) and sign(n) == ref_sign for n in regime) >= max(3, len(regime) - 1),
                                                       'detail': {n: (loads(n), sign(n)) for n in regime}}
    if broad:
        predictions['d_broad_slope_gate'] = {'pass': all(loads(n) for n in broad) and len({sign(n) for n in broad}) == 1,
                                             'detail': {n: (loads(n), sign(n)) for n in broad}}
    if 'base_checkpoint' in results:
        predictions['base_checkpoint_gate_loads'] = {'score_x_d2': results['base_checkpoint']['readout']['coefficients']['score_x_d2']}
    out = {'results': results, 'positive_control': control, 'contrasts': contrasts, 'predictions': predictions,
           'plan': plan['common'], 'readout_terms': READOUT_TERMS}
    save(root / 'summary.json', out)
    write_md(root, out)
    print(f'\npositive control: {"PASS" if control["pass"] else "FAIL"} (mean broad advantage {control_mean})')
    for k, v in predictions.items():
        print(f'  {k:36s} {v.get("pass", v.get("loads", "-"))}')
    return out


def write_md(root, out):
    L = ['# Prior-family ablation: mechanism readout', '', f'Budget: {out["plan"]}', '',
         '## Arms', '', '| arm | family | seed | best val nll | NaN | adv vs UniVol (t8) | markets | vs base network | distilled rule recovery of own adv |',
         '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for n, e in out['results'].items():
        h = e['health'] or {}; a = e['advantage_over_univol_common_t8']; b = e['versus_base_network_common_t8']
        rec = e.get('standalone_state_dep_gain_recovery_of_own_advantage')
        L.append(f'| {n} | {e["family"]} | {e["seed"]} | {h.get("best_val_nll", float("nan")) if h else float("nan"):.4f} | '
                 f'{"yes" if h.get("nan_seen") else "no"} | {a["mean_difference"]:+.6f} [{a["calendar_block_interval95"][0]:+.5f}, {a["calendar_block_interval95"][1]:+.5f}] | '
                 f'{a["markets_improved"]}/{a["markets"]} | {b["mean_difference"]:+.6f} | {"" if rec is None else f"{rec:+.1%}"} |')
    L += ['', '## Mechanism readout (OLS on the arm\'s own updates, per sd of the state variable; * = loads, market bootstrap)', '',
          '| arm | ' + ' | '.join(out['readout_terms']) + ' | R2 |', '|---|' + '---:|' * (len(out['readout_terms']) + 1)]
    for n, e in out['results'].items():
        c = e['readout']['coefficients']
        L.append(f'| {n} | ' + ' | '.join(f'{c[t]["estimate"]:+.4f}{"*" if c[t]["loads"] else ""} [{c[t]["interval95"][0]:+.4f}, {c[t]["interval95"][1]:+.4f}]'
                                          for t in out['readout_terms']) + f' | {e["readout"]["r2"]:.3f} |')
    pc = out['positive_control']
    L += ['', f'## Positive control: **{"PASS" if pc["pass"] else "FAIL"}** (mean broad advantage {pc["mean_advantage"]}, threshold +0.005)', '',
          '## Contrasts (paired by seed; interpreted only if the control passes)', '',
          '| contrast | seed pairs | mean difference | sign agreement | per seed |', '|---|---:|---:|---:|---|']
    for k, v in out['contrasts'].items():
        L.append(f'| {k} | {v["seed_pairs"]} | {v["mean_over_seeds"]:+.6f} | {v["sign_agreement"]}/{v["seed_pairs"]} | '
                 + '; '.join(f's{x["seed"]}: {x["mean_difference"]:+.6f} [{x["calendar_block_interval95"][0]:+.5f}, {x["calendar_block_interval95"][1]:+.5f}]' for x in v['per_seed']) + ' |')
    L += ['', '## Predictions of the amendment', '']
    for k, v in out['predictions'].items():
        L.append(f'* {k}: {json.dumps(v, default=str)}')
    (Path(root) / 'SUMMARY.md').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['plan', 'run', 'summarise'])
    ap.add_argument('--out', default='experiments/outputs/prior_ablation_v1')
    ap.add_argument('--reference', default='experiments/outputs/prior_adaptation_mechanism_v1/scores.csv')
    ap.add_argument('--garch-common', default='experiments/outputs/attribution_garch_common_v1/garch_common.csv')
    ap.add_argument('--base-extract', default='experiments/outputs/attribution_extract_v1')
    ap.add_argument('--include-base', action='store_true', default=True)
    ap.add_argument('--size', default='small'); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--tokens', type=int, default=8192); ap.add_argument('--lengths', default='256:0.3,512:0.5,1024:0.2')
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--aux', type=float, default=0.1); ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--device', default='auto')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--tiers', default='1,2,3', help='which tiers to run, e.g. 1 or 1,2')
    ap.add_argument('--arms', default=None, help='comma list of arm names to run (subset of the plan)')
    ap.add_argument('--checkpoint-rule', default='last', choices=['last', 'best'],
                    help='which checkpoint each arm contributes: the end of the matched schedule (last) or the best simulated-validation step')
    args = ap.parse_args()
    root = Path(args.out)
    if args.command == 'plan':
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'plan.json').exists() and not args.arms:
            raise SystemExit(f'{root / "plan.json"} exists; delete it to re-plan (arms already trained would be re-used by name)')
        here = Path(__file__).resolve().parent
        plan = make_plan(args)
        plan['inputs'] = {str(p): sha(Path(p)) for p in [__file__, here / 'drivers.py', here / 'gain_surgery.py', here / 'extract.py',
                                                          here / 'PROTOCOL.md', here / 'PROTOCOL_AMENDMENT_20260907.md',
                                                          Path('src/volpfn/model.py'), Path('src/volpfn/prior.py')]}
        save(root / 'plan.json', plan)
        print(json.dumps({k: v for k, v in plan.items() if k != 'inputs'}, indent=1))
    elif args.command == 'run':
        run_arms(args)
    else:
        summarise(args)


if __name__ == '__main__':
    main()
