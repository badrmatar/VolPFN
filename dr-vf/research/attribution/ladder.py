"""The adaptivity ladder (research/attribution/LADDER_RULES.md): learned and compressed filters
against the exact Bayes floor on the Phase-1 lattice jump paths.

Reuses phase1.py's DGP, seeding, floor, FLH-FTL and hindsight fixed-gain rungs unchanged, and adds
the network (volpfn_base_best.pt), the compressed rules with their real-data coefficients, and the
same rules refitted by Gaussian likelihood on the DGP. Scores every rung with phase1.xregret from
t = 100. Resumable: each finished cell is appended to <out>/cells.jsonl and skipped on re-run.

usage (from dr-vf, PYTHONPATH=.):
  python3 research/attribution/ladder.py fit    [--procs 4]              # hindsight refits, cached
  python3 research/attribution/ladder.py run    [--rungs all|no-network] [--procs 6] [--scale small|full]
  python3 research/attribution/ladder.py report
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import phase1 as P1
import falsify_I13 as F
from research.prior_adaptation.experiment import save, sha
import research.attribution.drivers as D
from volpfn_prior import initial_scale

SCORE_FROM = 100
RULES = {'core5': 'asymmetric_gain', 'drift7': 'drift_only', 'rule9': 'state_dep_gain'}
ALL_RUNGS = ['bayes', 'fixed_best', 'ewma_best', 'garch_best', 'flh_ftl',
             'core5_real', 'drift7_real', 'rule9_real', 'core5_fit', 'drift7_fit', 'rule9_fit', 'network']
GRID = {
    'small': dict(deltas=[2.0], Vs=[8.0], Ts_sparse=[512, 1024, 2048, 4096], Ts_dense=[2048], seeds=2),
    'full': dict(deltas=[2.0, 1.5], Vs=[8.0, 32.0], Ts_sparse=[512, 1024, 2048, 4096, 8192, 16384, 32768],
                 Ts_dense=[2048, 8192, 32768], seeds=5),
}
FIT_T, FIT_SEEDS = 4096, 3


# ----------------------------------------------------------------------------- paths (phase1's seeding, unchanged)
def make_cell(regime, T, V, seed, delta):
    rng = np.random.default_rng(10_000 + 7919 * seed + 13 * int(V) + T + (0 if regime == 'sparse' else 1_000_000))
    if regime == 'sparse':
        K = V / delta
    else:
        K = (T * V * V) ** (1.0 / 3.0); delta = V / K       # the caller's delta is ignored in the dense regime
    theta, lam, K_real, V_real = P1.make_jump_path(T, K, delta, rng)
    J = max(int(4.0 * math.sqrt(max(K, 1.0))) + 8, int(np.max(np.abs(np.round(theta / delta)))) + 2)
    r = np.exp(theta / 2.0) * P1.draw_innov(T, rng, 'gaussian')
    return dict(theta=theta, lam=lam, delta=delta, K=K, K_real=K_real, V_real=V_real, J=J, r=r, r2=r * r,
                hstar=np.exp(theta), K_window=int(np.sum(np.abs(np.diff(theta[SCORE_FROM - 1:])) > 0)))


def score_rung(out, name, hhat, cell):
    h = np.maximum(np.asarray(hhat, float), F.H_FLOOR)
    out[name] = P1.xregret(h[SCORE_FROM:], cell['hstar'][SCORE_FROM:])
    out[name + '_realised'] = F.regret(h[SCORE_FROM:], cell['r2'][SCORE_FROM:], cell['hstar'][SCORE_FROM:])


# ----------------------------------------------------------------------------- the compressed rules on a synthetic path
def load_real_coefficients(folds_path):
    """Fold-averaged likelihood coefficients and normalisation of the driver-swap study."""
    folds = json.loads(Path(folds_path).read_text())
    coef, norm = {}, {}
    for short, spec in RULES.items():
        names = D.param_names(spec)
        coef[short] = [float(np.mean([f[spec]['coefficients_likelihood'][n] for f in folds.values()])) for n in names]
    keys = next(iter(folds.values()))['state_dep_gain']['normalisation'].keys()
    norm = {k: float(np.mean([f['state_dep_gain']['normalisation'][k] for f in folds.values()])) for k in keys}
    return coef, norm


def packed_for(paths, norm):
    """drivers.stack over synthetic paths; the recursion starts at the true initial level, as every
    other rung does, and the fitting/scoring mask starts at SCORE_FROM."""
    prepared = []
    for r in paths:
        market = {'r': r, 'scale': float(initial_scale(r[:SCORE_FROM])), 'date': None}
        keep = np.arange(len(r)) >= SCORE_FROM
        prepared.append(D.prepare(market, norm, keep))
    P = D.stack(prepared)
    P['lh0'][:] = 0.0                    # theta0 = 0: log h_1 = 0, the same warm start as bayes/fixed/FLH
    return P


def rule_paths(P, beta, spec, norm):
    return D.run_batch(P, beta, spec, norm)


def gauss_nll(beta, P, spec, norm):
    H = D.run_batch(P, beta, spec, norm); k = P['keep']
    if not np.all(np.isfinite(H[k])) or np.any(H[k] <= 0):
        return 1e6
    return -float(np.mean(P1.loglik_r2(P['R2'][k], H[k], 'gaussian')))


def fit_one(args):
    """Hindsight refit of one rule form on FIT_SEEDS paths of one (regime, delta, V) at T = FIT_T."""
    short, regime, delta, V, start, norm, maxfev = args
    spec = RULES[short]
    paths = [make_cell(regime, FIT_T, V, 1000 + s, delta)['r'] for s in range(FIT_SEEDS)]
    P = packed_for(paths, norm)
    t0 = time.time()
    # Two starts, keep the better: the real-data coefficients, and a log-space EWMA (gain 0.05 on the
    # score, everything else zero), so a poor real-data start cannot hide what the class can do.
    ewma_like = np.zeros(len(start)); ewma_like[2] = 0.05
    best = None
    for x0 in (np.asarray(start, float), ewma_like):
        res = minimize(gauss_nll, x0, args=(P, spec, norm), method='Powell', options={'maxfev': maxfev, 'xtol': 1e-3, 'ftol': 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    return dict(key=f'{short}|{regime}|{delta if regime == "sparse" else "dense"}|{V}', short=short, regime=regime, delta=delta, V=V,
                beta=[float(x) for x in best.x], nll=float(best.fun), nll_start=float(gauss_nll(np.asarray(start, float), P, spec, norm)),
                nll_ewma_start=float(gauss_nll(ewma_like, P, spec, norm)),
                converged=bool(best.success), evaluations=int(best.nfev), seconds=time.time() - t0)


# ----------------------------------------------------------------------------- the network rung
_NET = {}

def network_paths(r, ckpt):
    import torch, volpfn as V
    if ckpt not in _NET:
        torch.set_num_threads(1)         # one thread per worker process; the pool supplies the parallelism
        _NET[ckpt] = V.load_model(ckpt, 'cpu')
    out = V.predict_series(_NET[ckpt], r, L=1024, stride=512, min_history=SCORE_FROM, levels=())
    h = np.asarray(out['h'], float)
    h[~np.isfinite(h)] = 1.0             # only t < SCORE_FROM, never scored
    return h


# ----------------------------------------------------------------------------- one cell
def cell(args):
    regime, T, V, seed, delta, rungs, coef, norm, fits, ckpt = args
    c = make_cell(regime, T, V, seed, delta); r2, hstar = c['r2'], c['hstar']
    out = dict(regime=regime, T=T, V=V, seed=seed, delta=c['delta'], K_target=c['K'], K_real=c['K_real'],
               K_window=c['K_window'], V_real=c['V_real'], segment=T * c['delta'] / V if regime == 'sparse' else T / max(c['K'], 1))
    t0 = time.time()
    if 'bayes' in rungs:
        score_rung(out, 'bayes', P1.bayes_jump_filter(r2, c['lam'], c['delta'], 0.0, 'gaussian', c['J']), c)
    if 'fixed_best' in rungs:
        Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
        he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
        score_rung(out, 'ewma_best', he, c); score_rung(out, 'garch_best', hg, c)
        score_rung(out, 'fixed_best', he if Re <= Rg else hg, c)
    if 'flh_ftl' in rungs:
        score_rung(out, 'flh_ftl', P1.flh_ftl(r2, hstar, 'ftl'), c)
    need_rules = [g for g in rungs if g.endswith('_real') or g.endswith('_fit')]
    if need_rules:
        P = packed_for([c['r']], norm)
        for g in need_rules:
            short, kind = g.split('_')
            if kind == 'real':
                beta = coef[short]
            else:
                key = f'{short}|{regime}|{c["delta"] if regime == "sparse" else "dense"}|{V}'
                beta = fits[key]['beta']
            H = rule_paths(P, beta, RULES[short], norm)[0, :T]
            score_rung(out, g, H, c)
    if 'network' in rungs:
        score_rung(out, 'network', network_paths(c['r'], ckpt), c)
    out['seconds'] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- drivers
def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def grid_jobs(g):
    jobs = [('sparse', T, V, s, d) for d in g['deltas'] for V in g['Vs'] for T in g['Ts_sparse'] for s in range(g['seeds'])]
    jobs += [('dense', T, V, s, None) for V in g['Vs'] for T in g['Ts_dense'] for s in range(g['seeds'])]
    return jobs


def do_fit(args, coef, norm):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True); cache = root / 'fits.json'
    fits = json.loads(cache.read_text()) if cache.exists() else {}
    g = GRID[args.scale]
    todo = []
    for short in RULES:
        for d in g['deltas']:
            for V in g['Vs']:
                if f'{short}|sparse|{d}|{V}' not in fits:
                    todo.append((short, 'sparse', d, V, coef[short], norm, args.maxfev))
        for V in g['Vs']:
            if f'{short}|dense|dense|{V}' not in fits:
                todo.append((short, 'dense', None, V, coef[short], norm, args.maxfev))
    print(f'{len(todo)} refits to do ({len(fits)} cached)', flush=True)
    if not todo:
        return fits
    if args.procs > 1:
        from multiprocessing import Pool
        with Pool(args.procs) as pool:
            for res in pool.imap_unordered(fit_one, todo):
                fits[res['key']] = res; save(cache, fits)
                print(f"  {res['key']:28s} nll {res['nll_start']:.5f} -> {res['nll']:.5f}  {res['evaluations']} evals  {res['seconds']:.0f}s", flush=True)
    else:
        for job in todo:
            res = fit_one(job); fits[res['key']] = res; save(cache, fits)
            print(f"  {res['key']:28s} nll {res['nll_start']:.5f} -> {res['nll']:.5f}  {res['evaluations']} evals  {res['seconds']:.0f}s", flush=True)
    return fits


def do_run(args, coef, norm):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    rungs = ALL_RUNGS if args.rungs == 'all' else ([g for g in ALL_RUNGS if g != 'network'] if args.rungs == 'no-network'
                                                    else [g.strip() for g in args.rungs.split(',')])
    fits = json.loads((root / 'fits.json').read_text()) if (root / 'fits.json').exists() else {}
    if any(g.endswith('_fit') for g in rungs) and not fits:
        raise SystemExit('no fits.json; run `fit` first or use --rungs without the *_fit rungs')
    done = {}
    for row in load_jsonl(root / 'cells.jsonl'):
        done[(row['regime'], row['T'], row['V'], row['seed'], row.get('delta_key'))] = row
    g = GRID[args.scale]
    jobs = []
    for regime, T, V, s, d in grid_jobs(g):
        key = (regime, T, V, s, d if regime == 'sparse' else None)
        row = done.get(key)
        missing = [r for r in rungs if row is None or r not in row]
        if any(r in ('ewma_best', 'garch_best') for r in missing) and 'fixed_best' not in missing:
            missing.append('fixed_best')
        if missing:
            jobs.append((regime, T, V, s, d if d is not None else 2.0, missing, coef, norm, fits, args.network_ckpt))
    jobs.sort(key=lambda j: -j[1])
    print(f'{len(jobs)} cells with missing rungs (of {len(grid_jobs(g))}); rungs {rungs}', flush=True)
    if not jobs:
        return
    here = Path(__file__).resolve().parent
    save(root / 'protocol.json', {
        'scope': 'Adaptivity ladder on synthetic lattice jump paths (LADDER_RULES.md). No market data.',
        'inputs': {str(p): sha(Path(p)) for p in [__file__, here / 'LADDER_RULES.md', here / 'drivers.py', Path('phase1.py'),
                                                   Path('falsify_I13.py'), Path(args.folds)] + ([Path(args.network_ckpt)] if 'network' in rungs and Path(args.network_ckpt).exists() else [])},
        'grid': g, 'score_from': SCORE_FROM, 'fit': {'T': FIT_T, 'seeds': FIT_SEEDS, 'maxfev': args.maxfev},
        'real_coefficients': coef, 'normalisation': norm})

    def record(out, job):
        regime, T, V, s, d = job[:5]
        key = (regime, T, V, s, d if regime == 'sparse' else None)
        row = done.get(key, {}); row.update(out); row['delta_key'] = d if regime == 'sparse' else None
        done[key] = row
        with open(root / 'cells.jsonl', 'w') as f:
            for r in done.values():
                f.write(json.dumps(r) + '\n')

    t0 = time.time()
    if args.procs > 1:
        from multiprocessing import Pool
        with Pool(args.procs) as pool:
            for i, (out, job) in enumerate(zip(pool.imap(cell, jobs), jobs)):
                record(out, job)
                print(f"  {i + 1}/{len(jobs)} {job[0]:6s} T={job[1]:6d} V={job[2]:g} d={job[4]:g} s={job[3]}  {out['seconds']:5.0f}s  ({time.time() - t0:.0f}s total)", flush=True)
    else:
        for i, job in enumerate(jobs):
            out = cell(job); record(out, job)
            print(f"  {i + 1}/{len(jobs)} {job[0]:6s} T={job[1]:6d} V={job[2]:g} d={job[4]:g} s={job[3]}  {out['seconds']:5.0f}s  ({time.time() - t0:.0f}s total)", flush=True)


def per_jump(rows, rung):
    return [r[rung] / max(r['K_window'], 1) for r in rows if rung in r and r[rung] == r[rung]]


def do_report(args):
    root = Path(args.out); rows = load_jsonl(root / 'cells.jsonl')
    if not rows:
        raise SystemExit('no cells')
    rungs = [g for g in ALL_RUNGS if any(g in r for r in rows)]
    L = []; P = L.append
    P('THE ADAPTIVITY LADDER: expected QLIKE regret against the true path, scored from t = 100; per-jump = regret / jumps in the scored window')
    P(f'{len(rows)} cells; rungs: {", ".join(rungs)}')
    decisions = {}
    for regime in ('sparse', 'dense'):
        rr = [r for r in rows if r['regime'] == regime]
        if not rr:
            continue
        P(''); P(f'=== regime {regime}')
        cells = sorted({(r.get('delta_key'), r['V'], r['T']) for r in rr})
        P('per-jump cost, mean over seeds (ratio to the floor in brackets)')
        P('cell'.ljust(24) + ''.join(f'{g:>16}' for g in rungs))
        curves = {}
        for (d, V, T) in cells:
            sel = [r for r in rr if r.get('delta_key') == d and r['V'] == V and r['T'] == T]
            seg = np.mean([r['segment'] for r in sel])
            line = f'd={d} V={V:g} T={T:6d} L={seg:6.0f}'.ljust(24)
            floor = np.mean(per_jump(sel, 'bayes')) if per_jump(sel, 'bayes') else float('nan')
            for g in rungs:
                v = per_jump(sel, g)
                if v:
                    m = float(np.mean(v)); curves.setdefault(g, []).append((d, V, T, seg, m, m / floor if floor == floor else float('nan')))
                    line += f'{m:9.2f}[{m / floor:4.2f}]' if floor == floor else f'{m:16.2f}'
                else:
                    line += ' ' * 16
            P(line)
        if regime == 'sparse':
            P(''); P('per-jump cost vs T (all sparse cells): power-law exponent and log-linear fit')
            fits = {}
            for g in rungs:
                pts = curves.get(g, [])
                xs = np.array([p[2] for p in pts], float); ys = np.array([p[4] for p in pts], float)
                ok = (ys > 0) & np.isfinite(ys)
                if ok.sum() >= 3:
                    b, se, r2p = P1.fit_power(xs[ok], ys[ok]); c1, c2, r2l = P1.fit_loglinear(xs[ok], ys[ok])
                    fits[g] = dict(power_exponent=float(b), power_se=float(se), power_r2=float(r2p), loglin_slope=float(c2), loglin_r2=float(r2l))
                    P(f'  {g:12s} T^{b:+.2f}±{se:.2f} (R²={r2p:.3f}) | {c1:.1f} + {c2:.2f} log T (R²={r2l:.3f})')
            # rules
            def sub(g, cond):
                return [p for p in curves.get(g, []) if cond(p)]
            inside = lambda p: p[3] <= 1024; outside = lambda p: p[3] > 1024
            if 'network' in curves and 'bayes' in curves:
                w = sub('network', inside)
                if len(w) >= 3:
                    xs = np.array([p[2] for p in w]); ratios = np.array([p[5] for p in w])
                    b, se, _ = P1.fit_power(xs, ratios)          # exponent of the ratio to the floor, not of the cost itself
                    decisions['L1_floor_tracking_within_window'] = {'pass': bool(ratios.max() <= 1.5 and b <= 0.10), 'max_ratio': float(ratios.max()),
                                                                     'ratio_exponent': float(b), 'cells': len(w)}
                o = sub('network', outside)
                if len(o) >= 3:
                    xs = np.array([p[2] for p in o]); rs = np.array([p[5] for p in o])
                    b, se, _ = P1.fit_power(xs, rs)
                    decisions['L2_window_effect'] = {'pass': bool(b > 0.10), 'ratio_exponent': float(b), 'cells': len(o)}
                if 'flh_ftl' in curves:
                    pairs = [(p[4], q[4]) for p in sub('network', inside) for q in curves['flh_ftl'] if q[:3] == p[:3]]
                    if pairs:
                        decisions['L4_network_below_flh_within_window'] = {'pass': bool(all(a < b for a, b in pairs)),
                                                                          'cells': len(pairs), 'worst_ratio_network_over_flh': float(max(a / b for a, b in pairs))}
            for g in ('rule9_fit', 'drift7_fit', 'core5_fit'):
                if g in fits:
                    f = fits[g]
                    decisions[f'L3_{g}_polynomial'] = {'pass': bool(f['power_exponent'] >= 0.25 and f['power_r2'] >= f['loglin_r2']), **f}
        else:
            if 'network' in curves:
                ratios = [p[5] for p in curves['network'] if p[5] == p[5]]
                decisions['L5_dense_near_floor'] = {'pass': bool(ratios and max(ratios) <= 1.3), 'max_ratio': float(max(ratios)) if ratios else None,
                                                    'cells': len(ratios)}
            P(''); P('ratios to the floor, mean over dense cells: ' + ', '.join(f'{g}: {np.nanmean([p[5] for p in curves[g]]):.2f}' for g in rungs if g in curves and g != 'bayes'))
    P(''); P('RULES (LADDER_RULES.md)')
    for k, v in decisions.items():
        P(f'  {k:42s} {"PASS" if v.get("pass") else "FAIL"}  {json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "pass"})}')
    rep = '\n'.join(L); print(rep)
    (root / 'LADDER_REPORT.txt').write_text(rep + '\n'); save(root / 'decisions.json', decisions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['fit', 'run', 'report'])
    ap.add_argument('--out', default='research/outputs/ladder_v1')
    ap.add_argument('--folds', default='research/outputs/attribution_driver_swap_v1/folds.json')
    ap.add_argument('--network-ckpt', default='out_volpfn/volpfn_base_best.pt')
    ap.add_argument('--scale', choices=list(GRID), default='full')
    ap.add_argument('--rungs', default='all', help='all | no-network | comma list')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--maxfev', type=int, default=800)
    args = ap.parse_args()
    if args.command == 'report':
        do_report(args); return
    coef, norm = load_real_coefficients(args.folds)
    if args.command == 'fit':
        do_fit(args, coef, norm)
    else:
        do_run(args, coef, norm)


if __name__ == '__main__':
    main()
