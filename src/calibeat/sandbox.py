"""Falsification sandbox for the calibeating claims (PROTOCOL.md).

Honest construction: every data-generating process has an exactly known conditional (VaR, ES) path
(the oracle), the black-box forecasters are causal and never see the truth, the calibeaters see only
the black box's forecasts and past outcomes, the hindsight recalibration is the benchmark the online
procedure is measured against (its gain over the original is the miscalibration MCB), and the
report fits log-log exponents and prints what failure would look like.

usage: python3 -m calibeat.sandbox --scale small|full --out outputs/sandbox_v1 [--procs 4]
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

from .fz import fz0, identification, student_t_var_es, normal_var_es, hindsight_recalibration, hindsight_ratio_recalibration
from .calibeater import RatioCalibeater, BinCalibeater

WARM = 250                                  # calibeaters pass through before this; scoring starts here
GRID = {
    'small': dict(alphas=[0.05, 0.025], Ts=[1000, 4000, 16000], seeds=2),
    'full': dict(alphas=[0.05, 0.025, 0.01], Ts=[1000, 2000, 4000, 8000, 16000, 32000], seeds=5),
}
DGPS = ['garch_t', 'regime_t', 'garch_t_break']
FORECASTERS = ['oracle', 'oracle_gauss', 'oracle_scaled', 'ewma_gauss', 'ewma_t5', 'hist250']
CALIBEATERS = ['ratio_ftl', 'ratio_win1000', 'ratio_win2000', 'ratio_sa', 'bins1', 'bins5', 'bins10']


# ----------------------------------------------------------------------------- data-generating processes
def draw_t(rng, nu, T):
    return rng.standard_t(nu, size=T) / np.sqrt(nu / (nu - 2.0))


def simulate(dgp, T, seed, alpha):
    """Returns y, the true conditional sigma, and the true conditional (VaR, ES) at level alpha."""
    rng = np.random.default_rng(1_000_003 * seed + int(1e6 * alpha) + {'garch_t': 0, 'regime_t': 1, 'garch_t_break': 2}[dgp] * 7919)
    if dgp in ('garch_t', 'garch_t_break'):
        omega, a, b = 0.05, 0.08, 0.90
        nus = np.full(T, 5.0)
        if dgp == 'garch_t_break':
            nus[: T // 2] = 8.0; nus[T // 2:] = 3.0             # tail-shape break halfway
        sig2 = np.empty(T); y = np.empty(T); s2 = omega / (1 - a - b)
        qv = np.empty(T); qe = np.empty(T); cache = {}
        for t in range(T):
            sig2[t] = s2; sig = np.sqrt(s2)
            nu = nus[t]
            if nu not in cache:
                cache[nu] = student_t_var_es(alpha, nu)
            qv[t], qe[t] = sig * cache[nu][0], sig * cache[nu][1]
            y[t] = sig * draw_t(rng, nu, 1)[0]
            s2 = omega + a * y[t] ** 2 + b * s2
        return y, np.sqrt(sig2), qv, qe
    # regime_t: two volatility regimes, Markov switching, t5 innovations
    nu = 5.0; levels = np.array([0.8, 2.5]); mean_dur = np.array([200.0, 50.0])
    q, es = student_t_var_es(alpha, nu)
    state = 0; sig = np.empty(T)
    for t in range(T):
        sig[t] = levels[state]
        if rng.random() < 1.0 / mean_dur[state]:
            state = 1 - state
    y = sig * draw_t(rng, nu, T)
    return y, sig, sig * q, sig * es


# ----------------------------------------------------------------------------- black-box forecasters (causal)
def ewma_sigma(y, lam=0.94):
    """RiskMetrics variance forecast for t made from y_{<t}; starts at the sample variance of the first 50."""
    T = len(y); s2 = np.empty(T); v = float(np.var(y[:50])) if T >= 50 else 1.0
    for t in range(T):
        s2[t] = v; v = lam * v + (1 - lam) * y[t] ** 2
    return np.sqrt(s2)


def forecast(name, y, sigma, qv, qe, alpha):
    if name == 'oracle':
        return qv.copy(), qe.copy()
    if name == 'oracle_gauss':                                 # the true sigma path with the wrong (Gaussian) tail law
        q, es = normal_var_es(alpha); return sigma * q, sigma * es
    if name == 'oracle_scaled':
        return 0.8 * qv, 0.8 * qe
    if name in ('ewma_gauss', 'ewma_t5'):
        s = ewma_sigma(y); q, es = normal_var_es(alpha) if name == 'ewma_gauss' else student_t_var_es(alpha, 5.0)
        return s * q, s * es
    if name == 'hist250':
        from .fz import empirical_var_es
        T = len(y); v = np.empty(T); e = np.empty(T)
        for t in range(T):
            lo = max(0, t - 250); w = y[lo:t] if t >= 30 else y[:30]
            v[t], e[t] = empirical_var_es(w, alpha)
        return v, e
    raise ValueError(name)


def calibeat(name, y, v, e, alpha):
    if name == 'ratio_ftl':
        cb = RatioCalibeater(alpha, warm=WARM, mode='ftl'); vh, eh = cb.run(y, v, e); return vh, eh, None
    if name == 'ratio_sa':
        cb = RatioCalibeater(alpha, warm=WARM, mode='sa', eta=0.002); vh, eh = cb.run(y, v, e); return vh, eh, None
    if name.startswith('ratio_win'):
        cb = RatioCalibeater(alpha, warm=WARM, mode='window', window=int(name[9:])); vh, eh = cb.run(y, v, e); return vh, eh, None
    if name.startswith('bins'):
        cb = BinCalibeater(alpha, K=int(name[4:]), warm=WARM); vh, eh = cb.run(y, v, e); return vh, eh, cb.groups
    raise ValueError(name)


# ----------------------------------------------------------------------------- one cell
def cell(args):
    dgp, fc, alpha, T, seed = args
    t0 = time.time()
    y, sigma, qv, qe = simulate(dgp, T, seed, alpha)
    v, e = forecast(fc, y, sigma, qv, qe, alpha)
    e = np.minimum(e, v - 1e-9)
    sc = slice(WARM, T)
    out = dict(dgp=dgp, forecaster=fc, alpha=alpha, T=T, seed=seed,
               oracle=float(np.mean(fz0(y[sc], qv[sc], qe[sc], alpha))),
               original=float(np.mean(fz0(y[sc], v[sc], e[sc], alpha))),
               original_hit=float(np.mean(y[sc] <= v[sc])),
               original_es_resid=float(np.mean(identification(y[sc], v[sc], e[sc], alpha)[1])),
               hits_scored=int(np.sum(y[sc] <= v[sc])))
    # hindsight benchmarks: MCB with respect to each grouping
    vr, er = hindsight_ratio_recalibration(y[sc], v[sc], alpha)
    out['hindsight_ratio'] = float(np.mean(fz0(y[sc], vr, er, alpha)))
    for name in CALIBEATERS:
        vh, eh, groups = calibeat(name, y, v, e, alpha)
        out[name] = float(np.mean(fz0(y[sc], vh[sc], eh[sc], alpha)))
        out[name + '_hit'] = float(np.mean(y[sc] <= vh[sc]))
        out[name + '_es_resid'] = float(np.mean(identification(y[sc], vh[sc], eh[sc], alpha)[1]))
        if groups is not None:
            vg, eg = hindsight_recalibration(y[sc], v[sc], e[sc], alpha, groups[sc])
            out['hindsight_' + name] = float(np.mean(fz0(y[sc], vg, eg, alpha)))
    out['seconds'] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- runner and report
def fit_power(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); ok = (ys > 0) & np.isfinite(ys)
    if ok.sum() < 3:
        return float('nan'), float('nan')
    X = np.column_stack([np.ones(ok.sum()), np.log(xs[ok])]); coef, *_ = np.linalg.lstsq(X, np.log(ys[ok]), rcond=None)
    res = np.log(ys[ok]) - X @ coef; se = np.sqrt(res @ res / max(1, ok.sum() - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return float(coef[1]), float(se)


def mean_over(rows, key, cond):
    v = [r[key] for r in rows if cond(r) and key in r and r[key] == r[key]]
    return float(np.mean(v)) if v else float('nan')


def report(rows, grid, out):
    L = []; P = L.append
    P('CALIBEATING THE (VaR, ES) PAIR: falsification sandbox (PROTOCOL.md)')
    P(f'{len(rows)} cells; scored from t = {WARM}; FZ0 mean per observation (lower is better); gain = original - calibeated; MCB = original - hindsight')
    decisions = {}
    Tmax = max(grid['Ts'])
    for dgp in DGPS:
        for alpha in grid['alphas']:
            rr = [r for r in rows if r['dgp'] == dgp and r['alpha'] == alpha]
            if not rr:
                continue
            P(''); P(f'=== dgp {dgp}, alpha {alpha}, T = {Tmax} (mean over seeds); columns: score, then gain over the original')
            head = 'forecaster'.ljust(14) + f'{"oracle":>9}{"original":>10}{"hit":>7}' + ''.join(f'{c:>14}' for c in CALIBEATERS) + f'{"MCB_ratio":>11}{"MCB_bins5":>11}'
            P(head)
            for fc in FORECASTERS:
                c = lambda r, fc=fc: r['forecaster'] == fc and r['T'] == Tmax
                o = mean_over(rr, 'original', c); line = fc.ljust(14) + f'{mean_over(rr, "oracle", c):9.4f}{o:10.4f}{mean_over(rr, "original_hit", c):7.3f}'
                for cb in CALIBEATERS:
                    line += f'{o - mean_over(rr, cb, c):+14.4f}'
                line += f'{o - mean_over(rr, "hindsight_ratio", c):+11.4f}{o - mean_over(rr, "hindsight_bins5", c):+11.4f}'
                P(line)
            # scaling: excess over the oracle (no-harm) and gain/MCB by T
            P(''); P('by T: oracle+ratio_ftl excess over oracle | ewma_gauss: gain(ratio_ftl)/MCB_ratio, gain(bins5)/MCB_bins5 | regret_T = T*(ratio_ftl - hindsight_ratio) for ewma_gauss')
            ex, gr, gb, rg = [], [], [], []
            for T in grid['Ts']:
                co = lambda r, T=T: r['forecaster'] == 'oracle' and r['T'] == T
                ce = lambda r, T=T: r['forecaster'] == 'ewma_gauss' and r['T'] == T
                excess = mean_over(rr, 'ratio_ftl', co) - mean_over(rr, 'oracle', co)
                o = mean_over(rr, 'original', ce)
                mcb_r = o - mean_over(rr, 'hindsight_ratio', ce); mcb_b = o - mean_over(rr, 'hindsight_bins5', ce)
                g_r = (o - mean_over(rr, 'ratio_ftl', ce)) / mcb_r if mcb_r > 1e-4 else float('nan')
                g_b = (o - mean_over(rr, 'bins5', ce)) / mcb_b if mcb_b > 1e-4 else float('nan')
                regret = T * (mean_over(rr, 'ratio_ftl', ce) - mean_over(rr, 'hindsight_ratio', ce))
                ex.append(excess); gr.append(g_r); gb.append(g_b); rg.append(regret)
                P(f'  T={T:6d}  excess {excess:+.5f}   gain/MCB ratio {g_r:6.3f}  bins5 {g_b:6.3f}   regret_T {regret:8.3f}')
            b_ex, se_ex = fit_power(grid['Ts'], ex); b_rg, se_rg = fit_power(grid['Ts'], rg)
            P(f'  fits: oracle excess ~ T^{b_ex:+.2f}±{se_ex:.2f};  regret_T ~ T^{b_rg:+.2f}±{se_rg:.2f}')
            key = f'{dgp}|{alpha}'
            mcb_ref = mean_over(rr, 'original', lambda r: r['forecaster'] == 'ewma_gauss' and r['T'] == Tmax) - mean_over(rr, 'hindsight_ratio', lambda r: r['forecaster'] == 'ewma_gauss' and r['T'] == Tmax)
            decisions[key] = {
                'D1_no_harm': {'pass': bool(ex[-1] <= 0.1 * mcb_ref and (b_ex <= -0.4 or ex[-1] <= 0)), 'excess_at_Tmax': ex[-1], 'exponent': b_ex, 'reference_MCB': mcb_ref},
                'D2_gain_captures_MCB': {'pass': bool(gr[-1] >= 0.8), 'gain_over_MCB_ratio_at_Tmax': gr[-1], 'bins5': gb[-1]},
                'D3_sublinear_regret': {'pass': bool(b_rg < 0.8), 'exponent': b_rg},
            }
            if dgp == 'garch_t_break':
                ce = lambda r: r['forecaster'] == 'ewma_gauss' and r['T'] == Tmax
                o = mean_over(rr, 'original', ce); mcb = o - mean_over(rr, 'hindsight_ratio', ce)
                late = slice(None)
                best_adaptive = max(o - mean_over(rr, k, ce) for k in ('ratio_win1000', 'ratio_win2000', 'ratio_sa'))
                ftl = o - mean_over(rr, 'ratio_ftl', ce); orc = mean_over(rr, 'oracle', ce)
                # under a break the hindsight constant pair is itself wrong in each half, so the adaptive variants are judged
                # against the oracle gap: an adaptive wrapper should close more of (original - oracle) than FTL does
                decisions[key]['D3b_adaptive_under_break'] = {'pass': bool(best_adaptive > ftl), 'gain_best_adaptive': best_adaptive, 'gain_ftl': ftl,
                                                              'original_minus_oracle': o - orc}
    P(''); P('DECISIONS (PROTOCOL.md)')
    for key, d in decisions.items():
        for k, v in d.items():
            P(f'  {key:22s} {k:28s} {"PASS" if v["pass"] else "FAIL"}  ' + ', '.join(f'{kk}={vv:+.4f}' if isinstance(vv, float) else f'{kk}={vv}' for kk, vv in v.items() if kk != 'pass'))
    P(''); P('What failure looks like: D1 failing = the wrapper harms an already-calibrated forecaster by more than a tenth of what it gains on a mis-specified one;')
    P('D2 failing = the online procedure captures well under 80% of the hindsight miscalibration at the largest T (data-starved tail);')
    P('D3 failing = regret grows nearly linearly, i.e. no online guarantee is available for this procedure; D3b failing = FTL cannot follow a tail-shape break and the SA variant does not fix it.')
    rep = '\n'.join(L); print(rep)
    Path(out, 'REPORT.txt').write_text(rep + '\n'); Path(out, 'decisions.json').write_text(json.dumps(decisions, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scale', choices=list(GRID), default='small')
    ap.add_argument('--out', default='outputs/sandbox_v1')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dgps', default=None); ap.add_argument('--forecasters', default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = GRID[a.scale]
    dgps = a.dgps.split(',') if a.dgps else DGPS; fcs = a.forecasters.split(',') if a.forecasters else FORECASTERS
    here = Path(__file__).resolve().parent
    Path(out, 'protocol.json').write_text(json.dumps({
        'grid': grid, 'dgps': dgps, 'forecasters': fcs, 'calibeaters': CALIBEATERS, 'warm': WARM,
        'inputs': {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in ('fz.py', 'calibeater.py', 'sandbox.py')},
        'protocol_md': hashlib.sha256((here.parent / 'PROTOCOL.md').read_bytes()).hexdigest() if (here.parent / 'PROTOCOL.md').exists() else None},
        indent=1) + '\n')
    done = {}
    cells_path = out / 'cells.jsonl'
    if cells_path.exists():
        for l in cells_path.read_text().splitlines():
            r = json.loads(l); done[(r['dgp'], r['forecaster'], r['alpha'], r['T'], r['seed'])] = r
    jobs = [(d, f, al, T, s) for d in dgps for f in fcs for al in grid['alphas'] for T in grid['Ts'] for s in range(grid['seeds'])
            if (d, f, al, T, s) not in done]
    jobs.sort(key=lambda j: -j[3])
    print(f'{len(jobs)} cells to run ({len(done)} cached)', flush=True)
    t0 = time.time()

    def record(r):
        done[(r['dgp'], r['forecaster'], r['alpha'], r['T'], r['seed'])] = r
        with open(cells_path, 'a') as f:
            f.write(json.dumps(r) + '\n')

    if a.procs > 1 and jobs:
        from multiprocessing import Pool
        with Pool(a.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(cell, jobs)):
                record(r)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    print(f'  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)
    else:
        for i, j in enumerate(jobs):
            record(cell(j))
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f'  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)
    report(list(done.values()), grid, out)


if __name__ == '__main__':
    main()
