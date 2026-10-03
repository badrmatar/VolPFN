"""Sandbox study of the adaptive guard (Theorem E). Pre-registered rules, written before the run:

  E1 no harm:   on the stationary DGPs with the oracle and oracle_gauss boxes, adapt minus the best
                single expert in hindsight <= B(alpha) = 0.0035 / 0.006 / 0.013.
  E2 capture:   for the mis-specified boxes (oracle_gauss, ewma_gauss, ewma_t5), adapt captures
                >= 80% of the best expert's gain over the original (all DGPs, all levels).
  E3 drift:     on the break DGP, adapt minus the piecewise best (best expert per half) <= B(alpha),
                and for the mis-specified boxes adapt beats the full-history wrapper at 0.05 and 0.025
                (prediction: pass; open at 0.01, where window noise exceeds the drift).
python3 -m calibeat.adapt_study run --out outputs/adapt_v1 --seeds 1 2 3 --procs 4
python3 -m calibeat.adapt_study report --out outputs/adapt_v1
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from . import sandbox as S
from . import adaptive as A
from .fz import fz0

DGPS = ['garch_t', 'regime_t', 'garch_t_break']; FORECASTERS = ['oracle', 'oracle_gauss', 'ewma_gauss', 'ewma_t5']
ALPHAS = [0.05, 0.025, 0.01]; T = 32000; WARM = 500
BUDGET = {0.05: 0.0035, 0.025: 0.006, 0.01: 0.013}


def cell(args):
    dgp, fc, alpha, seed = args; t0 = time.time()
    y, sigma, qv, qe = S.simulate(dgp, T, seed, alpha); v, e = S.forecast(fc, y, sigma, qv, qe, alpha); e = np.minimum(e, v - 1e-9)
    paths = A.expert_paths(y, v, e, alpha); names = list(paths); L = np.stack([fz0(y, *paths[k], alpha) for k in names], axis=1)
    orc = fz0(y, qv, qe, alpha); choice = A.ftl_window_select(L, start=WARM, W=2000, margin=0.5 * BUDGET[alpha])
    sel = L[np.arange(len(y)), choice]; sc = slice(WARM, None); halves = [slice(WARM, T // 2), slice(T // 2, None)]
    out = dict(dgp=dgp, forecaster=fc, alpha=alpha, seed=seed, oracle=float(orc[sc].mean()),
               experts={k: float(L[sc, i].mean()) for i, k in enumerate(names)}, adapt=float(sel[sc].mean()),
               piecewise_best=float(sum(L[h].mean(axis=0).min() * (h.stop or T) - (h.start) for h in halves) / (T - WARM)) if dgp == 'garch_t_break' else None,
               chosen={k: float(np.mean(choice[sc] == i)) for i, k in enumerate(names)}, seconds=time.time() - t0)
    if dgp == 'garch_t_break':
        out['piecewise_best'] = float(sum(L[h].mean(axis=0).min() * ((h.stop or T) - h.start) for h in halves) / (T - WARM))
    return out


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); path = out / 'cells.jsonl'
    done = set()
    if path.exists():
        done = {(r['dgp'], r['forecaster'], r['alpha'], r['seed']) for r in map(json.loads, open(path))}
    jobs = [(d, f, al, s) for d in DGPS for f in FORECASTERS for al in ALPHAS for s in a.seeds if (d, f, al, s) not in done]
    print(f'{len(jobs)} cells to run ({len(done)} done)', flush=True)
    import multiprocessing as mp
    with mp.Pool(a.procs) as pool, open(path, 'a') as fh:
        for r in pool.imap_unordered(cell, jobs):
            fh.write(json.dumps(r) + '\n'); fh.flush(); print(f"{r['dgp']} {r['forecaster']} {r['alpha']} s{r['seed']} {r['seconds']:.0f}s", flush=True)


def report(a):
    rows = [json.loads(l) for l in open(Path(a.out) / 'cells.jsonl')]; L = []; P = L.append; dec = []
    names = list(rows[0]['experts'])
    P('ADAPTIVE GUARD (FTL over windows, W = 2000, margin B/2): excess over the oracle, mean over seeds')
    P('%-14s %-12s %-6s | %8s ' % ('dgp', 'box', 'alpha', 'original') + ' '.join('%8s' % k for k in names[1:]) + ' | %8s %8s %8s | %s' % ('adapt', 'best', 'regret', 'chosen (share)'))
    for d in DGPS:
        for f in FORECASTERS:
            for al in ALPHAS:
                sub = [r for r in rows if r['dgp'] == d and r['forecaster'] == f and r['alpha'] == al]
                if not sub:
                    continue
                ex = {k: np.mean([r['experts'][k] - r['oracle'] for r in sub]) for k in names}; ad = np.mean([r['adapt'] - r['oracle'] for r in sub])
                best_k = min(names, key=lambda k: ex[k]); best = ex[best_k]; regret = ad - best
                ch = {k: np.mean([r['chosen'][k] for r in sub]) for k in names}; top = sorted(ch, key=ch.get, reverse=True)[:3]
                P('%-14s %-12s %-6s | %+8.4f ' % (d, f, al, ex['original']) + ' '.join('%+8.4f' % ex[k] for k in names[1:]) + ' | %+8.4f %+8.4f %+8.4f | %s' % (ad, best, regret, ', '.join(f'{k} {ch[k]:.2f}' for k in top)))
                if d != 'garch_t_break' and f in ('oracle', 'oracle_gauss'):
                    ok = regret <= BUDGET[al]; dec.append(('E1', d, f, al, ok, regret));
                if f in ('oracle_gauss', 'ewma_gauss', 'ewma_t5'):
                    gain_best = ex['original'] - best; gain_ad = ex['original'] - ad
                    cap = gain_ad / gain_best if gain_best > 1e-9 else float('nan'); ok = cap == cap and cap >= 0.8; dec.append(('E2', d, f, al, ok, cap))
                if d == 'garch_t_break':
                    pw = np.mean([r['piecewise_best'] - r['oracle'] for r in sub]); ok = (ad - pw) <= BUDGET[al]; dec.append(('E3a', d, f, al, ok, ad - pw))
                    if f != 'oracle':
                        ok2 = ad < ex['ftl']; dec.append(('E3b', d, f, al, ok2, ad - ex['ftl']))
    P(''); P('DECISIONS'); n_pass = 0
    for rule, d, f, al, ok, val in dec:
        binding = not (rule == 'E3b' and al == 0.01); n_pass += ok and binding
        P(f"{rule:<4} {d:<14} {f:<12} alpha={al:<6} {'PASS' if ok else 'FAIL'}  value={val:+.4f}{'' if binding else ' (informational at this level)'}")
    n_bind = sum(1 for r in dec if not (r[0] == 'E3b' and r[3] == 0.01))
    P(f'SUMMARY: {n_pass}/{n_bind} binding decisions pass'); txt = '\n'.join(L); (Path(a.out) / 'REPORT.txt').write_text(txt + '\n'); print(txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['run', 'report']); ap.add_argument('--out', default='outputs/adapt_v1')
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3]); ap.add_argument('--procs', type=int, default=4); a = ap.parse_args()
    (run if a.command == 'run' else report)(a)


if __name__ == '__main__':
    main()
