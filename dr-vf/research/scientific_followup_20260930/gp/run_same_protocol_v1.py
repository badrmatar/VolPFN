"""Explicit CPU replay recipe for all 28 markets and both primary GP seeds.

No default output is allowed. This is a separately versioned method path.
The runner must start from burn-in, keeps every prescribed phase/target, and
records failures. It does not resume old likelihood states or combine corrected
cells with old cells. The default roster is the full 28-market, two-seed roster.
This runner has NOT been executed to completion in the follow-up.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.gpvol_full import run_frontier_gpvol as original_runner
from research.gpvol_full.log_moment_backend import make_log_moment_wrapper
from research.scientific_followup_20260930.gp import likelihood_v1 as repair


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value


def bind_moments(module, state):
    if 'log_h' not in state.out:
        state.out['log_h'] = np.full(len(state.x), np.nan)
    def sink(log_h, metadata):
        state.out['log_h'][state.t] = log_h
    module.predictive_mixture = make_log_moment_wrapper(module.predictive_mixture, sink)


def full_cell(market, seed, bundle, rows, output):
    settings = original_runner.settings_from(SimpleNamespace(seed=seed, n_particles=200,
        burn_in=100, lam=.95, standardise_window=1000, code_indexing=False, chunk=32))
    segments = original_runner.prepare_segments(market, rows[market], bundle['market_paths'][market], settings)
    cell = output / 'cells' / 'gpvol_likelihood_v1' / str(seed) / market
    cell.mkdir(parents=True, exist_ok=False)
    all_rows, diagnostics, phase_receipts = [], {}, []
    for seg in segments:
        module = repair.load_filter_module()
        state = module.RAPCF(seg['x'], n_particles=200, burn_in=100, lam=.95,
            seed=seg['seg_seed'], code_indexing=False, chunk=32)
        bind_moments(module, state)
        start = time.perf_counter()
        failure = None
        try:
            state.run()
        except Exception as error:
            failure = repr(error)
        elapsed = time.perf_counter() - start
        state.save(cell / (seg['phase'] + '_state.npz'))
        for key, value in state.out.items():
            diagnostics[seg['phase']+'__'+key] = value
        phase_receipts.append(dict(phase=seg['phase'], state_t=state.t, chain_length=len(state.x),
            done=state.done, exception=failure, seconds=elapsed, seed=seg['seg_seed'],
            standardization=dict(mean=seg['mu_hat'], sd=seg['sd_hat'])))
        for origin in seg['prow']:
            t = origin['index']-seg['start']
            logvar = float(state.out['log_h'][t] + 2 * np.log(seg['sd_hat']))
            logmean2 = 2 * np.log(abs(seg['mu_hat'])) if seg['mu_hat'] else -np.inf
            logm2 = float(np.logaddexp(logvar, logmean2))
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                var = seg['sd_hat']**2 * state.out['h'][t]
                m2 = var + seg['mu_hat']**2
            logscore = float(state.out['logdens'][t] - np.log(seg['sd_hat']))
            valid = state.done and np.isfinite(logvar) and np.isfinite(logm2) and np.isfinite(logscore)
            all_rows.append(dict(origin, model='gpvol_likelihood_v1', seed=seed, mean=seg['mu_hat'],
                variance=float(var) if np.isfinite(var) and var>0 else None,
                second_moment=float(m2) if np.isfinite(m2) and m2>0 else None,
                log_variance=logvar, log_second_moment=logm2, log_score=logscore,
                substituted=False, floor_applied=False, forecast_failed=not bool(valid),
                moment_representation='log_moment_v1',
                native_float64_moment_unrepresentable=not (np.isfinite(var) and var>0 and np.isfinite(m2) and m2>0)))
    np.savez_compressed(cell/'diagnostics.npz', **diagnostics)
    failed = any(r['forecast_failed'] for r in all_rows)
    payload=dict(rows=all_rows, audit=dict(model='gpvol_likelihood_v1', seed=seed, market=market,
        settings=settings, runner='explicit CPU likelihood-v1; no old state imports',
        provenance=repair.provenance(), segments=phase_receipts,
        bundle_sha256=original_runner.sha(original_runner.FRONTIER/'bundle.json'),
        market_source_sha256=original_runner.sha(original_runner.ROOT/bundle['market_paths'][market]),
        failed=failed, full_ranking_established=False))
    # Invalid/nonfinite diagnostics are retained as explicit failure evidence,
    # never emitted as an apparently scoreable forecast cell.
    dest=cell/('failure.json' if failed else 'forecast.json')
    dest.write_text(json.dumps(json_safe(payload),indent=2,allow_nan=False)+'\n')
    (cell/(dest.name+'.sha256')).write_text(original_runner.sha(dest)+'\n')
    return dict(market=market,seed=seed,failed=failed,phases=phase_receipts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markets',nargs='+',help='diagnostic subset; omitted means all 28')
    p.add_argument('--seeds',nargs='+',type=int,choices=(1,2),default=[1,2])
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    bundle,rows=original_runner.load_plan()
    roster=a.markets or bundle['markets']
    if not set(roster).issubset(bundle['markets']):
        raise ValueError('Market outside original roster')
    manifest=dict(provenance=repair.provenance(),markets=roster,seeds=a.seeds,
        complete_primary_roster=(set(roster)==set(bundle['markets']) and set(a.seeds)=={1,2}),
        no_diagnostic_indexing=True,particles=200,burn_in=100,shrinkage=.95,
        standardise_window=1000,source_bundle_sha256=original_runner.sha(original_runner.FRONTIER/'bundle.json'),
        status='running; no empirical ranking',full_ranking_verified=False)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    for seed in a.seeds:
        for market in roster:
            result=full_cell(market,seed,bundle,rows,a.out)
            results.append(result)
            (a.out/'execution_receipt.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    manifest['status']='completed forecast execution; coverage/scoring/inference still require verification'
    manifest['failed_cells']=sum(r['failed'] for r in results)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
