"""Inventory and data-defined mask only; no new forecast performance is inspected."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .common import *

def prepare(out=OUTPUT):
    out = Path(out)
    require(not (out/'bundle.json').exists(), 'Preparation already exists; refuse overwrite')
    old, prep = read(PLAN), read(PREP)
    inputs = {}
    def register(path):
        path = Path(path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path
    for p in (PLAN, PREP, PLAN.parent/'contexts.npz', CACHE/'results.json',
              FRONTIER/'study.json', DEV/'manifest.json'):
        register(p)
    require(sha(PLAN.parent/'contexts.npz') == old['contexts_sha256'], 'Old context hash mismatch')
    for p, digest in old['archived_files'].items():
        # Paths belonged to this same workspace; verify their identity before reuse.
        source = Path(p)
        if not source.exists():
            source = ROOT / ('experiments/'+p.split('/experiments/',1)[1])
        require(sha(source) == digest, 'Archived identity mismatch: '+str(source))
        register(source)
    origins = old['calibration_origins'] + old['origins']
    with np.load(PLAN.parent/'contexts.npz', allow_pickle=False) as z:
        expected_contexts = {phase:z[phase].copy() for phase in ('calibration','evaluation')}
    context_offsets = dict(calibration=0, evaluation=0)
    require(len(old['origins']) == 8774 and len(old['calibration_origins']) == 72, 'Wrong masks')
    markets = sorted({o['market'] for o in old['origins']})
    require(len(markets) == 28, 'Wrong fixed population')
    market_paths, source = {}, {}
    for name in markets:
        path = PREP.parent/'markets'/f'{name}.npz'
        require(sha(path) == prep['artifact_hashes'][f'markets/{name}.npz'], 'Prepared market changed')
        register(path)
        market_paths[name] = str(path.relative_to(ROOT))
        with np.load(path, allow_pickle=False) as z:
            source[name] = {k:z[k].copy() for k in ('date','r','rk')}
    rows = []
    for o in origins:
        m = source[o['market']]
        j = int(np.searchsorted(m['date'].astype('U10'), o['date']))
        require(j < len(m['r']) and str(m['date'][j]) == o['date'], 'Date absent from prepared history')
        require(np.isclose(m['r'][j], o['target'], rtol=1e-12, atol=1e-12), 'Return alignment differs')
        require(j >= 1024, 'Insufficient causal history')
        ci = context_offsets[o['phase']]
        require(np.array_equal(m['r'][j-1024:j], expected_contexts[o['phase']][ci]),
                'Full 1024-return context differs from archived input')
        context_offsets[o['phase']] += 1
        rk = float(m['rk'][j])
        rows.append(dict(market=o['market'], date=o['date'], phase=o['phase'],
            index=j, r2=float(o['target'])**2, rk=rk if np.isfinite(rk) and rk > 0 else None,
            primary_eligible=bool(np.isfinite(rk) and rk>0)))
    for s in SEEDS:
        for phase in ('calibration','evaluation'):
            rr = read(register(CACHE/f'{s}_{phase}.json'))
            expected = [o for o in origins if o['phase'] == phase]
            require([key(o) for o in rr] == [key(o) for o in expected], 'VolPFN cache mask mismatch')
    for model in FMS:
        for directory, phase in ((FRONTIER,'evaluation'),(DEV,'calibration')):
            p = register(directory/'models'/model/'predictions.json')
            rr = [o for o in read(p) if o['phase']==phase]
            expected = {key(o) for o in origins if o['phase']==phase}
            require({key(o) for o in rr} == expected and len(rr)==len(expected), 'FM coverage differs')
    from experiments.return_only_benchmark.volpfn_adapter import DEFAULT_REGISTRY, pinned_checkpoint
    register(DEFAULT_REGISTRY); register(DEFAULT_REGISTRY.with_suffix('.sha256'))
    checkpoints = {}
    for s in SEEDS:
        p, entry = pinned_checkpoint(s)
        register(p); checkpoints[str(s)] = str(p.relative_to(ROOT))
    evaluation = [r for r in rows if r['phase']=='evaluation']
    bundle = dict(created_utc=now(), rows=rows, markets=markets,
        market_paths=market_paths, checkpoints=checkpoints, inputs=inputs,
        units=prep['history_schema'], eligibility='archived S_all and finite positive target RK; independent of forecast success',
        counts=dict(S_all=len(evaluation), S=sum(r['primary_eligible'] for r in evaluation),
                    calibration=72, markets=28),
        missing_rk=[{k:r[k] for k in ('market','date')} for r in evaluation if not r['primary_eligible']],
        calibration='exact archived72; equal-market mean(r2/second_moment); one scalar per model/seed',
        exposure='Previously inspected 2015-04-01 to 2016-06-30; selected for forecast reuse; not fresh confirmation')
    save(out/'bundle.json', bundle)
    return bundle

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUTPUT)
    b=prepare(p.parse_args().out); print(b['counts'])
