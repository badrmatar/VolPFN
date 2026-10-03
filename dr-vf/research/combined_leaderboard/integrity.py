"""Reproduce previously published gains without computing new leaderboard losses."""
from __future__ import annotations
import numpy as np
from research.frontier_benchmark.metrics import qlike, fit_scale
from .common import *

EXPECTED = {'chronos2':.08083137312229247,'moirai2':.07250414935510954,
            'tabpfn_ts':.14109471649574942}

def archived_reproduction():
    rows=read(CACHE/'8101_evaluation.json')
    markets=np.array([r['market'] for r in rows]); dates=np.array([r['date'] for r in rows])
    target=np.array([r['target']**2 for r in rows])
    equal=lambda v: float(np.mean([v[markets==m].mean() for m in np.unique(markets)]))
    losses=[]; scales={}
    for seed in SEEDS:
        evaluation=read(CACHE/f'{seed}_evaluation.json')
        require([key(r) for r in evaluation]==[key(r) for r in rows], 'Cache alignment failed')
        c=read(CACHE/f'{seed}_calibration.json')
        scale=fit_scale([r['target']**2 for r in c],[r['h'] for r in c],
                        [r['market'] for r in c],[r['date'] for r in c])
        scales[str(seed)]=scale
        losses.append(qlike(target,scale*np.array([r['h'] for r in evaluation])))
    own=np.mean(losses,axis=0)
    old_scales=read(FRONTIER/'study.json')['scales']
    result={}
    for model in FMS:
        c=[r for r in read(DEV/'models'/model/'predictions.json') if r['phase']=='calibration']
        scale=fit_scale([r['target']**2 for r in c],[r['h'] for r in c],
                        [r['market'] for r in c],[r['date'] for r in c])
        require(np.isclose(scale,old_scales[model],rtol=1e-12,atol=1e-12),'FM scale mismatch')
        lookup={key(r):r for r in read(FRONTIER/'models'/model/'predictions.json')}
        h=np.array([lookup[key(r)]['h'] for r in rows])
        gain=equal(qlike(target,h*scale)-own)
        require(abs(gain-EXPECTED[model])<1e-10,'Archived gain did not reproduce: '+model)
        result[model]=dict(gain=gain,expected=EXPECTED[model],scale=scale)
    return dict(status='passed',created_utc=now(),rows=len(rows),dates=len(set(dates)),
                seed_scales=scales,comparisons=result,tolerance=1e-10,
                scope='Old S_all/r2/foundation comparison only; not a new leaderboard result')

if __name__=='__main__':
    print(json.dumps(archived_reproduction(),indent=2))
