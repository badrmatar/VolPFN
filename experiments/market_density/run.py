"""Frozen development-only experiment; per-market checkpoint outputs."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from experiments.market_density.models import fit, advance, simulate, score, METHODS, PROBS, HORIZONS

DATA = Path('experiments/data/oxfordman_barde/development_v1')
OUTPUT = Path('experiments/outputs/market_density_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def run(root=OUTPUT):
    paths=sorted(DATA.glob('*.npz'))
    assert len(paths)==28
    root.mkdir(parents=True,exist_ok=False)
    (root/'source').mkdir();(root/'forecasts').mkdir();(root/'fits').mkdir()
    sources=sorted(Path('experiments/market_density').glob('*.py'))+[Path('experiments/market_density/PROTOCOL.md'),Path('experiments/test_market_density.py')]
    inputs=paths+[DATA/'manifest.json']
    hashes={str(p):digest(p) for p in sources+inputs}
    for p in sources:shutil.copy2(p,root/'source'/p.name)
    (root/'manifest.json').write_text(json.dumps(dict(hashes=hashes,methods=METHODS,horizons=HORIZONS.tolist(),probabilities=PROBS.tolist(),draws=4096,development_only=True),indent=2))
    start=time.time();cells=0;totalorigins=0
    for market,p in enumerate(paths):
        with np.load(p) as d:
            dates=d['date'].copy();split=d['split'].copy()
            values={k:d[k].copy() for k in ['rv5','rk_parzen']}
        assert np.all(dates<='2014-12-31') and np.all(split<2) and np.all(dates[1:]>dates[:-1])
        first=int(np.flatnonzero(split==1)[0]);assert first>=1022
        origins=np.arange(first,len(dates)-21,5)
        for proxy,measure in enumerate(['rv5','rk_parzen']):
            v=values[measure];assert np.all(np.isfinite(v)&(v>0))
            qs=[];scores=[];truths=[];scales=[];stress=[];negative=[];repeat=[];fitmeta=[];fitdates=[]
            lastfit=-100000;s=None
            for oi,t in enumerate(origins):
                history=v[:t]
                if t-lastfit>=22:
                    s=fit(history);lastfit=int(t)
                    fitmeta.append(dict(origin=int(t),date=str(dates[t]),last_input_date=str(dates[t-1]),state=jsonable(s)))
                else:advance(s,history)
                qrow=[];nrow=[];rerow=[]
                for mi,method in enumerate(METHODS):
                    seed=np.random.SeedSequence([107001,market,proxy,int(t),mi,0])
                    paths_drawn,neg=simulate(s,history,method,np.random.default_rng(seed))
                    q=np.quantile(paths_drawn,PROBS,axis=0).T
                    qrow.append(q);nrow.append(neg)
                    if oi%20==0:
                        seed2=np.random.SeedSequence([107001,market,proxy,int(t),mi,1])
                        again,_=simulate(s,history,method,np.random.default_rng(seed2))
                        rerow.append(np.quantile(again,PROBS,axis=0).T)
                # Labels are accessed only after every method has produced its forecast.
                truth=np.cumsum(v[t:t+22])[HORIZONS-1]
                qs.append(qrow);truths.append(truth);negative.append(nrow)
                scores.append([score(q,truth,s['scale']) for q in qrow])
                scales.append(s['scale']);stress.append(v[t-5:t].mean()>s['stress_threshold']);fitdates.append(lastfit)
                if rerow:repeat.append(dict(origin_number=oi,quantiles=np.array(rerow).tolist()))
            np.savez_compressed(root/'forecasts'/f'{p.stem}_{measure}.npz',dates=dates[origins],origins=origins,
                fit_origins=fitdates,quantiles=qs,scores=scores,truth=truths,scale=scales,stress=stress,negative_root_paths=negative)
            (root/'fits'/f'{p.stem}_{measure}.json').write_text(json.dumps(fitmeta))
            (root/'forecasts'/f'{p.stem}_{measure}_repeat.json').write_text(json.dumps(repeat))
            cells+=1;totalorigins+=len(origins)
            status=dict(state='running',complete_cells=cells,total_cells=56,forecast_origins=totalorigins,elapsed_seconds=round(time.time()-start,1))
            (root/'status.json').write_text(json.dumps(status,indent=2))
            print(f'{cells}/56 {p.stem} {measure}: {len(origins)} origins; elapsed {time.time()-start:.1f}s',flush=True)
    assert all(digest(p)==h for p,h in hashes.items())
    original=json.loads(Path('research_baselines/20260905_pre_audit/manifest.json').read_text())['files']
    assert all(digest(p)==r['sha256'] for p,r in original.items())
    (root/'verification.json').write_text(json.dumps(dict(original_files_unchanged=len(original),source_and_inputs_unchanged=True,
        reserved_rows_read=0,cells=cells,forecast_origins=totalorigins,finite_forecasts=True),indent=2))
    status.update(state='forecasting_complete',elapsed_seconds=round(time.time()-start,1))
    (root/'status.json').write_text(json.dumps(status,indent=2))
    print('FORECASTING COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
