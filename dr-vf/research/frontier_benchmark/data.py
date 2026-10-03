"""Development-only data boundary and strictly past contexts."""
from pathlib import Path
import json
import numpy as np
from research.seed_replication import causal

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'research/data/oxfordman_barde/calendar_full_v1'
INVENTORY=ROOT/'research/outputs/matched_prior_design_inventory_v1/inventory.json'
END='2014-12-31'


def markets():
    return json.loads(INVENTORY.read_text())['retained_index_population']


def load_development(market):
    if market not in markets():
        raise ValueError('Market outside fixed candidate inventory')
    with np.load(DATA/(market+'.npz'),allow_pickle=False) as z:
        mask=z['date']<=END
        date=z['date'][mask].copy()
        r=100.*z['returns'][mask].astype(float).copy()
    if not len(r) or not np.isfinite(r).all() or np.any(date[1:]<=date[:-1]):
        raise ValueError('Invalid development history')
    return dict(market=market,date=date,r=r)


def context(r,t,length=1024,min_history=100):
    r=np.asarray(r,dtype=float)
    if r.ndim!=1 or not min_history<=t<=len(r):
        raise ValueError('Insufficient history or invalid origin')
    past=r[max(0,t-length):t].copy()
    if not np.isfinite(past).all():
        raise ValueError('Nonfinite past history')
    return past


def covariates(past):
    """Historical target rows align to previous input; final row forecasts next."""
    f=causal.feature_batch(np.asarray(past,float)[None])['feats'][0]
    # Row0 has no past and is omitted, rather than using its own return.
    return f[:-1].copy(),f[-1:].copy()


def training_histories(held_out,end='2010-12-31'):
    if held_out not in markets() or end>END:
        raise ValueError('Invalid held-out identity or fit cutoff')
    result={}
    for market in markets():
        if market==held_out:
            continue
        a=load_development(market)
        keep=a['date']<=end
        result[market]={'date':a['date'][keep],'r':a['r'][keep]}
    if held_out in result or any(v['date'][-1]>end for v in result.values()):
        raise AssertionError('Training leakage')
    return result
