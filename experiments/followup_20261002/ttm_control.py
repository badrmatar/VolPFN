"""TTM-only online controls using the original, unchanged forecast panel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
SOURCE = Path('/Users/badrmatar/.codex/worktrees/e786/Diss_Project/dr-vf/experiments/outputs/intraday_extension_20260930_comparison_final_v3')
BLEND = 'volpfn_20260930_full+ttm_rk'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')


def qlike(y, h):
    y, h = np.asarray(y, float), np.asarray(h, float)
    if y.shape != h.shape or not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError('Invalid targets or issued forecast')
    value = y/h+np.log(h)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite score')
    return value


def calibrate(didx, h, y, affine=True):
    didx, h, y = np.asarray(didx), np.asarray(h, float), np.asarray(y, float)
    qlike(y,h)
    if didx.shape != y.shape or not np.issubdtype(didx.dtype, np.integer) or (didx < 0).any():
        raise ValueError('Invalid chronological indices')
    logh = np.log(h)
    out, weights = np.empty(len(y)), []
    for start in range(0, int(didx.max())+1, 20):
        params = np.array([0.,1.])
        past = didx < start
        if start >= 60:
            if affine:
                def objective(p):
                    z = p[0]+p[1]*logh[past]
                    with np.errstate(over='ignore', invalid='ignore'):
                        value = np.mean(y[past]*np.exp(-z)+z)
                    return float(value) if np.isfinite(value) else np.inf
                fit = minimize(objective, params, method='Nelder-Mead',
                               options={'xatol':1e-6,'fatol':1e-10,'maxiter':2000})
                if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun) or fit.fun > objective(params)+1e-8:
                    raise RuntimeError('Calibration optimizer failed: '+str(fit.message))
                params = fit.x
            else:
                multiplier = np.mean(y[past]/h[past])
                if not np.isfinite(multiplier) or multiplier <= 0:
                    raise RuntimeError('No positive finite scale optimum')
                params[0] = np.log(multiplier)
        sel = (didx >= start)&(didx < start+20)
        out[sel] = np.exp(params[0]+params[1]*logh[sel])
        weights.append(dict(from_date=int(start), a=float(params[0]), b=float(params[1]),
                            fit_rows=int(past.sum()) if start>=60 else 0))
    qlike(y,out)
    return out, weights


def self_test():
    rng=np.random.default_rng(82);didx=np.repeat(np.arange(100),3)
    h=np.exp(rng.normal(size=len(didx)));y=rng.lognormal(size=len(didx));changed=y.copy();changed[didx>=80]*=100
    for affine in (True,False):
        a,_=calibrate(didx,h,y,affine);b,_=calibrate(didx,h,changed,affine)
        np.testing.assert_array_equal(a[didx<100],b[didx<100])
        np.testing.assert_allclose(a[didx<60],h[didx<60],rtol=1e-14,atol=0)
    print('Same-date/future-target invariance and raw warm-up checks passed')


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    pins={}
    for name in ['common_mask.csv','row_forecasts/ttm_rk.csv.gz','row_forecasts/'+BLEND+'.csv.gz',
                 'row_forecasts/gjr_garch_t+ttm_rk.csv.gz']:
        p=SOURCE/name;q=out/'inputs'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        pins[str(q.relative_to(out))]=sha(q)
    shutil.copy2(__file__,out/'ttm_control_source.py');pins['ttm_control_source.py']=sha(__file__)
    save(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='retrospective_TTM_online_calibration_control_v1',input_hashes=pins,
        primary_control='ttm_affine: log h = a + b log hTTM, two fitted parameters like the blend',
        secondary_control='ttm_scale: analytic multiplicative calibration, one parameter',
        schedule='First 60 union dates raw TTM; pooled QLIKE on strictly earlier completed union dates only; refit every 20 union dates',
        proxies='Separate r2 and RK fits, identical original masks and units; not a coherent joint density',
        comparisons='Blend minus each control, both proxies; 4-slot Holm family; raw TTM and GJR+TTM descriptive',
        bootstrap=dict(resamples=20000,blocks=[60,20,120],seed=2026100203),
        scope='Conditional saved-forecast calendar uncertainty; excludes online calibration re-estimation, training/pretraining and research-selection uncertainty; exposed retrospective panel'))


def counts(n,reps,block,seed):
    rng=np.random.default_rng(seed);out=np.zeros((reps,n),float);offset=np.arange(block)
    for i in range(reps):
        idx=((rng.integers(0,n,size=int(np.ceil(n/block)))[:,None]+offset)%n).ravel()[:n]
        out[i]=np.bincount(idx,minlength=n)
    return out


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty_like(p)
    out[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out


def interval(estimate,draws,coverage=.95):
    tail=(1-coverage)/2;lo,hi=np.quantile(draws-estimate,[tail,1-tail])
    return [float(estimate-hi),float(estimate-lo)]


def run(out):
    out=Path(out);p=json.loads((out/'protocol.json').read_text())
    for name,digest in p['input_hashes'].items():
        if sha(out/name)!=digest:raise RuntimeError('Frozen input drift')
    if sha(__file__)!=p['input_hashes']['ttm_control_source.py']:raise RuntimeError('Live source drift')
    if (out/'leaderboard.csv').exists():raise FileExistsError('Completed result preserved')
    mask=pd.read_csv(out/'inputs/common_mask.csv',dtype={'date':str,'market':str})
    if mask.duplicated(['market','date']).any():raise ValueError('Duplicate mask')
    index=pd.MultiIndex.from_frame(mask[['market','date']]);calendar=sorted(mask.date.unique())
    didx=np.searchsorted(calendar,mask.date);models={};weights={}
    for name in ['ttm_rk',BLEND,'gjr_garch_t+ttm_rk']:
        f=pd.read_csv(out/'inputs/row_forecasts'/(name+'.csv.gz'),dtype={'seed':str,'date':str,'market':str})
        want=['none'] if name=='ttm_rk' else [str(x) for x in range(8101,8106)] if name==BLEND else ['none']
        if sorted(f.seed.unique())!=want:raise ValueError('Seed roster drift: '+name)
        aligned=[]
        for seed,g in f.groupby('seed'):
            g=g.set_index(['market','date'])
            if g.index.has_duplicates or len(g)!=len(index) or set(g.index)!=set(index):raise ValueError('Coverage drift')
            g=g.reindex(index).reset_index()
            for proxy in ['r2','rk']:
                np.testing.assert_allclose(g[proxy],mask[proxy],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(g[proxy+'_qlike'],qlike(g[proxy],g[proxy+'_forecast']),rtol=1e-12,atol=1e-12)
            aligned.append(g)
        models[name]=pd.concat(aligned,ignore_index=True)
    raw=models['ttm_rk']
    for name,affine in [('ttm_affine',True),('ttm_scale',False)]:
        f=mask.copy();f['model']=name;f['seed']='none'
        for proxy in ['r2','rk']:
            h,w=calibrate(didx,raw[proxy+'_forecast'],mask[proxy],affine)
            for row in w:
                row['block_start_date']=calendar[row['from_date']]
                row['latest_fit_target_date']=calendar[row['from_date']-1] if row['fit_rows'] else None
                if row['latest_fit_target_date'] and row['latest_fit_target_date']>=row['block_start_date']:raise ValueError('Noncausal fit')
            weights[name+'|'+proxy]=w
            f[proxy+'_forecast']=h;f[proxy+'_qlike']=qlike(mask[proxy],h)
        models[name]=f
    rows=[];loss={}
    (out/'row_forecasts').mkdir(exist_ok=False)
    for name,f in models.items():
        loss[name]=f.groupby(['market','date'])[['r2_qlike','rk_qlike']].mean().reindex(index).to_numpy()
        f.to_csv(out/'row_forecasts'/(name+'.csv.gz'),index=False,compression='gzip',float_format='%.17g')
        stats=f.groupby(['seed','market'])[['r2_qlike','rk_qlike']].mean().groupby('seed').mean().mean()
        rows.append(dict(model=name,**stats.to_dict()))
    leaderboard=pd.DataFrame(rows)
    for proxy in ['r2','rk']:leaderboard[proxy+'_rank']=leaderboard[proxy+'_qlike'].rank(method='min')
    leaderboard.to_csv(out/'leaderboard.csv',index=False)
    save(out/'weights.json',weights)
    results=[]
    for block in [60,20,120]:
        c=counts(len(calendar),20000,block,2026100203);contrasts=[]
        for control in ['ttm_affine','ttm_scale']:
            for j,proxy in enumerate(['r2','rk']):
                gains=loss[control][:,j]-loss[BLEND][:,j]
                matrix=pd.DataFrame({'date':mask.date,'market':mask.market,'gain':gains}).pivot(index='date',columns='market',values='gain').reindex(calendar).to_numpy()
                valid=np.isfinite(matrix).astype(float);denom=c@valid
                if np.any(denom==0):raise ValueError('A draw omitted a market')
                estimate=float(np.nanmean(matrix,axis=0).mean())
                draws=np.mean((c@np.nan_to_num(matrix))/denom,axis=1)
                pvalue=float((1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/20001)
                seed_gains=[]
                for seed,g in models[BLEND].groupby('seed'):
                    delta=loss[control][:,j]-g[proxy+'_qlike'].to_numpy()
                    seed_gains.append(float(pd.DataFrame({'market':mask.market,'delta':delta}).groupby('market').delta.mean().mean()))
                contrasts.append(dict(block=block,control=control,proxy=proxy,gain=estimate,
                    interval95=interval(estimate,draws),family_interval95=interval(estimate,draws,1-.05/4),
                    p_raw=pvalue,seed_gains=seed_gains,markets_improved=int((np.nanmean(matrix,axis=0)>0).sum())))
        for row,adjusted in zip(contrasts,holm([r['p_raw'] for r in contrasts])):
            row['p_holm']=float(adjusted)
            row['supports_incremental_gain']=row['gain']>0 and adjusted<.05 and row['family_interval95'][0]>0 and all(x>0 for x in row['seed_gains'])
        results.extend(contrasts)
    save(out/'statistics.json',dict(direction='control loss minus blend loss; positive favours blend',family_slots=4,contrasts=results,scope=p['scope']))
    save(out/'receipt.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='receipt.json'})
    print(leaderboard.to_string(index=False))
    print('Primary contrasts:',json.dumps([r for r in results if r['block']==60]))


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('action',choices=['test','freeze','run']);cli.add_argument('--out',type=Path)
    a=cli.parse_args()
    if a.action=='test':self_test()
    elif a.action=='freeze':freeze(a.out)
    else:run(a.out)
