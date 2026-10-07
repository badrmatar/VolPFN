"""Inference-only native-tail comparison; all writes stay in this study folder."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats
from volpfn.inference import load_model
from volpfn.preprocessing import CAUSAL_SPEC, NORMALIZATION_SPEC
from calibeat import market as M
from calibeat.fz import fz0
from mixtures import predict_mixture, combine
from tails import var_es, numerical_es
from diagnostics import holm, coverage

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def protocol(check_inputs=True):
    assert sha(HERE/'protocol.json')==(HERE/'protocol.sha256').read_text().strip(),'Protocol changed'
    p=json.loads((HERE/'protocol.json').read_text())
    if check_inputs:
        for name,digest in p['protected_sha256'].items():
            assert sha(name)==digest,'Protected file changed: '+name
    return p


def data(p,market):
    with np.load(Path(p['data_dir'])/(market+'.npz'),allow_pickle=False) as z:
        date=z['date'].astype('U10');r=z['returns'].astype(float)*100;split=z['split'].astype(int)
    assert np.isfinite(r).all() and np.all(date[1:]>date[:-1])
    return date,r,split


def infer(p):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    settings=p['inference']
    started=time.time()
    for entry in p['checkpoints']:
        seed=entry['seed']
        folder=OUT/'mixtures'/str(seed);folder.mkdir(parents=True,exist_ok=True)
        ck=torch.load(entry['path'],map_location='cpu',weights_only=False)
        assert ck['seed']==seed and ck['step']==20000
        assert ck['causal_spec']==CAUSAL_SPEC and ck['normalization_spec']==NORMALIZATION_SPEC
        model=load_model(entry['path'],settings['device'])
        for name in p['markets']:
            target=folder/(name+'.npz')
            if target.exists():
                continue
            date,r,split=data(p,name)
            pred=predict_mixture(model,r,**{k:settings[k] for k in ('L','stride','batch_windows','min_history')})
            assert all(np.isfinite(v[1:]).all() for v in pred.values())
            assert np.all(pred['df'][1:]>2) and np.all(pred['scale'][1:]>0)
            np.savez_compressed(target,date=date,**pred)
            print('INFERENCE',seed,name,'elapsed',round(time.time()-started,1),flush=True)
        del model
        torch.mps.empty_cache()
    save(OUT/'inference.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
         seconds=time.time()-started,checkpoints=p['checkpoints'],device=settings['device'],
         mixture_files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'mixtures').glob('*/*.npz')}))


def make_tails(p):
    crosschecks=[];caps=[]
    folder=OUT/'tails';folder.mkdir(exist_ok=True)
    for name in p['markets']:
        date,r,split=data(p,name)
        seeds=[]
        for seed in p['seeds']:
            with np.load(OUT/'mixtures'/str(seed)/(name+'.npz'),allow_pickle=False) as z:
                assert np.array_equal(date,z['date'])
                seeds.append({k:z[k][1:] for k in ('weight','location','scale','df')})
        mix=combine(seeds)
        rows=np.flatnonzero(split[1:]>0)
        checked=rows[[0,len(rows)//2,-1]]
        saved=dict(date=date[1:],returns=r[1:],split=split[1:])
        for alpha in p['tails']['alphas']:
            q,e,res=var_es(mix['weight'],mix['location'],mix['scale'],mix['df'],alpha)
            for index in checked:
                numerical,error=numerical_es(*(mix[k][index] for k in ('weight','location','scale','df')),alpha,q[index])
                difference=abs(numerical-e[index])
                assert difference <= 2e-7+2e-7*abs(e[index]),(name,alpha,index,difference)
                crosschecks.append(dict(market=name,date=str(date[index+1]),alpha=alpha,analytic=float(e[index]),numerical=float(numerical),absolute_error=float(difference),quadrature_error=float(error)))
            v=np.minimum(q,M.V_CAP);es=np.minimum(e,v-1e-6)
            vw,ew=M.calibeat('ratio_win1000',r[1:],v,es,alpha)
            vg,eg=M.switch(fz0(r[1:],v,es,alpha),fz0(r[1:],vw,ew,alpha),v,es,vw,ew,
                          window=M.GUARD_WINDOW,margin=M.GUARD_MARGIN[alpha])
            for label,a in [('raw_var',q),('raw_es',e),('native_var',v),('native_es',es),('guard_var',vg),('guard_es',eg)]:
                saved[f'{label}|{alpha}']=a
            for phase,s in [('development',1),('later',2)]:
                ix=split[1:]==s
                caps.append(dict(market=name,panel=phase,alpha=alpha,days=int(ix.sum()),var_caps=int(np.sum((q!=v)&ix)),es_caps=int(np.sum((e!=es)&ix)),max_cdf_residual=float(np.max(res[ix]))))
        np.savez_compressed(folder/(name+'.npz'),**saved)
        print('TAILS',name,flush=True)
    pd.DataFrame(crosschecks).to_csv(OUT/'es_quadrature_checks.csv',index=False)
    pd.DataFrame(caps).to_csv(OUT/'numerical_audit.csv',index=False)


def score(p):
    records=[];per_market=[];diag=[];diag_summary=[];matches=[]
    for phase,s in [('later',2),('development',1)]:
        roots=p['comparator_dirs'][phase]
        _,daily,dates=M.load_cells(roots[0],roots[1:])
        assert sorted(daily)==sorted(p['markets'])
        nband=int(np.mean([len(dates[m]) for m in p['markets']]))
        for name in p['markets']:
            with np.load(OUT/'tails'/(name+'.npz'),allow_pickle=False) as z:
                mask=z['split']==s
                assert np.array_equal(z['date'][mask],dates[name]),('Unmatched dates',phase,name)
                original_date,_,_=data(p,name)
                assert np.searchsorted(original_date,dates[name][0])>=100
                y=z['returns'][mask]
                matches.append(dict(panel=phase,market=name,days=len(y),first_date=str(dates[name][0]),last_date=str(dates[name][-1])))
                for alpha in p['tails']['alphas']:
                    for method in ('native','guard'):
                        v=z[f'{method}_var|{alpha}'][mask];e=z[f'{method}_es|{alpha}'][mask]
                        assert np.isfinite(v).all() and np.isfinite(e).all() and np.isfinite(y).all()
                        values=fz0(y,v,e,alpha)
                        assert np.isfinite(values).all()
                        daily[name][f'{method}|{alpha}|original']=values
                        d=coverage(y<=v,alpha)
                        low,high=stats.binom.ppf([.025,.975],nband,alpha)/nband
                        ownlow,ownhigh=stats.binom.ppf([.025,.975],len(y),alpha)/len(y)
                        diag.append(dict(panel=phase,market=name,method=method,alpha=alpha,**d,
                                         panel_band_n=nband,band_low=low,band_high=high,in_band=bool(low<=d['hit_rate']<=high),
                                         in_market_n_band=bool(ownlow<=d['hit_rate']<=ownhigh),reject_ind=d['p_ind']<.05,reject_cc=d['p_cc']<.05))
        for alpha in p['tails']['alphas']:
            for comparator in ('fhs_garch','garch_t','gas_fz0','volpfn_t8','native'):
                left='guard' if comparator=='native' else 'native'
                keya=f'{left}|{alpha}|original';keyb=f'{comparator}|{alpha}|original'
                matrix,names=M.diff_matrix(daily,dates,keya,keyb)
                assert len(names)==28 and sorted(names)==sorted(p['markets'])
                result=M.paired_block(matrix)
                contrast=f'{left}_minus_{comparator}'
                primary=phase=='later' and comparator in ('fhs_garch','garch_t')
                record=dict(panel=phase,contrast=contrast,alpha=alpha,primary=primary,
                            mean_difference=result['mean_difference'],ci_low=result['interval95'][0],ci_high=result['interval95'][1],
                            raw_p=result['p_two_sided'],holm_p=None,markets_improved=result['markets_improved'],markets=28)
                records.append(record)
                # Per-market intervals retain the same complete panel calendar (NaNs for market holidays).
                for j,name in enumerate(names):
                    local=M.paired_block(matrix[:,j:j+1])
                    per_market.append(dict(panel=phase,market=name,contrast=contrast,alpha=alpha,matched_days=int(np.isfinite(matrix[:,j]).sum()),
                                           mean_difference=local['mean_difference'],ci_low=local['interval95'][0],ci_high=local['interval95'][1],raw_p=local['p_two_sided']))
                print('SCORED',phase,contrast,alpha,flush=True)
    primary=[r for r in records if r['primary']]
    assert len(primary)==6
    for r,adjusted in zip(primary,holm([r['raw_p'] for r in primary])):
        r['holm_p']=float(adjusted)
    for r in records:
        if r['ci_low']<=0<=r['ci_high']:
            r['decision']='tie (interval includes zero)'
        elif (r['holm_p'] if r['primary'] else r['raw_p'])<.05:
            r['decision']='beats' if r['mean_difference']<0 else 'worse'
        else:
            r['decision']='no adjusted evidence' if r['primary'] else 'no unadjusted evidence'
    frame=pd.DataFrame(diag)
    for (phase,method,alpha),g in frame.groupby(['panel','method','alpha']):
        diag_summary.append(dict(panel=phase,method=method,alpha=alpha,markets=len(g),in_hit_band=int(g.in_band.sum()),
                                 in_market_n_band=int(g.in_market_n_band.sum()),christoffersen_ind_rejections=int(g.reject_ind.sum()),
                                 christoffersen_cc_rejections=int(g.reject_cc.sum()),panel_band_n=int(g.panel_band_n.iloc[0])))
    pd.DataFrame(records).to_csv(OUT/'comparisons.csv',index=False)
    pd.DataFrame(per_market).to_csv(OUT/'per_market_comparisons.csv',index=False)
    frame.to_csv(OUT/'per_market_diagnostics.csv',index=False)
    pd.DataFrame(diag_summary).to_csv(OUT/'diagnostic_summary.csv',index=False)
    pd.DataFrame(matches).to_csv(OUT/'matched_dates.csv',index=False)
    save(OUT/'comparisons.json',records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['infer','tails','score','verify']);args=ap.parse_args()
    p=protocol();OUT.mkdir(exist_ok=True)
    if args.stage=='infer':infer(p)
    elif args.stage=='tails':make_tails(p)
    elif args.stage=='score':score(p)
    else:
        protocol()
        save(OUT/'preservation.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(HERE/'protocol.json'),protected_files=len(p['protected_sha256']),all_unchanged=True))
        print('All protected inputs, scripts, results and tracked dissertation files unchanged.')


if __name__=='__main__':
    main()
