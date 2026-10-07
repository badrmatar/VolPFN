"""Fit and score nine post-hoc comparisons in a new output directory."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time,traceback
import numpy as np
import pandas as pd
from scipy import stats
from calibeat import market as M
from calibeat.fz import fz0
from models import gjr_paths,es_caviar_path

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def protocol(verify=True):
    assert sha(HERE/'POST_HOC_PROTOCOL.json')==(HERE/'POST_HOC_PROTOCOL.sha256').read_text().strip()
    p=json.loads((HERE/'POST_HOC_PROTOCOL.json').read_text())
    if verify:
        for name,digest in p['protected_sha256'].items():assert sha(name)==digest,'Existing file changed: '+name
    return p


def fit_market(name):
    p=protocol(False);began=time.time()
    with np.load(Path(p['data_dir'])/(name+'.npz'),allow_pickle=False) as z:
        date=z['date'].astype('U10');r=z['returns'].astype(float)*100;split=z['split']
    mask=split==2
    assert np.array_equal(mask,date>='2015-01-01')
    with np.load(Path(p['native_tails_dir'])/(name+'.npz'),allow_pickle=False) as z:
        native_mask=z['split']==2
        np.testing.assert_array_equal(date[mask],z['date'][native_mask])
        np.testing.assert_array_equal(r[mask],z['returns'][native_mask])
    pairs={};states={};audit={}
    try:pairs,states,audit['gjr']=gjr_paths(r,date,p['alphas'])
    except Exception as exc:
        audit['gjr']=dict(status='unavailable',exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        for model in ('gjr_garch_t','fhs_gjr_garch'):
            for a in p['alphas']:states[f'{model}|{a}']=dict(status='unavailable',reason=audit['gjr']['exception'])
    audit['es_caviar']={}
    for a in p['alphas']:
        key=f'es_caviar|{a}'
        try:
            pair,details=es_caviar_path(r,a)
            details.update(history_start=str(date[0]),history_end=str(date[999]))
            audit['es_caviar'][str(a)]=details
            if pair is None:
                states[key]=dict(status='unavailable',reason=details['reason'])
            else:
                v,e=(x[mask] for x in pair)
                if not np.isfinite(v).all() or not np.isfinite(e).all() or np.any(v>=0) or np.any(e>=v):
                    states[key]=dict(status='unavailable',reason='invalid_scored_es_caviar_tail_path')
                else:
                    pairs[key]=(v,e);states[key]=dict(status='complete')
        except Exception as exc:
            states[key]=dict(status='unavailable',reason=f'{type(exc).__name__}: {exc}')
            audit['es_caviar'][str(a)]=dict(status='unavailable',exception=states[key]['reason'],traceback=traceback.format_exc())
    arrays=dict(date=date[mask],returns=r[mask]);caps=[]
    for key,(v,e) in pairs.items():
        if not np.isfinite(v).all() or not np.isfinite(e).all() or np.any(e>=v):
            states[key]=dict(status='unavailable',reason='invalid_scored_tail_path');continue
        clipped_v=np.minimum(v,M.V_CAP);clipped_e=np.minimum(e,clipped_v-1e-6)
        arrays[key+'|raw_var']=v;arrays[key+'|raw_es']=e
        arrays[key+'|var']=clipped_v;arrays[key+'|es']=clipped_e
        caps.append(dict(model_level=key,var_caps=int(np.sum(v!=clipped_v)),es_caps=int(np.sum(e!=clipped_e))))
    audit.update(market=name,states=states,caps=caps,seconds=time.time()-began,scored_days=int(mask.sum()))
    np.savez_compressed(OUT/'forecasts'/(name+'.npz'),**arrays)
    save(OUT/'fits'/(name+'.json'),audit)
    return dict(market=name,seconds=round(time.time()-began,2),unavailable=sum(x['status']!='complete' for x in states.values()))


def fit(p,workers):
    (OUT/'forecasts').mkdir(parents=True,exist_ok=True);(OUT/'fits').mkdir(exist_ok=True)
    pending=[m for m in p['markets'] if not (OUT/'fits'/(m+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(fit_market,m):m for m in pending}
        for job in as_completed(jobs):print('FIT',json.dumps(job.result()),flush=True)


def holm(p):
    p=np.asarray(p,float);ix=np.argsort(p,kind='stable');out=np.empty_like(p)
    out[ix]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]))
    return out


def score(p):
    rows=[];failed_fits=[];unavailable=[];daily={};dates={};fit_counts={}
    for name in p['markets']:
        audit=json.loads((OUT/'fits'/(name+'.json')).read_text())
        g=audit['gjr']
        fit_counts[name]=dict(gjr_attempts=len(g.get('attempts',[])),gjr_rejected=sum(not a['accepted'] for a in g.get('attempts',[])),
                              es_candidates=sum(len(a.get('candidates',[])) for a in audit['es_caviar'].values()))
        for a in g.get('attempts',[]):
            if not a['accepted']:
                failed_fits.append(dict(market=name,model='GJR-GARCH-t / FHS-GJR',alpha=None,origin=a['origin'],kind='rejected_scheduled_refit',
                                         reason='; '.join(a['rejection_reasons']),action=a.get('action',''),selected=True))
        for a in g.get('fhs_errors',[]):
            failed_fits.append(dict(market=name,model='FHS-GJR',alpha=None,origin=a['origin'],kind='residual_tail_failure',reason=a['reason'],action='unavailable',selected=True))
        for a,fit_audit in audit['es_caviar'].items():
            for c in fit_audit.get('candidates',[]):
                if not c['success']:
                    failed_fits.append(dict(market=name,model='ES-CAViaR',alpha=float(a),origin=fit_audit.get('history_end'),kind='unsuccessful_optimizer_start',
                                             reason=c['message'],action='legacy minimum-objective selection unchanged',selected=c['start']==fit_audit['selected_start']))
        with np.load(OUT/'forecasts'/(name+'.npz')) as z,np.load(Path(p['native_tails_dir'])/(name+'.npz')) as native:
            mask=native['split']==2;dates[name]=z['date'];daily[name]={}
            np.testing.assert_array_equal(z['date'],native['date'][mask]);np.testing.assert_array_equal(z['returns'],native['returns'][mask])
            for alpha in p['alphas']:
                daily[name][f'native|{alpha}']=fz0(z['returns'],native[f'native_var|{alpha}'][mask],native[f'native_es|{alpha}'][mask],alpha)
                for model in p['models']:
                    key=f'{model}|{alpha}';state=audit['states'][key]
                    if state['status']!='complete':
                        unavailable.append(dict(market=name,model=model,alpha=alpha,reason=state['reason']));continue
                    daily[name][key]=fz0(z['returns'],z[key+'|var'],z[key+'|es'],alpha)
                    assert np.isfinite(daily[name][key]).all()
    for model in p['models']:
        for alpha in p['alphas']:
            key=f'{model}|{alpha}'
            missing=[m for m in p['markets'] if key not in daily[m]]
            row=dict(model=model,alpha=alpha,mean_difference=None,ci_low=None,ci_high=None,raw_p=None,holm_p=None,
                     holm_family_input=1.,markets=28-len(missing),missing_markets=','.join(missing),matched_rows=None,status='unavailable' if missing else 'complete')
            if not missing:
                matrix,names=M.diff_matrix(daily,dates,f'native|{alpha}',key)
                assert len(names)==28 and sum(len(dates[m]) for m in names)==50073
                result=M.paired_block(matrix)
                row.update(mean_difference=result['mean_difference'],ci_low=result['interval95'][0],ci_high=result['interval95'][1],raw_p=result['p_two_sided'],
                           holm_family_input=result['p_two_sided'],matched_rows=50073)
            rows.append(row)
    assert len(rows)==9
    for row,adjusted in zip(rows,holm([r['holm_family_input'] for r in rows])):
        if row['status']=='unavailable':row['decision']='unavailable; reserved Holm slot p=1';continue
        row['holm_p']=float(adjusted)
        if row['ci_low']<=0<=row['ci_high']:row['decision']='tie'
        elif adjusted<.05:row['decision']='beats' if row['mean_difference']<0 else 'worse'
        else:row['decision']='no adjusted evidence'
    save(OUT/'results.json',rows);pd.DataFrame(rows).to_csv(OUT/'results_table.csv',index=False)
    pd.DataFrame(failed_fits,columns=['market','model','alpha','origin','kind','reason','action','selected']).to_csv(OUT/'failed_fit_attempts.csv',index=False)
    pd.DataFrame(unavailable,columns=['market','model','alpha','reason']).to_csv(OUT/'unavailable_models.csv',index=False)
    save(OUT/'fit_counts.json',fit_counts)
    save(OUT/'failures.json',dict(unavailable_market_levels=unavailable,failed_fit_attempts=failed_fits))
    print('SCORED nine comparisons; unavailable cells',len(unavailable),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','score','verify']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    p=protocol();OUT.mkdir(exist_ok=True)
    if args.stage=='fit':fit(p,args.workers)
    elif args.stage=='score':score(p)
    else:
        save(OUT/'preservation.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),all_unchanged=True,protected_files=len(p['protected_sha256']),post_hoc_protocol_sha256=sha(HERE/'POST_HOC_PROTOCOL.json')))
        print('Protected original files and POST-HOC protocol unchanged.')


if __name__=='__main__':main()
