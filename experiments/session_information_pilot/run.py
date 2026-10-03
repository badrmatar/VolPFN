"""Build/audit the pilot without changing any earlier experiment.

Empirical fitting requires explicit UTC session records and their provenance.
Use `python -m experiments.session_information_pilot.run --help`.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DATA = ROOT/'experiments/data/oxfordman_barde/calendar_full_v1'
SOURCE = ROOT/'experiments/data/oxfordman_barde/oxfordmanrealizedvolatilityindices.csv'
CHECKPOINT = ROOT/'out_volpfn/volpfn_base_best.pt'
EXTRACT = ROOT/'experiments/outputs/attribution_extract_full_v1'
PREVIOUS = ROOT/'experiments/outputs/evaluation_strengthening_v1'
FOLDS = ROOT/'experiments/outputs/attribution_driver_swap_v1/folds.json'
DEFAULT_OUT = ROOT/'experiments/outputs/session_information_pilot_v1'
MODELS = ('network', 'compact', 'gjr')
PARAMS = ('logw', 'loc', 'scale', 'nu')
BUFFER_MINUTES = 15


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def market_names():
    return json.loads((PREVIOUS/'protocol.json').read_text())['markets']


def locked_manifest(path, value=None):
    """Verify immutable manifest content and every file it identifies."""
    path = Path(path); digest = path.with_suffix(path.suffix+'.sha256')
    if path.exists():
        if not digest.exists() or sha(path) != digest.read_text().strip():
            raise RuntimeError('Manifest changed: '+str(path))
        actual = json.loads(path.read_text())
        if value is not None:
            # Timestamp is informational, not a reason to invalidate a resume.
            proposed = {k:v for k,v in value.items() if k!='created_utc'}
            prior = {k:v for k,v in actual.items() if k!='created_utc'}
            if proposed != prior:
                raise RuntimeError('Run inputs/specification changed; use a new output directory')
    else:
        if value is None: raise FileNotFoundError(path)
        save(path, value); digest.write_text(sha(path)+'\n'); actual = value
    for filename, expected in actual['hashes'].items():
        if not Path(filename).exists() or sha(filename) != expected:
            raise RuntimeError('Frozen input changed: '+filename)
    return actual


def preparation_manifest():
    files = list(PACKAGE.glob('*.py'))+[PACKAGE/'PROTOCOL.md', CHECKPOINT, FOLDS,
        ROOT/'src/volpfn/model.py', ROOT/'src/volpfn/prior.py', ROOT/'experiments/attribution/drivers.py',
        ROOT/'experiments/attribution/ladder.py', ROOT/'experiments/prior_adaptation/model.py',
        ROOT/'experiments/evaluation_strengthening/classical.py',
        ROOT/'experiments/evaluation_strengthening/statistics.py',
        PREVIOUS/'protocol.json', PREVIOUS/'scoring_mask.csv']
    files += [DATA/(n+'.npz') for n in market_names()]
    files += [EXTRACT/(n+'.npz') for n in market_names()]
    import importlib.metadata
    return {'created_utc': now(), 'kind':'base_forecast_preparation_only',
            'markets':market_names(), 'scope':'No cross-market coefficients or new performance comparisons',
            'environment': {p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','torch','arch')},
            'hashes':{str(p):sha(p) for p in sorted(set(files))}}


def audit(out):
    """Metadata/provenance inventory only, not a performance test."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    first_column=pd.read_csv(SOURCE,nrows=0).columns[0]
    raw=pd.read_csv(SOURCE, usecols=[first_column,'Symbol','open_time','close_time'])
    raw['date']=raw.iloc[:,0].astype(str).str[:10]
    coverage={}
    for name in market_names():
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as a:
            date=a['date'].astype('U10')
            coverage[name]={'first':str(date[0]),'last':str(date[-1]),'rows':len(date),
                'development_rows':int(((date>='2011-01-01')&(date<'2015-01-01')).sum()),
                'evaluation_rows':int((date>='2015-01-01').sum()),
                'positive_kernel_rows':int(np.sum(np.isfinite(a['rk_parzen'])&(a['rk_parzen']>0)))}
    examples=raw.loc[raw.Symbol.isin(['.N225','.FTSE','.SPX']) & raw.date.isin(['2014-01-06','2014-07-07'])]
    result={'created_utc':now(),'status':'awaiting_verified_session_records','markets':coverage,
        'source_sha256':sha(SOURCE),'source_has_intraday_kernel':True,
        'raw_hhmmss_fields_present':True,'raw_time_timezone_verified':False,
        'time_examples':examples[['Symbol','date','open_time','close_time']].to_dict('records'),
        'data_exposure':'2015-2022 panel already inspected; not a new holdout',
        'missing':['Verified historical UTC session records and source/return-horizon alignment',
                   'Documented UTC interpretation; raw HHMMSS fields cannot be assumed local or UTC'],
        'future_extension':'Post-2022 OHLC cannot supply realised kernels. Need separate intraday/kernel source and exposure review.'}
    save(out/'preflight.json',result)
    template={'session_sha256':'REPLACE_WITH_SHA256_OF_SESSION_CSV',
        'timestamp_basis':'historical_exchange_session_records',
        'timezone_evidence':'REPLACE_WITH_DOCUMENTED_TIMEZONE_SOURCE',
        'provenance_sources':[], 'target_alignment_verified':False,
        'availability_buffer_minutes':BUFFER_MINUTES,
        'latency_assumption':'15 minute idealised access delay; historical vendor latency is not observed',
        'data_exposure':'Retrospective previously inspected 2015-2022 panel'}
    if not (out/'session_provenance.template.json').exists(): save(out/'session_provenance.template.json',template)
    if not (out/'session_records.template.csv').exists():
        (out/'session_records.template.csv').write_text('market,date,open_utc,close_utc,available_utc\n')
    (out/'DATA_READINESS.md').write_text(
        '# Session pilot data readiness\n\n'
        'The 28 retained markets contain open-to-close returns and Parzen realised kernels through February 2022. '
        'The later period has already been inspected. No new empirical pilot score was calculated.\n\n'
        'The raw source has opening/closing HHMMSS fields, but their timezone interpretation and relationship '
        'to official session boundaries are not verified. Examples in preflight.json show seasonal shifts even '
        'for Tokyo. These fields are deliberately not converted by assumption.\n\n'
        'Required next input is an audited session-record CSV with market,date,open_utc,close_utc,available_utc, '
        'plus provenance that documents historical calendars, timezone conversion and return-horizon alignment. '
        'Availability must be close plus the registered 15-minute buffer.\n\n'
        'The runner refuses empirical coefficient fitting and scoring without these records. Base-forecast '
        'preparation and synthetic validation can proceed independently. Daily OHLC cannot reconstruct '
        'realised kernels for a post-2022 extension.\n')
    return result


def ewma_surprises(r):
    r=np.asarray(r,float)
    if r.ndim!=1 or len(r)<=100 or not np.isfinite(r).all():
        raise ValueError('Finite return history longer than 100 required')
    h=np.full(len(r),np.nan); h[100]=max(float(np.mean(r[:100]**2)),1e-8)
    for t in range(101,len(r)): h[t]=.94*h[t-1]+.06*r[t-1]**2
    u=np.full(len(r),np.nan); u[100:]=np.clip(r[100:]**2/h[100:]-1,-3,3)
    return u


def prepare(out, names=None):
    """Reconstruct distributions, including development, with no pilot scores."""
    from .baselines import compact_distribution, gjr_distribution
    import torch
    import volpfn.legacy as V
    from volpfn.prior import initial_scale
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    manifest=locked_manifest(out/'preparation.json',preparation_manifest())
    names=market_names() if names is None else names
    if not set(names).issubset(manifest['markets']): raise ValueError('Unknown market')
    cells=out/'forecasts'; cells.mkdir(exist_ok=True)
    torch.set_num_threads(2); model=None
    for name in names:
        dest=cells/(name+'.npz'); record=cells/(name+'.json')
        if dest.exists() or record.exists():
            if not dest.exists() or not record.exists(): raise RuntimeError('Incomplete cache '+name)
            a=json.loads(record.read_text())
            if a['preparation_sha256']!=sha(out/'preparation.json') or a['cache_sha256']!=sha(dest):
                raise RuntimeError('Stale/corrupt cache '+name)
            print('CACHED',name,flush=True); continue
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as a:
            data={k:a[k].copy() for k in a.files}
        r=data['returns'].astype(float)*100; date=data['date'].astype('U10')
        sc=initial_scale(r[:100]); m=dict(name=name,r=r,date=date,z=r/sc,scale=sc,split=data['split'])
        if model is None: model=V.load_model(str(CHECKPOINT),'cpu')
        net=V.predict_series(model,r,L=1024,stride=512,min_history=100,batch_windows=4,levels=())
        with np.load(EXTRACT/(name+'.npz'),allow_pickle=False) as a:
            np.testing.assert_array_equal(date,a['date'].astype('U10'))
            np.testing.assert_allclose(net['h'],a['h'],rtol=5e-5,atol=1e-6,equal_nan=True)
            np.testing.assert_allclose(net['logscore'],a['logscore'],rtol=0,atol=1e-4,equal_nan=True)
        from .scaling import scale_mixture, mixture_logpdf
        valid=np.arange(len(r))>=100
        reconstructed=scale_mixture(**{k:net[k][valid] for k in PARAMS},log_scale=0.)
        np.testing.assert_allclose(reconstructed['second_moment'],net['h'][valid],rtol=5e-5,atol=1e-6)
        np.testing.assert_allclose(mixture_logpdf(r[valid],**{k:net[k][valid] for k in PARAMS}),net['logscore'][valid],rtol=0,atol=1e-4)
        arrays={'date':date,'r':r,'rk':data['rk_parzen'].astype(float)*10000,
                'surprise':ewma_surprises(r)}
        metadata={'market':name,'preparation_sha256':sha(out/'preparation.json'),'models':{}}
        for key in PARAMS: arrays['network_'+key]=net[key]
        metadata['models']['network']={'status':'valid','reproduction':'archived second moments and log scores passed'}
        for which,fn in [('compact',lambda:compact_distribution(m,FOLDS)),('gjr',lambda:gjr_distribution(m))]:
            try:
                b=fn()
                for key in PARAMS: arrays[which+'_'+key]=b[key]
                metadata['models'][which]={'status':'valid','metadata':b.get('metadata',{}),'fits':b.get('fits',[])}
            except RuntimeError as exc:
                if which!='gjr': raise
                metadata['models'][which]={'status':'failed','reason':str(exc)}
                print('BENCHMARK UNAVAILABLE',name,which,str(exc),flush=True)
        np.savez_compressed(dest,**arrays)
        metadata['cache_sha256']=sha(dest); save(record,metadata)
        print('PREPARED',name,len(date),'observations; no pilot comparison scored',flush=True)
    locked_manifest(out/'preparation.json')
    done=sorted(p.stem for p in cells.glob('*.npz'))
    result={'status':'base_forecasts_complete' if done==sorted(market_names()) else 'base_forecasts_partial',
            'completed_markets':done,'total_markets':len(market_names()),'updated_utc':now(),
            'empirical_pilot':'awaiting verified UTC sessions; no correction fitted or comparison scored'}
    save(out/'build_status.json',result)
    return result


def validate_session_provenance(csv, provenance):
    p=json.loads(Path(provenance).read_text())
    if p.get('session_sha256')!=sha(csv): raise ValueError('Session hash missing or mismatched')
    if p.get('timestamp_basis')!='historical_exchange_session_records': raise ValueError('Historical session records required')
    if p.get('target_alignment_verified') is not True: raise ValueError('Return-horizon alignment must be verified')
    if not p.get('provenance_sources') or not p.get('timezone_evidence') or 'REPLACE' in str(p['timezone_evidence']):
        raise ValueError('Document session provenance and timezone evidence')
    if p.get('availability_buffer_minutes')!=BUFFER_MINUTES: raise ValueError('Wrong registered availability buffer')
    from .sessions import validate_sessions
    s=validate_sessions(pd.read_csv(csv,dtype={'market':str,'date':str}))
    if not ((s.available_utc-s.close_utc)==pd.Timedelta(minutes=BUFFER_MINUTES)).all():
        raise ValueError('Availability must equal close plus registered buffer')
    for market, rows in s.groupby('market'):
        rows=rows.sort_values('date')
        if (rows.open_utc.diff().dropna()<=pd.Timedelta(0)).any():
            raise ValueError('Session dates/timestamps are not ordered: '+market)
        if (rows.close_utc.shift(1)>=rows.open_utc).any():
            raise ValueError('Overlapping own-market sessions: '+market)
    return s


def fz0(r,q,e,alpha):
    r,q,e=np.broadcast_arrays(np.asarray(r,float),np.asarray(q,float),np.asarray(e,float))
    if not np.isfinite(e).all() or np.any(e>=0) or np.any(e>q):
        raise ValueError('FZ0 needs negative lower-tail ES no greater than its quantile')
    return -(r<=q).astype(float)*(q-r)/(alpha*e)+q/e+np.log(-e)-1


def evaluate(panel, distributions, features, reps=2000, block=60, seed=927601):
    """Fit only the declared development rows; score fixed later corrections.

    Pure-array interface also supports the synthetic end-to-end validation. All
    timestamps and file hashes must be checked by run() before empirical use.
    """
    from .scaling import fit_pooled_coefficient, mixture_logpdf, scale_mixture, mixture_lower_tail
    from experiments.evaluation_strengthening.statistics import qlike, paired_summary, holm
    p=panel.reset_index(drop=True); f=features.reset_index(drop=True)
    if not p[['market','date']].equals(f[['market','date']]): raise ValueError('Feature/forecast alignment mismatch')
    dev=(p.date>='2011-01-01')&(p.date<'2015-01-01'); test=p.date>='2015-01-01'
    if not dev.any() or not test.any(): raise ValueError('Both development and evaluation rows required')
    if set(p.loc[dev,'market'])!=set(p.loc[test,'market']): raise ValueError('Development/evaluation market mismatch')
    if not np.isfinite(p.r).all(): raise ValueError('Invalid target returns')
    fitted={}; losses={}; exploratory={}; forecasts={}
    r=p.r.to_numpy(float)
    for model in MODELS:
        if model not in distributions: continue
        d=distributions[model]
        if any(np.asarray(d[k]).shape[0]!=len(p) for k in PARAMS): raise ValueError('Wrong distribution rows')
        fitted[model]={}
        for variant in ('base','constant','lagged','latest'):
            x=(np.zeros(len(p)) if variant=='base' else np.ones(len(p)) if variant=='constant' else f[variant+'_x'].to_numpy(float))
            if not np.isfinite(x).all() or np.any(np.abs(x)>3+1e-12): raise ValueError('Invalid clipped feature')
            if variant=='base': c=0.
            else:
                fit=fit_pooled_coefficient(r[dev],x[dev],**{k:d[k][dev] for k in PARAMS},markets=p.loc[dev,'market'].to_numpy(),bound=1.)
                fitted[model][variant]=fit; c=fit['coefficient']
            idx=test.to_numpy(); log_s=c*x[idx]
            scaled=scale_mixture(**{k:d[k][idx] for k in PARAMS},log_scale=log_s)
            key=model+'_'+variant
            ld=mixture_logpdf(r[idx],**{k:scaled[k] for k in PARAMS})
            losses[key]={'r2':qlike(r[idx]**2,scaled['second_moment'])}
            rk=p.loc[test,'rk'].to_numpy(float); good=np.isfinite(rk)&(rk>0)
            rk_loss=np.full(len(rk),np.nan); rk_loss[good]=qlike(rk[good],scaled['variance'][good]); losses[key]['rk']=rk_loss
            forecasts[key]=dict(mean=scaled['mean'],variance=scaled['variance'],second_moment=scaled['second_moment'],logscore=ld,log_scale=log_s)
            exploratory[key]={'equal_market_logscore':float(pd.Series(ld).groupby(p.loc[test,'market'].to_numpy()).mean().mean()),'tails':{}}
            # Quantiles/ES scale exactly: compute base tails once per model.
            if variant=='base':
                base_tails={str(a):mixture_lower_tail(**{k:d[k][idx] for k in PARAMS},alpha=a) for a in (.01,.025)}
            for a in (.01,.025):
                tail=base_tails[str(a)]
                q=np.asarray(tail['quantile'])*np.exp(log_s); e=np.asarray(tail['expected_shortfall'])*np.exp(log_s)
                forecasts[key]['var_'+str(a)]=q; forecasts[key]['es_'+str(a)]=e
                try:
                    loss=fz0(r[idx],q,e,a)
                    value={'equal_market_fz0':float(pd.Series(loss).groupby(p.loc[test,'market'].to_numpy()).mean().mean()),
                           'pooled_exceedance_rate':float(np.mean(r[idx]<=q))}
                except ValueError as exc: value={'unavailable':str(exc)}
                exploratory[key]['tails'][str(a)]=value
    test_panel=p.loc[test].reset_index(drop=True)
    def contrast(candidate, control, target):
        if candidate not in losses or control not in losses:
            return {'status':'unavailable','p_null':1.,'candidate':candidate,'control':control,'target':target}
        a,b=losses[control][target],losses[candidate][target]; good=np.isfinite(a)&np.isfinite(b)
        if not good.any() or set(test_panel.loc[good,'market'])!=set(test_panel.market):
            return {'status':'unavailable','p_null':1.,'candidate':candidate,'control':control,'target':target,'reason':'Missing proxy coverage'}
        summary=paired_summary(test_panel.loc[good,'date'].to_numpy(),test_panel.loc[good,'market'].to_numpy(),a[good]-b[good],reps=reps,block=block,seed=seed)
        return {'status':'scored','candidate':candidate,'control':control,'target':target,**summary}
    family_a=[contrast(m+'_latest',m+'_base',t) for m in MODELS for t in ('r2','rk')]
    family_b=[contrast('network_latest','network_lagged',t) for t in ('r2','rk')]
    family_b += [{'status':'not_implemented_reserved_slot','candidate':'network_adaptive','control':'network_latest','target':t,'p_null':1.} for t in ('r2','rk')]
    for fam in (family_a,family_b):
        for row,adj in zip(fam,holm([x['p_null'] for x in fam])):
            row['holm_p']=float(adj)
            row['supported_positive']=bool(row['status']=='scored' and row['mean_gain']>0 and row['interval95'][0]>0 and adj<=.05)
    channel=all(row['supported_positive'] and row.get('mean_gain',0)>=.010 for row in family_a[:2])
    timing=all(row['supported_positive'] for row in family_b[:2])
    calibration_diagnostics=[]
    for m in MODELS:
        if m+'_constant' not in losses: continue
        for t in ('r2','rk'):
            gain=losses[m+'_constant'][t]-losses[m+'_latest'][t]
            good=np.isfinite(gain)
            mean=None if not good.any() else float(pd.Series(gain[good]).groupby(test_panel.loc[good,'market'].to_numpy()).mean().mean())
            calibration_diagnostics.append({'model':m,'target':t,'latest_minus_constant_qlike_gain':mean,
                'scope':'exploratory mean only; no significance/selection claim; positive favors latest information over constant scale calibration'})
    return {'coefficients':fitted,'family_a':family_a,'family_b':family_b,
            'channel_gate_passed':channel,'timing_gate_passed':timing,'exploratory':exploratory,
            'constant_calibration_diagnostics':calibration_diagnostics,
            'evaluation_rows':len(test_panel),'development_rows':int(dev.sum()),
            'interpretation':'Retrospective information pilot; no new method, no no-harm guarantee; ordinary intervals and family-specific adjusted tests'},test_panel,forecasts


def run(out, session_csv, provenance):
    from .sessions import build_session_features
    out=Path(out); locked_manifest(out/'preparation.json')
    sessions=validate_session_provenance(session_csv,provenance)
    frames=[]; donor_frames=[]; by_model={m:[] for m in MODELS}; unavailable=set(); caches=[]
    for name in market_names():
        cache=out/'forecasts'/(name+'.npz'); record=out/'forecasts'/(name+'.json')
        a=json.loads(record.read_text())
        if a['preparation_sha256']!=sha(out/'preparation.json') or a['cache_sha256']!=sha(cache): raise RuntimeError('Corrupt forecast cache '+name)
        caches += [cache,record]
        with np.load(cache,allow_pickle=False) as z:
            dates=z['date'].astype('U10'); mask=dates>='2011-01-01'
            frame=pd.DataFrame({'market':name,'date':dates[mask],'r':z['r'][mask],'rk':z['rk'][mask]})
            frames.append(frame)
            donor_frames.append(pd.DataFrame({'market':name,'date':dates[100:],'surprise':z['surprise'][100:]}))
            for m in MODELS:
                if a['models'][m]['status']!='valid': unavailable.add(m);continue
                by_model[m].append({k:z[m+'_'+k][mask] for k in PARAMS})
    panel=pd.concat(frames,ignore_index=True); donors=pd.concat(donor_frames,ignore_index=True)
    expected=pd.read_csv(PREVIOUS/'scoring_mask.csv',dtype=str)[['market','date']].sort_values(['market','date']).reset_index(drop=True)
    actual=panel.loc[panel.date>='2015-01-01',['market','date']].sort_values(['market','date']).reset_index(drop=True)
    if not actual.equals(expected): raise ValueError('Historical evaluation mask changed')
    source=sessions.merge(donors,on=['market','date'],how='right',validate='one_to_one',indicator=True)
    if not source['_merge'].eq('both').all(): raise ValueError('Session records missing for retained donor history')
    source=source.drop(columns='_merge')
    targets=panel[['market','date']].merge(sessions[['market','date','open_utc']],on=['market','date'],how='left',validate='one_to_one')
    if targets.open_utc.isna().any(): raise ValueError('Missing target-session opens')
    # The own-market historical base must itself be available at issuance.
    for name,g in source.groupby('market'):
        g=g.sort_values('date'); eligible=g.date>='2011-01-01'
        previous=g.available_utc.shift(1)
        if (eligible & previous.notna() & (previous>=g.open_utc)).any(): raise ValueError('Own-market history unavailable at open: '+name)
    features=build_session_features(source,targets=targets,max_age_hours=120.,clip=3.)
    features=panel[['market','date']].merge(features,on=['market','date'],how='left',validate='one_to_one')
    if features[['latest_x','lagged_x']].isna().any().any(): raise ValueError('Incomplete feature panel')
    if not (features.new_session_count>0).any(): raise ValueError('Timing positive control failed: no newly available sessions anywhere')
    hashes={str(Path(p).resolve()):sha(p) for p in caches+[Path(session_csv),Path(provenance),out/'preparation.json',out/'preflight.json']}
    spec={'created_utc':now(),'hashes':hashes,'kind':'retrospective_pilot','availability_buffer_minutes':15,
          'feature':'clipped EWMA squared-return surprise; equal matched donors; maxage120h',
          'unavailable_models':sorted(unavailable),'coefficient_fit_dates':['2011-01-01','2014-12-31'],
          'comparison_start':'2015-01-01','source_code_manifest':str(out/'preparation.json')}
    locked_manifest(out/'pilot_manifest.json',spec)
    if (out/'results.json').exists(): raise FileExistsError('Pilot results already exist; preserve the completed run')
    features.to_csv(out/'features.csv.gz',index=False,compression='gzip')
    distributions={m:{k:np.concatenate([d[k] for d in ds]) for k in PARAMS} for m,ds in by_model.items() if m not in unavailable}
    results,test_panel,forecasts=evaluate(panel,distributions,features)
    locked_manifest(out/'preparation.json');locked_manifest(out/'pilot_manifest.json')
    save(out/'results.json',results)
    for name,values in forecasts.items():
        np.savez_compressed(out/(name+'_evaluated.npz'),market=test_panel.market.to_numpy(dtype='U20'),date=test_panel.date.to_numpy(dtype='U10'),**values)
    lines=['# Foreign-session information pilot','',
        '**Retrospective, previously inspected panel.** No fresh holdout or methodological novelty established.','',
        f"Development rows: {results['development_rows']}; comparison rows: {results['evaluation_rows']}.",'',
        '| Family | Candidate / control | Target | Gain | Ordinary 95% interval | Holm p |',
        '|---|---|---|---:|---|---:|']
    for label in ('a','b'):
        for row in results['family_'+label]:
            gain=f"{row['mean_gain']:+.6f}" if row['status']=='scored' else row['status']
            lines.append(f"| {label.upper()} | {row['candidate']} / {row['control']} | {row['target']} | {gain} | {row.get('interval95','unavailable')} | {row['holm_p']:.4f} |")
    lines += ['',f"Channel gate: {results['channel_gate_passed']}. Timing gate: {results['timing_gate_passed']}.",
        '', 'A positive gate supports the specified information use; it does not establish a new inference method. '
        'Adaptive coefficients and guard protection are not tested. A constant-scale calibration control is reported '
        'descriptively in results.json: the channel gate alone cannot separate global recalibration from varying foreign information. '
        'Log scores and coherent density-derived tails are exploratory in results.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'build_status.json',{'status':'retrospective_pilot_complete','updated_utc':now()})
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['audit','prepare','run'])
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--markets',nargs='+')
    ap.add_argument('--sessions',type=Path);ap.add_argument('--session-provenance',type=Path)
    args=ap.parse_args()
    if args.action=='audit': result=audit(args.out)
    elif args.action=='prepare': result=prepare(args.out,args.markets)
    else:
        if not args.sessions or not args.session_provenance: ap.error('run requires --sessions and --session-provenance; no UTC assumptions are allowed')
        result=run(args.out,args.sessions,args.session_provenance)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','channel_gate_passed','timing_gate_passed','completed_markets')},indent=2))

if __name__=='__main__': main()
