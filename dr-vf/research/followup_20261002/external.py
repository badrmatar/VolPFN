"""Frozen new ETF panel. Human exposure remains a separate explicit gate."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'external_confirmation_v1'
OLD=Path('/Users/badrmatar/.codex/worktrees/e786/Diss_Project/dr-vf')
TICKERS='XLB XLE XLF XLI XLK XLP XLU XLV XLY GLD SLV TLT IEF LQD HYG EWJ EWZ EWG EWU EWA'.split()


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    copies={}
    for source,name in [(Path(__file__),'external_source.py'),(HERE/'historical.py','historical_source.py'),
                        (OLD/'research/tier_a_confirmation/econometric.py','econometric.py')]:
        shutil.copy2(source,OUT/name);copies[name]=sha(OUT/name)
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),tickers=TICKERS,
        source_hashes=copies,start='2005-01-01',end_exclusive='2026-09-01',evaluation_start='2022-03-01',evaluation_end='2026-08-31',
        target='Next observed session 100*log(Close/Open), retain zero returns, no total-return or overnight substitution',
        forecast_context=1024,models=['full','noaux','historical_regularized','gjr_garch_t_rolling1000'],seeds=list(range(8101,8106)),
        neural_policy='Freeze original full/noaux checkpoint identities before data acquisition; historical checkpoint chosen only by already frozen 2011-2014 index-development procedure. No ETF data updates/selects neural weights.',
        econometric_policy='Registered existing rolling GJR-GARCH(1,1,1)-t with constant mean, 1000 past sessions, refit every 20 market sessions, public arch recursion, retain failed-refit carries and failures; same M2 target.',
        price_policy='Yahoo/yfinance daily unadjusted OHLC; auto_adjust=False/back_adjust=False/repair=False; vendor-local date labels, no forward fill, no outcome-driven ticker replacement',
        eligibility='All20 fixed tickers required; ETF metadata and US session timezone; 1024 valid pre-evaluation returns; at least800 test rows; invalid test OHLC<=1%; unique increasing dates; no performance filter',
        primary_family=['full_vs_gjr_qlike','noaux_vs_gjr_qlike','noaux_vs_historical_qlike'],family_slots=3,
        inference='20000 synchronized circular calendar resamples, blocks60(primary)/20/120, equal markets then seeds, Holm3-slot family, basic95% and Bonferroni family intervals; conditional forecasts only',
        seed=2026100204,secondary='Native return NLL and common zero-mean t8 NLL descriptive. No realised-kernel or TTM comparison: new panel lacks certified intraday proxy.',
        exposure='No ticker-named ETF results located in targeted main-project/worktree inventory. This does not prove zero human exposure; classification requires user attestation. Overlapping calendar dates/common shocks are not independent of index panels.',
        execution_gate='Do not score a claimed untouched confirmation until human exposure is established; prepare/source-check data while waiting.',
        scope='New external population, frozen before data download. Neural models trained on old indices/synthetic data. Not a prospective new-calendar holdout; vendor revisions/asset correlations limit interpretation.'))


def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    for name,digest in p['source_hashes'].items():
        if sha(OUT/name)!=digest:raise RuntimeError('Frozen source drift')
    if sha(__file__)!=p['source_hashes']['external_source.py']:raise RuntimeError('Live acquisition source drift')
    return p


def checkpoint_snapshot():
    import torch
    p=verify();receipt=json.loads((OLD/'research/outputs/experiment_programme_20260930/component_aggregate_v1/aggregation_receipt.json').read_text())
    folder=OUT/'checkpoints';folder.mkdir(exist_ok=False);identities=[]
    for shard in receipt['shards']:
        if shard['arm'] not in ['full','noaux']:continue
        paths=[Path(x) for x in shard['pins'] if x.endswith('/final.pt') and Path(x).exists()]
        if len(paths)!=1:raise ValueError('Ambiguous original checkpoint')
        source=paths[0]
        if sha(source)!=shard['checkpoint_sha256']:raise ValueError('Original checkpoint identity drift')
        ck=torch.load(source,map_location='cpu',weights_only=False)
        if ck['arm']!=shard['arm'] or ck['seed']!=shard['seed'] or ck['step']!=20000 or ck['smoke']:
            raise ValueError('Incomplete/wrong original model')
        dest=folder/(shard['arm']+'_'+str(shard['seed'])+'.pt')
        # Exact inference weights: optimizer/RNG/history omitted, never retrained.
        torch.save({k:ck[k] for k in ['state_dict','cfg','arm','seed','step','manifest_sha256','normalization_spec','causal_spec']},dest)
        identities.append(dict(arm=shard['arm'],seed=shard['seed'],original_path=str(source),original_sha256=sha(source),
                               file=str(dest.relative_to(OUT)),sha256=sha(dest),weight_conversion='Exact same state_dict; omit optimizer and training logs only'))
    if len(identities)!=10:raise ValueError('All ten original full/noaux models required')
    save(OUT/'checkpoint_identity.json',identities)


def acquire():
    p=verify()
    if not (OUT/'checkpoint_identity.json').exists():raise ValueError('Freeze model identities before data download')
    raw_dir=OUT/'raw';data_dir=OUT/'data';raw_dir.mkdir(exist_ok=False);data_dir.mkdir()
    import yfinance as yf
    yf.set_tz_cache_location(str(OUT/'vendor_cache'))
    members=[];mask=[]
    for ticker in p['tickers']:
        row=dict(ticker=ticker,status='unavailable')
        try:
            asset=yf.Ticker(ticker)
            frame=asset.history(start=p['start'],end=p['end_exclusive'],interval='1d',auto_adjust=False,
                                back_adjust=False,repair=False,actions=True,keepna=True,rounding=False,timeout=30,raise_errors=True)
            metadata=asset.get_history_metadata()
            save(raw_dir/(ticker+'_metadata.json'),metadata)
            frame.to_csv(raw_dir/(ticker+'.csv'),float_format='%.17g')
            if frame.empty:raise ValueError('Empty vendor history')
            if metadata.get('instrumentType')!='ETF' or metadata.get('exchangeTimezoneName')!='America/New_York':
                raise ValueError('ETF identity or US session timezone not established')
            dates=np.asarray(frame.index.strftime('%Y-%m-%d'))
            if len(set(dates))!=len(dates) or np.any(dates[1:]<=dates[:-1]):raise ValueError('Duplicate or unordered dates')
            prices=frame[['Open','High','Low','Close']].to_numpy(float)
            o,h,l,c=prices.T;tol=1e-8*np.nan_to_num(np.max(np.abs(prices),axis=1))
            valid=np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)&(l<=np.minimum(o,c)+tol)&(h+tol>=np.maximum(o,c))&(h+tol>=l)
            in_test=(dates>=p['evaluation_start'])&(dates<=p['evaluation_end'])
            invalid_fraction=float(np.mean(~valid[in_test])) if in_test.any() else 1.
            if invalid_fraction>.01:raise ValueError('Too many invalid test OHLC rows')
            rejected=dates[~valid].tolist();dates=dates[valid];r=100*np.log(c[valid]/o[valid])
            if not np.isfinite(r).all():raise ValueError('Nonfinite session returns')
            before=int(np.count_nonzero(dates<p['evaluation_start']));indices=np.flatnonzero((dates>=p['evaluation_start'])&(dates<=p['evaluation_end']))
            if before<1024 or len(indices)<800:raise ValueError('Insufficient context or test coverage')
            np.savez_compressed(data_dir/(ticker+'.npz'),date=dates.astype('U10'),returns=r)
            mask.extend(dict(market=ticker,date=str(dates[i]),row_index=int(i),return_value=float(r[i])) for i in indices)
            row.update(status='eligible',test_rows=len(indices),pretest_rows=before,invalid_test_fraction=invalid_fraction,
                       rejected_raw_dates=rejected,zero_test_returns=int(np.count_nonzero(r[indices]==0)),
                       source_sha256=sha(raw_dir/(ticker+'.csv')),data_sha256=sha(data_dir/(ticker+'.npz')))
        except Exception as e:row['failure']=repr(e)
        members.append(row);save(OUT/'data_status.json',dict(members=members))
        print('ACQUIRE',ticker,row['status'],row.get('test_rows'),row.get('failure'),flush=True)
    complete=all(x['status']=='eligible' for x in members) and len(members)==20
    pd.DataFrame(mask).rename(columns={'return_value':'return'}).to_csv(OUT/'mask.csv',index=False,float_format='%.17g')
    save(OUT/'DATA_RECEIPT.json',dict(status='complete' if complete else 'unavailable_incomplete_fixed_population',
        created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),
        model_identity_sha256=sha(OUT/'checkpoint_identity.json'),yfinance_version=yf.__version__,members=members,
        files={str(f.relative_to(OUT)):sha(f) for directory in [raw_dir,data_dir] for f in directory.rglob('*') if f.is_file()},
        mask_sha256=sha(OUT/'mask.csv'),scope='Price/identity checks only; no model forecasts scored'))


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('action',choices=['freeze','checkpoints','acquire']);a=cli.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='checkpoints':checkpoint_snapshot()
    else:acquire()
