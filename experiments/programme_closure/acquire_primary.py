"""Acquire the fixed index block after a separate vendor identity seal.

Outcome-blind: no model imports, inference, calibration or statistical scoring.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
from curl_cffi import requests
from experiments.matched_prior_confirmatory import data_gate as D


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def seal_mapping(metadata_root,out):
    root=Path(metadata_root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((root/'inventory.json').read_text())['rows']
    if len(rows)!=28: raise ValueError('Incomplete candidate metadata inventory')
    decisions=[]
    # Decisions after reading all vendor names/types, before acquiring full OHLC.
    # Oslo's GI versus original price-index convention remains unresolved.
    unresolved={'KSE':'The specified vendor index symbol returned404; no replacement index.',
                'SMSI':'The specified vendor index symbol returned404; no IBEX substitution.',
                'OSEAX':'Vendor labels this series GI; compatibility with the original index valuation convention is unresolved.'}
    for row in rows:
        market=row['market'];meta=row.get('meta',{})
        verified=(market not in unresolved and row.get('http_status')==200 and meta.get('instrumentType')=='INDEX')
        decision=dict(market=market,identity_verified=verified,vendor_symbol=row['symbol'],
            exact_index_name=meta.get('longName') or meta.get('shortName') or row['expected_identity'],
            intended_index=row['expected_identity'],
            identity_evidence=dict(vendor_metadata_url=row['url'],metadata_sha256=sha(root/(market+'.json')),
                vendor_symbol=meta.get('symbol'),instrument_type=meta.get('instrumentType'),
                reviewed_name=meta.get('longName') or meta.get('shortName'),
                scope='Vendor index identity/name and quote convention; not an independent exchange price audit'),
            price_basis='Unadjusted daily OHLC of the named vendor index; no adjusted close, ETF or dividend adjustment',
            session_date_convention='Chart timestamp converted to vendor exchangeTimezoneName, then local calendar date; daily interval',
            exchange_timezone=meta.get('exchangeTimezoneName'),
            exclusion_reason=unresolved.get(market) if not verified else None)
        decisions.append(decision)
    result=dict(candidate_population=[r['market'] for r in rows],locked_utc=now(),markets=decisions,
                known_earlier_contact='Whole-historyYahooaudits ofSPX,FTSE,GDAXI,N225; metadata request here returned up to one daily observation for available candidates; no forecast outputs generated.')
    save(out/'mapping.json',result);(out/'mapping.sha256').write_text(sha(out/'mapping.json')+'\n')
    return result


def canonical_chart(response,expected_symbol):
    result=response['chart']['result'][0];meta=result['meta']
    if meta['symbol']!=expected_symbol or meta['instrumentType']!='INDEX':
        raise ValueError('Vendor returned a different instrument')
    timestamps=result['timestamp'];quote_rows=result['indicators']['quote'][0]
    local_dates=pd.to_datetime(timestamps,unit='s',utc=True).tz_convert(meta['exchangeTimezoneName']).strftime('%Y-%m-%d')
    frame=pd.DataFrame({'date':local_dates,**{key:quote_rows[key] for key in ('open','high','low','close')}})
    if len(frame)!=len(timestamps): raise ValueError('Mismatched quote rows')
    return frame,meta


def acquire(mapping_dir,out):
    mapping_dir=Path(mapping_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    mapping_path=mapping_dir/'mapping.json'
    if sha(mapping_path)!=(mapping_dir/'mapping.sha256').read_text().strip(): raise ValueError('Mapping seal changed')
    mapping=json.loads(mapping_path.read_text())
    (out/'mapping.json').write_bytes(mapping_path.read_bytes())
    for sub in ('raw','canonical','returns'): (out/sub).mkdir()
    start=int(datetime(2010,1,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,9,1,tzinfo=timezone.utc).timestamp())
    reports={};market_files={};eligible=[];masks=[]
    session=requests.Session(impersonate='chrome')
    for entry in mapping['markets']:
        market=entry['market'];symbol=entry['vendor_symbol']
        if not entry['identity_verified']:
            reports[market]=dict(market=market,eligible=False,failure_codes=['index_identity_unresolved'],reason=entry['exclusion_reason'])
            continue
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')+f'?period1={start}&period2={end}&interval=1d&events=history'
        try:
            response=session.get(url,timeout=30)
            raw_path=out/'raw'/(market+'.vendor.json');raw_path.write_bytes(response.content)
            response.raise_for_status()
            body=response.json()
            if body['chart'].get('error'): raise ValueError(str(body['chart']['error']))
            frame,meta=canonical_chart(body,symbol)
            frame.to_csv(out/'canonical'/(market+'.csv'),index=False,float_format='%.17g')
            # The canonical file is the exact input the verifier reopens.
            canonical=pd.read_csv(out/'canonical'/(market+'.csv'))
            report,prepared=D.check_market(canonical,market,identity_verified=True)
            metadata=dict(vendor_symbol=symbol,retrieved_utc=now(),request=url,response_metadata=meta,
                source='YahooFinance chart endpoint',price_basis=entry['price_basis'],session_date_convention=entry['session_date_convention'],
                raw_sha256=sha(raw_path),identity_mapping_sha256=sha(mapping_path),
                transformations='Timezone to localdate, columnnames and round-trip17-significant-digit CSV only. No sorting,deduplication,forwardfill,adjustments or price replacement.')
            save(out/'raw'/(market+'.metadata.json'),metadata)
            np.savez_compressed(out/'returns'/(market+'.npz'),date=prepared.date.to_numpy(dtype='U10'),r=prepared.r.to_numpy(float))
            market_files[market]=dict(vendor_raw='raw/'+market+'.vendor.json',metadata='raw/'+market+'.metadata.json',
                ohlc='canonical/'+market+'.csv',returns='returns/'+market+'.npz')
            reports[market]=report
            if report['eligible']:
                eligible.append(market);masks.append(prepared.loc[prepared.is_primary,['market','date']])
            print(market,json.dumps(report),flush=True)
        except Exception as exc:
            reports[market]=dict(market=market,eligible=False,failure_codes=['vendor_acquisition_or_format_failure'],error=repr(exc))
            print(market,'UNAVAILABLE',repr(exc),flush=True)
        save(out/'qc.json',reports)
    mask=pd.concat(masks,ignore_index=True) if masks else pd.DataFrame(columns=['market','date'])
    mask.to_csv(out/'scoring_mask.csv',index=False)
    manifest=dict(kind='programme_closure_index_data_lock_v1',created_utc=now(),candidate_population=mapping['candidate_population'],
        data_gate=D.specification(),locked_before_model_scoring=True,
        exposure_disclosure='Known whole-historyYahooaudits previously touchedSPX,FTSE,GDAXI,N225. Historical foundation-model results and earlier research are already inspected. A one-day metadata inventory preceded this acquisition. This request acquired all vendor-provided dailyOHLC from2010-01-01 through2026-08-31 for resolvedfixedindexidentities, before any corrected-model training or scoring. No claim of globally untouched data or independent foundation-model pretraining exclusion.',
        mapping_file='mapping.json',qc_file='qc.json',eligible_population=eligible,scoring_mask='scoring_mask.csv',market_files=market_files,
        primary_available=len(eligible)>=20,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out/'manifest.json',manifest);(out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n')
    print('ELIGIBLE',len(eligible),eligible,'ROWS',len(mask),flush=True)
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['mapping','acquire']);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();(seal_mapping if a.action=='mapping' else acquire)(a.source,a.out)
