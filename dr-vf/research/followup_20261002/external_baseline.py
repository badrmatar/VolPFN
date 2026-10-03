"""Registered rolling GJR-t, using a copied and previously tested procedure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'external_confirmation_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    execution=json.loads((OUT/'execution_protocol.json').read_text())
    if sha(__file__)!=execution['baseline_sha256']:raise ValueError('Baseline source drift')
    design=json.loads((OUT/'protocol.json').read_text());receipt=json.loads((OUT/'DATA_RECEIPT.json').read_text())
    if receipt['status']!='complete' or receipt['mask_sha256']!=sha(OUT/'mask.csv'):raise ValueError('Data/mask drift')
    if sha(OUT/'econometric.py')!=design['source_hashes']['econometric.py']:raise ValueError('Copied procedure drift')
    spec=importlib.util.spec_from_file_location('external_copied_econometric',OUT/'econometric.py')
    E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
    dest=OUT/'baseline';dest.mkdir(exist_ok=False)
    mask=pd.read_csv(OUT/'mask.csv',dtype={'date':str});rows=[];fits={};failures=[]
    for market in design['tickers']:
        path=OUT/'data'/(market+'.npz')
        if sha(path)!=receipt['files'][str(path.relative_to(OUT))]:raise ValueError('Data drift')
        with np.load(path,allow_pickle=False) as z:dates=z['date'].copy();r=z['returns'].copy()
        result=E.forecast_econometric({'r':r,'date':dates,'return_units':'percent'},'gjr_garch_t',design['evaluation_start'],history_window=1000)
        fits[market]={k:result[k] for k in ['status','metadata','attempts','fits','failures']}
        (dest/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
        if result['status']!='complete':failures.append(market);continue
        indices=mask.loc[mask.market==market,'row_index'].to_numpy(int)
        for i in indices:
            h=float(result['second_moment'][i]);mu=float(result['mean'][i]);v=float(result['variance'][i]);nu=float(result['nu'][i])
            if not np.isfinite([h,mu,v,nu]).all() or h<=0 or v<=0 or nu<=2:raise ValueError('Invalid native GJR moments')
            fit_index=int(np.searchsorted(dates,result['fit_origin'][i]))
            if fit_index>i or fit_index<1000 or dates[fit_index-1]>=dates[i]:raise ValueError('GJR refit not causal')
            rows.append(dict(model='gjr_garch_t_rolling1000',seed='none',market=market,date=str(dates[i]),row_index=int(i),
                context_end=str(dates[i-1]),fit_origin=str(result['fit_origin'][i]),latest_fit_input_date=str(dates[fit_index-1]),return_value=float(r[i]),mean=mu,
                variance=v,second_moment=h,nu=nu,qlike=float(r[i]**2/h+np.log(h))))
        print('BASELINE',market,len(indices),'failed_refits',len(result['failures']),flush=True)
    f=pd.DataFrame(rows).rename(columns={'return_value':'return'})
    if failures or len(f)!=len(mask):
        (dest/'status.json').write_text(json.dumps(dict(status='unavailable',failed_markets=failures,rows=len(f),expected_rows=len(mask)),indent=2)+'\n')
        raise RuntimeError('Baseline unavailable on complete fixed population; no silent row intersection')
    f.to_csv(dest/'forecasts.csv',index=False,float_format='%.17g')
    (dest/'status.json').write_text(json.dumps(dict(status='complete',rows=len(f),forecast_sha256=sha(dest/'forecasts.csv'),
        source_sha256=sha(OUT/'econometric.py'),execution_protocol_sha256=sha(OUT/'execution_protocol.json')),indent=2)+'\n')


if __name__=='__main__':main()
