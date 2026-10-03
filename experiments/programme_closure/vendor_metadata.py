"""One-pass public vendor identity inventory; no forecasting or score access."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from curl_cffi import requests

CANDIDATES = {
 'AEX':('^AEX','AEX'), 'AORD':('^AORD','ALL ORDINARIES'),
 'BFX':('^BFX','BEL 20'), 'BSESN':('^BSESN','S&P BSE SENSEX'),
 'BVSP':('^BVSP','IBOVESPA'), 'DJI':('^DJI','Dow Jones Industrial Average'),
 'FCHI':('^FCHI','CAC 40'), 'FTSE':('^FTSE','FTSE 100'),
 'GDAXI':('^GDAXI','DAX'), 'GSPTSE':('^GSPTSE','S&P/TSX Composite'),
 'HSI':('^HSI','Hang Seng'), 'IBEX':('^IBEX','IBEX 35'),
 'IXIC':('^IXIC','NASDAQ Composite'), 'KS11':('^KS11','KOSPI Composite'),
 'KSE':('^KSE','Karachi 100'), 'MXX':('^MXX','IPC Mexico'),
 'N225':('^N225','Nikkei 225'), 'NSEI':('^NSEI','NIFTY 50'),
 'OMXC20':('^OMXC20','OMX Copenhagen 20'), 'OMXHPI':('^OMXHPI','OMX Helsinki all-share price'),
 'OMXSPI':('^OMXSPI','OMX Stockholm all-share price'), 'OSEAX':('OSEAX.OL','Oslo all-share'),
 'RUT':('^RUT','Russell 2000'), 'SMSI':('^SMSI','Madrid general index'),
 'SPX':('^GSPC','S&P 500'), 'SSEC':('000001.SS','Shanghai Composite'),
 'SSMI':('^SSMI','Swiss Market Index'), 'STOXX50E':('^STOXX50E','EURO STOXX 50'),
}


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    specification=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidates=CANDIDATES,
        purpose='Identity-only inventory before later-period OHLC acquisition; response may include one daily observation',
        no_fallback_symbols=True,metadata_is_not_ohlc_validity=True)
    (out/'requested.json').write_text(json.dumps(specification,indent=2)+'\n')
    rows=[]
    session=requests.Session(impersonate='chrome')
    for market,(symbol,expected) in CANDIDATES.items():
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')+'?range=1d&interval=1d'
        row=dict(market=market,symbol=symbol,expected_identity=expected,url=url,identity_verified=False)
        try:
            response=session.get(url,timeout=20)
            (out/(market+'.json')).write_bytes(response.content)
            row['http_status']=response.status_code
            response.raise_for_status()
            data=response.json()['chart']
            if data.get('error') or not data.get('result'): raise ValueError(str(data.get('error')))
            meta=data['result'][0]['meta']
            row['meta']={k:meta.get(k) for k in ('symbol','instrumentType','shortName','longName','exchangeName','fullExchangeName','exchangeTimezoneName','currency','firstTradeDate','dataGranularity','range')}
            row['status']='metadata_available_requires_identity_review'
        except Exception as exc:
            row.update(status='unavailable',error=type(exc).__name__+': '+str(exc))
        rows.append(row)
        (out/'inventory.json').write_text(json.dumps(dict(rows=rows,metadata_only=True),indent=2)+'\n')
        print(market,json.dumps(row),flush=True)
        if row.get('http_status')==429: break


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    main(p.parse_args().out)
