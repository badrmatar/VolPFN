"""All frozen rows required; no ranking on incomplete model cells."""
import json
from pathlib import Path
import numpy as np
from experiments.runpod_transfer.worker import sha,save
from experiments.frontier_benchmark import metrics


def validate(rows,origins):
 if len(rows)!=len(origins):raise ValueError('Incomplete rows')
 for r,o in zip(rows,origins):
  if any(r[k]!=o[k] for k in ('market','date','index','phase','target','context_start','context_end','context_count')):raise ValueError('Misaligned row')
  q=np.asarray(r['raw_quantiles'])
  if q.shape!=(9,) or not np.isfinite(q).all() or np.any(np.diff(q)<0):raise ValueError('Invalid quantiles')
  if not np.isfinite(r['h']) or r['h']<=0:raise ValueError('Invalid variance point')
 return [r for r in rows if r['phase']=='evaluation']


def report(out):
 out=Path(out);s=json.loads((out/'study.json').read_text());m=json.loads((out/'bundle/manifest.json').read_text())
 if sha(out/'study.json')!=(out/'study.sha256').read_text().strip() or sha(out/'bundle/manifest.json')!=s['bundle_manifest_sha256']:raise RuntimeError('Study or bundle manifest drift')
 results={};unavailable={};inputs={}
 for name in s['order']:
  cell=out/'models'/name;status=cell/'status.json'
  if not status.exists():unavailable[name]='not_started';continue
  st=json.loads(status.read_text())
  if st['status']!='complete':unavailable[name]=st['status'];continue
  p=cell/'predictions.json'
  if sha(p)!=st['forecast_sha256']:raise ValueError('Prediction hash mismatch')
  if name=='tabpfn_ts' and st.get('bridge_passed') is not True:raise ValueError('Bridge missing')
  rows=validate(json.loads(p.read_text()),m['origins']);inputs[str(p)]=sha(p)
  market={}
  for k in s['markets']:
   rr=[r for r in rows if r['market']==k]
   if not rr:raise ValueError('Missing market')
   y=np.array([r['target'] for r in rr]);h=np.array([r['h'] for r in rr]);q=np.array([r['raw_quantiles'] for r in rr])
   market[k]=dict(rows=len(rr),pinball=float(metrics.pinball(y,q).mean()),qlike_native=float(metrics.qlike(y*y,h).mean()),qlike_fixed_scale=float(metrics.qlike(y*y,s['scales'][name]*h).mean()))
  results[name]=dict(scale=s['scales'][name],per_market=market,mean={key:float(np.mean([v[key] for v in market.values()])) for key in ('pinball','qlike_native','qlike_fixed_scale')})
 result=dict(status='complete' if not unavailable else 'partial',evaluation_rows=s['evaluation_count'],models=results,unavailable=unavailable,inputs=inputs,calibration=s['calibration'],inferential_claims=False)
 save(out/'comparison.json',result)
 lines=['# January 2015: 28-market development screen','','Status: '+result['status']+'. '+str(s['evaluation_count'])+' evaluation origins; the three engineering bridge forecasts are excluded.','',
 'Previously inspected historical data. One fixed-final VolPFN seed; no untouched-data, significance or native-density superiority claim. Existing 2012–2013 three-market calibration multipliers are transferred unchanged to this panel. They are not full-panel calibration.','',
 '| Model | Pinball | Native-point QLIKE | Fixed-scale QLIKE |','|---|---:|---:|---:|']
 for name in s['order']:
  if name in results:
   v=results[name]['mean'];lines.append(f"| {name} | {v['pinball']:.6f} | {v['qlike_native']:.6f} | {v['qlike_fixed_scale']:.6f} |")
  else:lines.append(f'| {name} ({unavailable[name]}) | — | — | — |')
 lines+=['','Lower is better. Equal weights across markets, with means calculated within markets first. Quantile scores compare native return quantiles. Squared-return mean/median semantics differ; a global scale does not remove that distinction.','', '| Market | Model | Pinball | Native QLIKE | Fixed-scale QLIKE |','|---|---|---:|---:|---:|']
 for k in s['markets']:
  for name,v in results.items():
   x=v['per_market'][k];lines.append(f"| {k} | {name} | {x['pinball']:.6f} | {x['qlike_native']:.6f} | {x['qlike_fixed_scale']:.6f} |")
 (out/'READOUT.md').write_text('\n'.join(lines)+'\n');save(out/'report_hashes.json',{f:sha(out/f) for f in ('comparison.json','READOUT.md')})
 return result

if __name__=='__main__':
 import sys;print(json.dumps(report(sys.argv[1]),indent=2))
