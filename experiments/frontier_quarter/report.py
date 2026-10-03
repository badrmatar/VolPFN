"""Cumulative quarter report with immutable January reuse."""
import json
from pathlib import Path
import numpy as np
from experiments.frontier_benchmark import metrics
from experiments.runpod_transfer.worker import sha, save
from .diagnose import load_cell


def combine(january, increment):
    keys=lambda rows:[(r['market'],r['date']) for r in rows]
    a,b=keys(january),keys(increment)
    if len(set(a+b))!=len(a+b):raise ValueError('Duplicate origin across increments')
    if any(not '2015-01-01'<=r['date']<='2015-01-31' for r in january):raise ValueError('January boundary')
    if any(not '2015-02-01'<=r['date']<='2015-03-31' for r in increment):raise ValueError('Increment boundary')
    return sorted(january+increment,key=lambda r:(r['date'],r['market']))


def report(root):
    root=Path(root);study=json.loads((root/'study.json').read_text())
    for p,h in study['reused_january'].items():
        if sha(p)!=h:raise ValueError('January forecast changed')
    january=root.parent/'frontier_one_month_201501_v1'
    results={};pending={}
    for name in study['order']:
        if not (root/'models'/name/'status.json').exists():pending[name]='not_started';continue
        st=json.loads((root/'models'/name/'status.json').read_text())
        if st['status']!='complete':pending[name]=st['status'];continue
        inc=load_cell(root,name);old=load_cell(january,name);both=combine(old,inc)
        if len(inc)!=1152 or len(both)!=1730:raise ValueError('Incomplete sample')
        results[name]={}
        for label,rr in [('February–March',inc),('January–March',both)]:
            per={}
            for m in study['markets']:
                subset=[r for r in rr if r['market']==m]
                if not subset:raise ValueError('Missing market')
                y=np.array([r['target'] for r in subset]);h=np.array([r['h'] for r in subset]);q=np.array([r['raw_quantiles'] for r in subset])
                per[m]=dict(n=len(subset),pinball=float(metrics.pinball(y,q).mean()),
                            qlike_native=float(metrics.qlike(y*y,h).mean()),
                            qlike_fixed_scale=float(metrics.qlike(y*y,study['scales'][name]*h).mean()))
            results[name][label]=dict(per_market=per,mean={k:float(np.mean([v[k] for v in per.values()])) for k in ('pinball','qlike_native','qlike_fixed_scale')})
    out=dict(status='partial' if pending else 'complete',pending=pending,models=results,inferential_claims=False)
    save(root/'comparison.json',out)
    lines=['# First-quarter 2015 exploratory comparison','',f"Status: {out['status']}",'',
           'January is reused unchanged. All 28 markets and extreme days remain included.',
           'One VolPFN seed; fixed three-market calibration; differing mean/median semantics.',
           'Historical exposed data; no significance, native-density or prior-causality claim.','',
           '| Period | Model | Pinball | Native QLIKE | Fixed-scale QLIKE |','|---|---|---:|---:|---:|']
    for name,cells in results.items():
        for label,cell in cells.items():
            v=cell['mean'];lines.append(f"| {label} | {name} | {v['pinball']:.6f} | {v['qlike_native']:.6f} | {v['qlike_fixed_scale']:.6f} |")
    if pending:lines+=['', 'Incomplete models: '+json.dumps(pending)]
    (root/'READOUT.md').write_text('\n'.join(lines)+'\n')
    save(root/'report_hashes.json',{f:sha(root/f) for f in ('comparison.json','READOUT.md')})
    return out


if __name__=='__main__':
    import sys
    print(json.dumps(report(sys.argv[1]),indent=2))
