"""Fixed extension: comparison and synchronized date-block uncertainty."""
import json
from pathlib import Path
import numpy as np
from experiments.frontier_quarter.diagnose import load_cell
from experiments.frontier_quarter.report import combine as quarter_combine
from experiments.frontier_uncertainty.run import matrix_for, resample, losses
from experiments.population_transfer.statistics import _calendar_counts, _basic_interval, holm
from experiments.runpod_transfer.worker import sha, save

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/outputs/frontier_overnight_increment_201504_201606_v1'
COMPETITORS=('chronos2','moirai2','tabpfn_ts')
METRICS=('pinball','qlike_native','qlike_fixed_scale')


def combine(old,new):
    if any(not '2015-01-01'<=r['date']<='2015-03-31' for r in old):
        raise ValueError('Old-period boundary')
    if any(not '2015-04-01'<=r['date']<='2016-06-30' for r in new):
        raise ValueError('Extension boundary')
    keys=[(r['market'],r['date']) for r in old+new]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate origin')
    return sorted(old+new,key=lambda r:(r['date'],r['market']))


def verdict(estimate,p,interval):
    if p<=.05 and estimate>0 and interval[0]>0:return 'supported_superiority'
    if p<=.05 and estimate<0 and interval[1]<0:return 'supported_inferiority'
    return 'inconclusive'


def report(root=OUT):
    root=Path(root);spec=json.loads((root/'study.json').read_text())
    if sha(root/'study.json')!=(root/'study.sha256').read_text().strip():raise ValueError('Study drift')
    if sha(root/'PRELAUNCH_AUDIT.json')!=(root/'PRELAUNCH_AUDIT.sha256').read_text().strip():raise ValueError('Audit drift')
    extra=json.loads((root/'PRELAUNCH_AUDIT.json').read_text())['extra_sources']
    for p,h in extra.items():
        if sha(p)!=h:raise ValueError('Analysis dependency drift: '+p)
    for p,h in {**spec['hashes'],**spec['sources'],**spec['reused_january'],**spec['reused_quarter']}.items():
        if sha(p)!=h:raise ValueError('Frozen input drift: '+p)
    # No partial comparisons: every cell must validate first.
    extension={m:load_cell(root,m) for m in spec['order']}
    keys=lambda rr:[(r['date'],r['market'],r['target']) for r in rr]
    reference=extension['volpfn']
    if len(reference)!=8774 or any(keys(rr)!=keys(reference) for rr in extension.values()):
        raise ValueError('Incomplete or mismatched extension')
    old={m:quarter_combine(load_cell(root.parent/'frontier_one_month_201501_v1',m),
                          load_cell(root.parent/'frontier_quarter_increment_20150203_v1',m)) for m in spec['order']}
    cumulative={m:combine(old[m],extension[m]) for m in spec['order']}
    if any(len(rr)!=10504 for rr in cumulative.values()):raise ValueError('Cumulative count')
    dst=root/'analysis_v1';dst.mkdir(exist_ok=False)
    forecast_hashes={str(root/'models'/m/'predictions.json'):sha(root/'models'/m/'predictions.json') for m in spec['order']}
    save(dst/'inputs.json',dict(study_sha256=sha(root/'study.json'),forecasts=forecast_hashes))
    summaries={};extension_loss={};floor_events={}
    for label,all_rows in [('extension',extension),('cumulative_descriptive',cumulative)]:
        summaries[label]={}
        for model,rows in all_rows.items():
            model_loss=losses(rows,spec['scales'][model]);market=np.array([r['market'] for r in rows])
            per={m:{metric:float(values[market==m].mean()) for metric,values in model_loss.items()} for m in spec['markets']}
            summaries[label][model]=dict(per_market=per,mean={metric:float(np.mean([v[metric] for v in per.values()])) for metric in METRICS})
            if label=='extension':
                extension_loss[model]=model_loss
                floor_events[model]=[dict(market=r['market'],date=r['date'],target=r['target'],h=r['h']) for r in rows if r['floor_activated']]
    matrices=[];labels=[];dates=[r['date'] for r in reference];markets=[r['market'] for r in reference]
    for model in COMPETITORS:
        for metric in METRICS:
            mat,calendar,population=matrix_for(dates,markets,extension_loss[model][metric]-extension_loss['volpfn'][metric])
            matrices.append(mat);labels.append((model,metric))
    assert len(calendar)==327 and len(population)==28
    settings=spec['uncertainty'];reps=settings['resamples'];seed=settings['seed']
    results=[];all_draws={};all_counts={};audit_checks=0
    for block in (settings['main_block'],*settings['sensitivity_blocks']):
        counts=_calendar_counts(len(calendar),reps,block,seed)
        all_counts['block'+str(block)]=counts.astype(np.int16)
        group=[]
        for (model,metric),matrix in zip(labels,matrices):
            estimate,draws,market_gains=resample(matrix,counts)
            expected=summaries['extension'][model]['mean'][metric]-summaries['extension']['volpfn']['mean'][metric]
            np.testing.assert_allclose(estimate,expected,rtol=1e-10,atol=1e-12)
            for i in range(12):
                direct=[]
                for j in range(matrix.shape[1]):
                    ids=[t for t in range(len(calendar)) if np.isfinite(matrix[t,j])]
                    direct.append(sum(float(counts[i,t])*float(matrix[t,j]) for t in ids)/sum(float(counts[i,t]) for t in ids))
                np.testing.assert_allclose(draws[i],sum(direct)/len(direct),rtol=1e-10,atol=1e-12);audit_checks+=1
            p=(1+int(np.count_nonzero(abs(draws-estimate)>=abs(estimate))))/(reps+1)
            group.append(dict(model=model,metric=metric,block=block,gain=estimate,
                              interval95=_basic_interval(estimate,draws,.95),
                              family_interval95=_basic_interval(estimate,draws,1-.05/9),
                              p_raw=p,markets_improved=int(np.sum(market_gains>0)),
                              per_market_gain=dict(zip(population,map(float,market_gains)))))
            all_draws[f'{model}__{metric}__block{block}']=draws
        for r,p in zip(group,holm([r['p_raw'] for r in group])):
            r['p_holm9']=float(p);r['verdict']=verdict(r['gain'],p,r['family_interval95'])
        order=sorted(range(9),key=lambda i:group[i]['p_raw']);running=0.
        for rank,i in enumerate(order):
            running=max(running,(9-rank)*group[i]['p_raw'])
            assert abs(group[i]['p_holm9']-min(1.,running))<1e-12;audit_checks+=1
        results.extend(group)
    for r,p in zip(results,holm([r['p_raw'] for r in results])):r['p_holm27']=float(p)
    for p,h in forecast_hashes.items():
        if sha(p)!=h:raise ValueError('Forecast changed during analysis')
    np.savez_compressed(dst/'draws.npz',**all_draws);np.savez_compressed(dst/'counts.npz',**all_counts)
    main=next(r for r in results if r['block']==10 and r['model']=='tabpfn_ts' and r['metric']=='qlike_fixed_scale')
    save(dst/'results.json',dict(status='complete',historical_followup=True,primary=main,contrasts=results,
                               summaries=summaries,floor_events=floor_events,rows=8774,calendar_dates=327,
                               cumulative_rows=10504,scope='Conditional fixed-checkpoint historical comparison; no prior-causal or untouched-data claim'))
    save(dst/'AUDIT.json',dict(passed=True,independent_checks=audit_checks,hashes_unchanged=True))
    lines=['# Fixed overnight follow-up','',f"Primary verdict: **{main['verdict']}** (VolPFN versus TabPFN-TS, fixed-scale QLIKE).",'',
           'April 2015–June 2016: 8,774 rows, 28 markets, 327 dates. Positive gains favour VolPFN.',
           'Historical follow-up selected after viewing Q1. One checkpoint; fixed three-market scales;',
           'different output semantics and unknown foundation-model pretraining overlap. No causal prior claim.','',
           '| Competitor | Metric | Gain | Pointwise 95% CI | Family 95% CI | Holm p | Verdict |',
           '|---|---|---:|---|---|---:|---|']
    ci=lambda x:f'[{x[0]:.6g}, {x[1]:.6g}]'
    for r in results[:9]:lines.append(f"| {r['model']} | {r['metric']} | {r['gain']:.6g} | {ci(r['interval95'])} | {ci(r['family_interval95'])} | {r['p_holm9']:.6g} | {r['verdict']} |")
    lines+=['','Main block length 10. Ordinary intervals are pointwise; family intervals use Bonferroni.',
            'Inconclusive does not mean equivalence. Moirai QLIKE must be read alongside saved floor events.','',
            '| Competitor | Metric | Block | Pointwise 95% CI | Holm-nine p | Holm-27 p |','|---|---|---:|---|---:|---:|']
    for r in results:lines.append(f"| {r['model']} | {r['metric']} | {r['block']} | {ci(r['interval95'])} | {r['p_holm9']:.6g} | {r['p_holm27']:.6g} |")
    lines+=['','## Score tables (cumulative period is descriptive)','',
            '| Period | Model | Pinball | Native QLIKE | Fixed-scale QLIKE |','|---|---|---:|---:|---:|']
    for label,group in summaries.items():
        for m,v in group.items():
            x=v['mean'];lines.append(f"| {label} | {m} | {x['pinball']:.6g} | {x['qlike_native']:.6g} | {x['qlike_fixed_scale']:.6g} |")
    (dst/'READOUT.md').write_text('\n'.join(lines)+'\n')
    save(dst/'hashes.json',{p.name:sha(p) for p in dst.iterdir() if p.is_file()})
    return main


if __name__=='__main__':print(json.dumps(report(),indent=2))
