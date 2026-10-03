"""Post-hoc calendar-block uncertainty; frozen forecasts only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.frontier_benchmark import metrics
from research.frontier_quarter.diagnose import load_cell
from research.frontier_quarter.report import combine
from research.population_transfer.statistics import _calendar_counts, _basic_interval, holm
from research.runpod_transfer.worker import sha, save

ROOT=Path(__file__).resolve().parents[2]
QUARTER=ROOT/'research/outputs/frontier_quarter_increment_20150203_v1'
JANUARY=QUARTER.parent/'frontier_one_month_201501_v1'
OUT=QUARTER.parent/'frontier_quarter_uncertainty_v1'
MODELS=('chronos2','moirai2','tabpfn_ts')
METRICS=('pinball','qlike_native','qlike_fixed_scale')
REPS=20000
SEED=2026091901


def matrix_for(dates,markets,gains):
    f=pd.DataFrame(dict(date=dates,market=markets,gain=gains))
    if f.empty or f[['date','market']].isna().any().any() or not np.isfinite(f.gain).all():
        raise ValueError('Invalid paired rows')
    if f.duplicated(['date','market']).any():raise ValueError('Duplicate origin')
    pivot=f.pivot(index='date',columns='market',values='gain').sort_index()
    return pivot.to_numpy(float),list(pivot.index),list(pivot.columns)


def resample(matrix,counts):
    valid=np.isfinite(matrix).astype(float)
    denom=counts@valid
    if np.any(denom==0):raise ValueError('Missing market in bootstrap draw')
    means=np.nanmean(matrix,axis=0)
    draws=np.mean((counts@np.nan_to_num(matrix,nan=0.))/denom,axis=1)
    return float(means.mean()),draws,means


def losses(rows,scale):
    y=np.array([r['target'] for r in rows]);h=np.array([r['h'] for r in rows])
    q=np.array([r['raw_quantiles'] for r in rows])
    return dict(pinball=metrics.pinball(y,q),qlike_native=metrics.qlike(y*y,h),
                qlike_fixed_scale=metrics.qlike(y*y,scale*h))


def main():
    spec=json.loads((QUARTER/'study.json').read_text())
    point_report=json.loads((QUARTER/'comparison.json').read_text())
    if point_report['status']!='complete':raise ValueError('Incomplete comparison')
    for p,h in {**spec['hashes'],**spec['sources'],**spec['reused_january']}.items():
        if sha(p)!=h:raise ValueError('Input/source drift: '+p)
    paths=[p for root in (JANUARY,QUARTER) for p in root.glob('models/*/*.json')]
    paths += [QUARTER/'study.json',QUARTER/'comparison.json']
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md')]
    sources += [ROOT/p for p in ('research/population_transfer/statistics.py',
                'research/frontier_quarter/diagnose.py','research/frontier_quarter/report.py',
                'research/frontier_staged/report.py','research/frontier_benchmark/metrics.py')]
    OUT.mkdir(exist_ok=False)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='frozen_before_resampling_after_point_estimates',main_block=10,
                  sensitivity_blocks=[5,20],reps=REPS,seed=SEED,
                  contrasts=[dict(model=m,metric=k) for m in MODELS for k in METRICS],
                  inferential_scope='exploratory conditional date uncertainty only',
                  files={str(p):sha(p) for p in paths+sources})
    save(OUT/'manifest.json',manifest);(OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')
    rows={name:combine(load_cell(JANUARY,name),load_cell(QUARTER,name)) for name in ('volpfn',)+MODELS}
    ref=rows['volpfn'];keys=lambda rr:[(r['market'],r['date'],r['target']) for r in rr]
    if len(ref)!=1730 or any(keys(rr)!=keys(ref) for rr in rows.values()):raise ValueError('Mismatched sample')
    dates=[r['date'] for r in ref];markets=[r['market'] for r in ref]
    loss={name:losses(rr,spec['scales'][name]) for name,rr in rows.items()}
    matrices=[];labels=[];gain_columns={}
    for model in MODELS:
        for metric in METRICS:
            gain=loss[model][metric]-loss['volpfn'][metric]
            matrix,calendar,population=matrix_for(dates,markets,gain)
            matrices.append(matrix);labels.append((model,metric));gain_columns[model+'__'+metric]=gain
            expected=point_report['models'][model]['January–March']['mean'][metric]-point_report['models']['volpfn']['January–March']['mean'][metric]
            np.testing.assert_allclose(np.nanmean(matrix,axis=0).mean(),expected,rtol=1e-11,atol=1e-12)
    pd.DataFrame(dict(date=dates,market=markets,**gain_columns)).to_csv(OUT/'paired_gains.csv',index=False)
    output=[];draw_archive={};count_archive={};audit_checks=0
    for block in (10,5,20):
        counts=_calendar_counts(len(calendar),REPS,block,SEED)
        count_archive['block_'+str(block)]=counts.astype(np.int16)
        current=[]
        for (model,metric),matrix in zip(labels,matrices):
            estimate,draws,means=resample(matrix,counts)
            # Independent direct weighted calculation on the first 12 draws.
            for i in range(12):
                direct=[]
                for j in range(matrix.shape[1]):
                    ix=[t for t in range(len(calendar)) if np.isfinite(matrix[t,j])]
                    direct.append(sum(float(counts[i,t])*float(matrix[t,j]) for t in ix)/sum(float(counts[i,t]) for t in ix))
                np.testing.assert_allclose(draws[i],sum(direct)/len(direct),rtol=1e-11,atol=1e-12);audit_checks+=1
            error=draws-estimate
            p=(1+int(np.count_nonzero(np.abs(error)>=abs(estimate))))/(REPS+1)
            ordinary=_basic_interval(estimate,draws,.95)
            family=_basic_interval(estimate,draws,1-.05/9)
            current.append(dict(model=model,metric=metric,block=block,mean_gain=estimate,
                          interval95_pointwise=ordinary,interval95_family_bonferroni=family,
                          p_two_sided=p,markets_improved=int(np.sum(means>0)),
                          per_market_mean_gain=dict(zip(population,map(float,means))),
                          ordinary_interval_positive=ordinary[0]>0,family_interval_positive=family[0]>0,
                          interpretation_warning='Four floored variance forecasts dominate; not reliable evidence of general forecasting superiority' if model=='moirai2' and metric!='pinball' else None))
            draw_archive[f'{model}__{metric}__block{block}']=draws
        adjusted=holm([r['p_two_sided'] for r in current])
        for r,p in zip(current,adjusted):
            r['p_holm9']=float(p);r['exploratory_positive_holm9']=bool(r['mean_gain']>0 and p<=.05)
        output+=current
    for r,p in zip(output,holm([r['p_two_sided'] for r in output])):r['p_holm27_all_blocks']=float(p)
    # Independent Holm reconstruction, separate from imported implementation.
    for block in (10,5,20):
        group=[r for r in output if r['block']==block];ordered=sorted(range(9),key=lambda i:group[i]['p_two_sided']);running=0
        for rank,i in enumerate(ordered):
            running=max(running,(9-rank)*group[i]['p_two_sided'])
            assert abs(group[i]['p_holm9']-min(1.,running))<1e-12;audit_checks+=1
    np.savez_compressed(OUT/'bootstrap_draws.npz',**draw_archive)
    np.savez_compressed(OUT/'bootstrap_date_counts.npz',**count_archive)
    for p,h in manifest['files'].items():
        if sha(p)!=h:raise ValueError('Input changed during analysis')
    result=dict(status='complete',exploratory=True,rows=1730,markets=28,calendar_dates=len(calendar),
                main_block=10,bootstrap_reps=REPS,seed=SEED,contrasts=output,
                scope='Conditional on fixed forecasts, calibration and market selection; excludes training and research-selection uncertainty',
                caveat='Only 65 dates. Bootstrap validity requires suitable time-series regularity; approximate intervals/p-values are not statistical proof or fresh confirmation.')
    save(OUT/'results.json',result)
    save(OUT/'AUDIT.json',dict(passed=True,independent_draw_and_holm_checks=audit_checks,
                             point_estimates_match_prior_report=True,all_input_hashes_unchanged=True))
    lines=['# First-quarter uncertainty: exploratory analysis','',
           '1,730 market-days; 28 markets; **65 distinct calendar dates**. Positive gains favour VolPFN.',
           'Specified after inspecting outcomes. These are approximate conditional bootstrap results, not confirmation.',
           '20,000 synchronized circular date-block draws. Main block = 10 observed dates; sensitivity = 5 and 20.',
           'Calibration, models, checkpoint and population are fixed. Training and selection uncertainty are excluded.', '',
           '## Main setting: 10-date blocks','',
           '| Competitor | Metric | Gain | Ordinary 95% CI | Holm p (9) | Family 95% CI (Bonferroni) | Markets improved |',
           '|---|---|---:|---|---:|---|---:|']
    interval=lambda x:f'[{x[0]:.6g}, {x[1]:.6g}]'
    for r in output[:9]:
        lines.append(f"| {r['model']} | {r['metric']} | {r['mean_gain']:.6g} | {interval(r['interval95_pointwise'])} | {r['p_holm9']:.4f} | {interval(r['interval95_family_bonferroni'])} | {r['markets_improved']}/28 |")
    lines+=['', 'Ordinary intervals are pointwise, not corrected for nine comparisons. Family intervals',
            'use a Bonferroni adjustment; their coverage is approximate and depends on bootstrap validity.',
            'Do not infer equivalence from an interval containing zero. No useful-effect threshold was declared.', '',
            '## Block-length sensitivity','',
            '| Competitor | Metric | Block | Ordinary 95% CI | Holm p (9 within block) | Holm p (all 27) |',
            '|---|---|---:|---|---:|---:|']
    for r in output:
        lines.append(f"| {r['model']} | {r['metric']} | {r['block']} | {interval(r['interval95_pointwise'])} | {r['p_holm9']:.4f} | {r['p_holm27_all_blocks']:.4f} |")
    lines+=['', '## Limits','',
            '- The effective time span is one quarter; 10-date blocks leave only about 6.5 blocks.',
            '- Shared shocks are resampled jointly. Markets are not independent replications.',
            '- Circular resampling and approximate time-series regularity can be questionable around structural shocks.',
            '- Moirai QLIKE is dominated by four floored negative median predictions; retain the rows but do not treat a nominal result as general model superiority.',
            '- No forecast was refitted or excluded. This does not identify the effect of the volatility prior.',
            '- An untouched, longer-period comparison and seed replication remain separate requirements.']
    (OUT/'READOUT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'output_hashes.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='output_hashes.json'})
    print(json.dumps(dict(status='complete',output=str(OUT),main=[{k:r[k] for k in ('model','metric','mean_gain','interval95_pointwise','p_holm9','p_holm27_all_blocks','markets_improved')} for r in output[:9]]),indent=2))


if __name__=='__main__':main()
