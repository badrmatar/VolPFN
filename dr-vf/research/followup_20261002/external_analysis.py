"""Exact fixed ETF population, all five seeds, three registered QLIKE contrasts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln
from ttm_control import counts, holm, interval

HERE=Path(__file__).resolve().parent
OUT=HERE/'external_confirmation_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (OUT/'statistics.json').exists():raise FileExistsError('Completed external analysis preserved')
    design=json.loads((OUT/'protocol.json').read_text());execution=json.loads((OUT/'execution_protocol.json').read_text())
    receipt=json.loads((OUT/'DATA_RECEIPT.json').read_text());mask=pd.read_csv(OUT/'mask.csv',dtype={'date':str})
    if receipt['status']!='complete' or sha(OUT/'mask.csv')!=receipt['mask_sha256']:raise ValueError('Fixed data changed')
    if sorted(mask.market.unique())!=sorted(design['tickers']):raise ValueError('All 20 declared ETFs required')
    if mask.duplicated(['market','date']).any():raise ValueError('Duplicate target mask')
    index=pd.MultiIndex.from_frame(mask[['market','date']]);frames={};statuses=[]
    for seed in design['seeds']:
        run=OUT/'neural_runs'/str(seed);s=json.loads((run/'status.json').read_text())
        if s['status']!='complete' or s['protocol_sha256']!=sha(OUT/'protocol.json') or s['execution_protocol_sha256']!=sha(OUT/'execution_protocol.json'):
            raise ValueError('Incomplete or mismatched neural seed')
        for arm in ['full','noaux','historical_regularized']:
            path=run/(arm+'.csv')
            if sha(path)!=s['forecast_hashes'][path.name]:raise ValueError('Neural forecast changed')
            frames.setdefault(arm,[]).append(pd.read_csv(path,dtype={'date':str,'seed':str}))
        statuses.append(s)
    baseline=json.loads((OUT/'baseline/status.json').read_text())
    path=OUT/'baseline/forecasts.csv'
    if baseline['status']!='complete' or sha(path)!=baseline['forecast_sha256']:raise ValueError('Incomplete/changed baseline')
    frames['gjr_garch_t_rolling1000']=[pd.read_csv(path,dtype={'date':str,'seed':str})]
    loss={};seed_loss={};summaries=[];all_frames={}
    for arm,parts in frames.items():
        table=pd.concat(parts,ignore_index=True);expected=['none'] if arm=='gjr_garch_t_rolling1000' else [str(x) for x in design['seeds']]
        if sorted(table.seed.unique())!=expected:raise ValueError('Required seed roster missing')
        scores=[];aligned=[]
        for seed,g in table.groupby('seed',sort=True):
            g=g.set_index(['market','date'])
            if g.index.has_duplicates or len(g)!=len(index) or set(g.index)!=set(index):raise ValueError('Complete original ETF mask required')
            g=g.reindex(index).reset_index();r=g['return'].to_numpy(float);h=g.second_moment.to_numpy(float)
            np.testing.assert_allclose(r,mask['return'],rtol=1e-12,atol=1e-12)
            if not np.isfinite(h).all() or (h<=0).any():raise ValueError('Invalid issued M2')
            np.testing.assert_allclose(h,g.variance+g['mean']**2,rtol=1e-10,atol=1e-10)
            q=r*r/h+np.log(h);np.testing.assert_allclose(q,g.qlike,rtol=1e-11,atol=1e-11)
            if (g.context_end>=g.date).any():raise ValueError('Lookahead in neural context')
            if arm=='gjr_garch_t_rolling1000':
                if (g.latest_fit_input_date>=g.date).any():raise ValueError('Lookahead in GJR refit')
                nu=g.nu.to_numpy(float);v=g.variance.to_numpy(float);scale2=v*(nu-2)/nu;z=r-g['mean'].to_numpy(float)
                g['native_nll']=-(gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(nu*np.pi*scale2)-(nu+1)/2*np.log1p(z*z/(nu*scale2)))
            t8scale=h*.75
            g['common_t8_nll']=-(gammaln(4.5)-gammaln(4)-.5*np.log(8*np.pi*t8scale)-4.5*np.log1p(r*r/(8*t8scale)))
            if not np.isfinite(g[['qlike','native_nll','common_t8_nll']]).all().all():raise ValueError('Invalid scores')
            stat=g.groupby('market')[['qlike','native_nll','common_t8_nll']].mean().mean().to_dict()
            summaries.append(dict(model=arm,seed=seed,**stat));scores.append(q);aligned.append(g)
        seed_loss[arm]=np.stack(scores);loss[arm]=seed_loss[arm].mean(axis=0);all_frames[arm]=pd.concat(aligned,ignore_index=True)
    pd.DataFrame(summaries).to_csv(OUT/'per_seed.csv',index=False)
    leaderboard=pd.DataFrame(summaries).groupby('model')[['qlike','native_nll','common_t8_nll']].mean()
    leaderboard['qlike_rank']=leaderboard.qlike.rank(method='min');leaderboard.to_csv(OUT/'leaderboard.csv')
    dates=mask.date.to_numpy();markets=mask.market.to_numpy();calendar=sorted(set(dates));results=[]
    contrasts=[('full','gjr_garch_t_rolling1000'),('noaux','gjr_garch_t_rolling1000'),('noaux','historical_regularized')]
    for block in [60,20,120]:
        c=counts(len(calendar),20000,block,design['seed']);rows=[]
        for reference,comparator in contrasts:
            delta=loss[comparator]-loss[reference]
            matrix=pd.DataFrame({'date':dates,'market':markets,'gain':delta}).pivot(index='date',columns='market',values='gain').reindex(calendar).to_numpy()
            valid=np.isfinite(matrix).astype(float);denom=c@valid
            if (denom==0).any():raise ValueError('Bootstrap omitted an ETF')
            estimate=float(np.nanmean(matrix,axis=0).mean());draws=np.mean((c@np.nan_to_num(matrix))/denom,axis=1)
            raw=float((1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/20001)
            seed_gains=[]
            for j in range(5):
                comparison=seed_loss[comparator][j] if len(seed_loss[comparator])==5 else loss[comparator]
                gain=comparison-seed_loss[reference][j]
                seed_gains.append(float(pd.DataFrame({'market':markets,'gain':gain}).groupby('market').gain.mean().mean()))
            rows.append(dict(block=block,reference=reference,comparator=comparator,gain=estimate,interval95=interval(estimate,draws),
                family_interval95=interval(estimate,draws,1-.05/3),p_raw=raw,paired_seed_gains=seed_gains,
                markets_improved=int((np.nanmean(matrix,axis=0)>0).sum())))
        for row,adjusted in zip(rows,holm([x['p_raw'] for x in rows])):
            row['p_holm']=float(adjusted);row['supports_reference_advantage']=bool(row['gain']>0 and adjusted<.05 and row['family_interval95'][0]>0 and all(x>0 for x in row['paired_seed_gains']))
            row['supports_reference_disadvantage']=bool(row['gain']<0 and adjusted<.05 and row['family_interval95'][1]<0 and all(x<0 for x in row['paired_seed_gains']))
        results.extend(rows)
    stats=dict(classification=execution['classification'],family_slots=3,resamples=20000,blocks=[60,20,120],
        direction='Comparator loss minus reference loss; positive favours reference',contrasts=results,
        limitations='New fixed ETF population, not verified untouched human exposure or an independent calendar/shock holdout. Conditional saved-forecast inference excludes model training, online refits, source revisions, and research selection.')
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',fixed_20_ETFs=True,rows_per_model_seed=len(mask),
        all_five_neural_seeds=True,all_masks_exact=True,all_scores_recomputed=True,context_and_fit_cutoffs_verified=True,
        original_weights_unchanged=True,neural_seed_receipts=statuses),indent=2)+'\n')
    (OUT/'FINAL_RECEIPT.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['FINAL_RECEIPT.json'] and p.suffix not in ['.log','.db','.sqlite','.sqlite-wal','.sqlite-shm'] and 'vendor_cache' not in p.parts},indent=2,sort_keys=True)+'\n')
    print(leaderboard.to_string());print(json.dumps([r for r in results if r['block']==60],indent=2))


if __name__=='__main__':main()
