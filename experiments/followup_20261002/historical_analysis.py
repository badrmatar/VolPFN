"""Verify all five historical seeds before comparisons with immutable old results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln
from ttm_control import counts, holm, interval

HERE=Path(__file__).resolve().parent
OUT=HERE/'historical_analysis_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=json.loads((OUT/'protocol.json').read_text())
    if (OUT/'statistics.json').exists():raise FileExistsError('Completed analysis preserved')
    design=json.loads((HERE/'protocol.json').read_text())
    mask=pd.read_csv(HERE/'evaluation_mask.csv',dtype={'date':str})
    index=pd.MultiIndex.from_frame(mask[['market','date']]);models={};receipts=[]
    for arm,entry in p['inputs'].items():
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Previous forecast changed')
        models[arm]=pd.read_csv(entry['path'],dtype={'date':str})
    frames=[]
    for seed in design['seeds']:
        run=HERE/'runs'/str(seed);s=json.loads((run/'status.json').read_text())
        if s['status']!='complete' or s['smoke_unpromotable'] or s['protocol_sha256']!=sha(HERE/'protocol.json'):
            raise ValueError('Incomplete or wrong-protocol seed')
        for name,key in [('best.pt','checkpoint_sha256'),('forecasts.csv','forecast_sha256')]:
            if sha(run/name)!=s[key]:raise ValueError('Seed artifact changed')
        dev=pd.read_json(run/'development.jsonl',lines=True)
        first=dev.sort_values('step').loc[lambda f:f.qlike==f.qlike.min()].iloc[0]
        if int(first.step)!=s['best_step']:raise ValueError('Not first strict development minimum')
        if not np.isclose(first.qlike,s['development_qlike'],atol=1e-12,rtol=1e-12):raise ValueError('Development receipt mismatch')
        f=pd.read_csv(run/'forecasts.csv',dtype={'date':str});frames.append(f);receipts.append(s)
    models['historical_regularized']=pd.concat(frames,ignore_index=True)
    loss={};seed_loss={};summary=[];checks=[]
    for arm,f in models.items():
        if sorted(f.seed.unique())!=design['seeds']:raise ValueError('Five seeds required')
        aligned=[];scores=[]
        for seed,g in f.groupby('seed',sort=True):
            g=g.set_index(['market','date'])
            if g.index.has_duplicates or len(g)!=len(mask) or set(g.index)!=set(index):raise ValueError('Exact original mask required')
            g=g.reindex(index).reset_index();h=g.second_moment.to_numpy();r=g['return'].to_numpy()
            if not np.isfinite(h).all() or (h<=0).any():raise ValueError('Invalid forecast')
            np.testing.assert_allclose(r,mask['return'],rtol=1e-12,atol=1e-12)
            q=r*r/h+np.log(h);np.testing.assert_allclose(q,g.qlike,rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(g.variance+g['mean']**2,h,rtol=1e-10,atol=1e-10)
            scale2=h*.75
            g['common_t8_nll']=-(gammaln(4.5)-gammaln(4)-.5*np.log(8*np.pi*scale2)-4.5*np.log1p(r*r/(8*scale2)))
            scores.append(q);aligned.append(g)
            stat=g.groupby('market')[['qlike','native_nll','common_t8_nll']].mean().mean().to_dict()
            summary.append(dict(model=arm,seed=int(seed),**stat))
            checks.append(dict(model=arm,seed=int(seed),rows=len(g),qlike_over_100=int((q>100).sum()),
                               maximum_qlike=float(q.max()),maximum_second_moment=float(h.max())))
        seed_loss[arm]=np.stack(scores);loss[arm]=seed_loss[arm].mean(axis=0)
        models[arm]=pd.concat(aligned,ignore_index=True)
    pd.DataFrame(summary).to_csv(OUT/'per_seed.csv',index=False)
    pd.DataFrame(summary).groupby('model')[['qlike','native_nll','common_t8_nll']].mean().to_csv(OUT/'leaderboard.csv')
    pd.DataFrame(checks).to_csv(OUT/'tail_diagnostics.csv',index=False)
    dates=mask.date.to_numpy();markets=mask.market.to_numpy();calendar=sorted(set(dates));results=[]
    for block in p['blocks']:
        c=counts(len(calendar),p['resamples'],block,p['seed']);rows=[]
        for comparator in ['noaux','full']:
            delta=loss['historical_regularized']-loss[comparator]
            matrix=pd.DataFrame({'date':dates,'market':markets,'gain':delta}).pivot(index='date',columns='market',values='gain').reindex(calendar).to_numpy()
            valid=np.isfinite(matrix).astype(float);denom=c@valid
            if (denom==0).any():raise ValueError('A bootstrap draw omitted a market')
            estimate=float(np.nanmean(matrix,axis=0).mean());draws=np.mean((c@np.nan_to_num(matrix))/denom,axis=1)
            raw=float((1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/(p['resamples']+1))
            seed_gains=[]
            for j in range(5):
                gain=seed_loss['historical_regularized'][j]-seed_loss[comparator][j]
                seed_gains.append(float(pd.DataFrame({'market':markets,'gain':gain}).groupby('market').gain.mean().mean()))
            rows.append(dict(block=block,synthetic=comparator,gain=estimate,interval95=interval(estimate,draws),
                family_interval95=interval(estimate,draws,1-.05/2),p_raw=raw,paired_seed_gains=seed_gains,
                markets_improved_by_synthetic=int((np.nanmean(matrix,axis=0)>0).sum())))
        for row,adjusted in zip(rows,holm([x['p_raw'] for x in rows])):
            row['p_holm']=float(adjusted)
            row['supports_synthetic_advantage']=bool(row['gain']>0 and adjusted<.05 and row['family_interval95'][0]>0 and all(x>0 for x in row['paired_seed_gains']))
        results.extend(rows)
    (OUT/'statistics.json').write_text(json.dumps(dict(scope=p['scope'],direction=p['direction'],family_slots=2,contrasts=results),indent=2)+'\n')
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',all_original_masks_exact=True,
        all_five_seeds=True,all_checkpoints_development_selected=True,all_hashes_matched=True,original_results_read_only=True,
        new_run_receipts=receipts),indent=2)+'\n')
    (OUT/'FINAL_RECEIPT.json').write_text(json.dumps({str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ['FINAL_RECEIPT.json','run.log']},indent=2,sort_keys=True)+'\n')
    print(pd.DataFrame(summary).groupby('model')[['qlike','native_nll','common_t8_nll']].mean().to_string())
    print(json.dumps([r for r in results if r['block']==60],indent=2))


if __name__=='__main__':main()
