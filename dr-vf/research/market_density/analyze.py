"""Date-block inference preserving contemporaneous market dependence."""
import json
from pathlib import Path
import numpy as np
from research.market_density.models import METHODS, HORIZONS, PROBS, score
from research.market_density.run import OUTPUT

METRICS=('pinball','coverage95','lower01','upper01','width95')


def block_weights(length,block,replicates=2000,seed=108001):
    rng=np.random.default_rng(seed+block)
    weights=np.zeros((replicates,length),dtype=float)
    for j in range(replicates):
        starts=rng.integers(length,size=int(np.ceil(length/block)))
        ix=((starts[:,None]+np.arange(block))%length).ravel()[:length]
        weights[j]=np.bincount(ix,minlength=length)
    return weights


def panel_summary(values,date_index,date_count,weights):
    sums=np.zeros((date_count,)+values.shape[1:]);counts=np.zeros(date_count)
    np.add.at(sums,date_index,values);np.add.at(counts,date_index,1)
    denom=weights@counts
    boot=(weights@sums.reshape(date_count,-1))/denom[:,None]
    boot=boot.reshape((len(weights),)+values.shape[1:])
    return values.mean(0),boot


def summarize(root=OUTPUT):
    summary={};allmarket=[];lines=['# Real-market density diagnostic','',
        'Exploratory 2011–2014 development panel. Lower pinball is better; desired coverage is 95%, upper/lower violations 1%. Horizons count retained observations. Separately fitted RV5 and RK models; no latent-variance or return-VaR claim.','']
    for proxy in ['rv5','rk_parzen']:
        paths=sorted((root/'forecasts').glob(f'*_{proxy}.npz'))
        scores=[];dates=[];stress=[];negs=[];mc=[];mcquant=[];mcflips=[];marketrows=[]
        for path in paths:
            with np.load(path) as d:
                sc=d['scores'];scores.append(sc);dates.append(d['dates']);stress.append(d['stress']);negs.append(d['negative_root_paths'])
                market=path.stem.removesuffix('_'+proxy) if hasattr(str,'removesuffix') else path.stem[:-len(proxy)-1]
                row=dict(market=market,proxy=proxy,origins=len(sc),scores=sc.mean(0).tolist());marketrows.append(row);allmarket.append(row)
                repeats=json.loads(path.with_name(path.stem+'_repeat.json').read_text())
                for r in repeats:
                    oi=r['origin_number'];q=np.array(r['quantiles'])
                    rs=np.array([score(a,d['truth'][oi],d['scale'][oi]) for a in q])
                    mc.append(rs-sc[oi]);mcquant.append(np.abs(q-d['quantiles'][oi])/(HORIZONS[None,:,None]*d['scale'][oi]))
                    mcflips.append(rs[:,:,1:4]!=sc[oi,:,:,1:4])
        values=np.concatenate(scores);dd=np.concatenate(dates);ss=np.concatenate(stress);negative=np.concatenate(negs)
        unique,ix=np.unique(dd,return_inverse=True);w=block_weights(len(unique),60)
        proxyresult=dict(origins=len(values),markets=len(paths),date_min=str(unique[0]),date_max=str(unique[-1]),groups={},
            negative_root_path_fraction=negative.mean(0).tolist(),mc_repeated_origins=len(mc),
            mc_mean_score_change=np.mean(mc,axis=0).tolist(),mc_mean_absolute_quantile_change=np.mean(mcquant,axis=0).tolist(),
            mc_calibration_classification_flip_rate=np.mean(mcflips,axis=0).tolist())
        groups={'all':np.ones(len(values),bool),'2011_2012':dd<'2013-01-01',
                '2013_2014':dd>='2013-01-01','high_volatility':ss,'ordinary_volatility':~ss}
        for name,mask in groups.items():
            point,boot=panel_summary(values[mask],ix[mask],len(unique),w)
            # Empty resampled subgroup blocks are possible; retain valid ratios only.
            valid=np.all(np.isfinite(boot),axis=(1,2,3));boot=boot[valid]
            ci=np.quantile(boot,[.025,.975],axis=0)
            result=dict(origins=int(mask.sum()),point=point.tolist(),ci95=ci.tolist(),valid_bootstrap_replicates=len(boot),pairs={})
            for a,b in [(0,1),(2,1),(3,1),(4,1),(0,3),(2,3),(4,3)]:
                delta=boot[:,a,:,0]-boot[:,b,:,0]
                result['pairs'][METHODS[a]+'_minus_'+METHODS[b]]=dict(point=(point[a,:,0]-point[b,:,0]).tolist(),ci95=np.quantile(delta,[.025,.975],axis=0).tolist())
            proxyresult['groups'][name]=result
        sensitivity={}
        for block in [20,120]:
            point,boot=panel_summary(values,ix,len(unique),block_weights(len(unique),block))
            sensitivity[str(block)]=dict(ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),
                bayes_minus_bootstrap_garch_pinball_ci95=np.quantile(boot[:,0,:,0]-boot[:,3,:,0],[.025,.975],axis=0).tolist())
        proxyresult['block_sensitivity']=sensitivity
        marketpoint=np.array([r['scores'] for r in marketrows])
        proxyresult['market_wins_vs_bootstrap_loghar']=(marketpoint[:,:,:,0]<marketpoint[:,1:2,:,0]).sum(0).tolist()
        summary[proxy]=proxyresult
        lines.extend([f'## {proxy}: {len(values):,} origins across {len(paths)} markets','',
            '| Method | Horizon | Pinball | 95% coverage [date-block interval] | Lower 1% | Upper 1% | Width |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        r=proxyresult['groups']['all'];p=np.array(r['point']);ci=np.array(r['ci95'])
        for m,method in enumerate(METHODS):
            for hi,h in enumerate(HORIZONS):
                a=p[m,hi];lines.append(f'| {method} | {h} | {a[0]:.5f} | {100*a[1]:.2f}% [{100*ci[0,m,hi,1]:.2f}, {100*ci[1,m,hi,1]:.2f}] | {100*a[2]:.2f}% | {100*a[3]:.2f}% | {a[4]:.3f} |')
        lines.extend(['','### Twenty-two-observation subsets','',
            '| Subset | Origins | Method | Pinball | Coverage | Upper 1% |','|---|---:|---|---:|---:|---:|'])
        for group in ['2011_2012','2013_2014','high_volatility','ordinary_volatility']:
            rr=proxyresult['groups'][group];pp=np.array(rr['point'])
            for m,method in enumerate(METHODS):
                a=pp[m,2];lines.append(f'| {group} | {rr["origins"]} | {method} | {a[0]:.5f} | {100*a[1]:.2f}% | {100*a[3]:.2f}% |')
        lines.extend(['',f'Square-root model: {100*negative[:,-1].mean():.2f}% of simulated paths contain a negative signed root before squaring. This is a support limitation of its Gaussian linear specification, not negative variance after transformation.',
            f'Independent 4,096-draw repeats at {len(mc)} origins are stored in summary.json with quantile changes, score changes and classification flip rates.',''])
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'market_scores.json').write_text(json.dumps(allmarket,indent=2))
    lines.extend(['## Interpretation limits','',
        'The calendar-date block intervals preserve shared market dates and use 60-date blocks (20/120 sensitivity in JSON). They are exploratory and unadjusted for the many comparisons. The development period has been reused. Tail events per individual market are sparse. Overlapping horizons, historical selection, data filtering, and omitted contemporary model classes limit generalization.',
        'Bayesian log-HAR parameter uncertainty and GARCH log paths can imply problematic or infinite variance-level predictive moments. We score finite quantiles and never interpret Monte Carlo sample means as exact finite predictive means. No clipping was applied.',
        'The Gaussian square-root HAR-GARCH model follows the published functional form; this is not a reproduction of the original paper’s table, its NIG variant, or SHARP/SHARK. Benchmark choice alone cannot establish novelty. See PROTOCOL.md and DECISION.md.',''])
    (root/'REPORT.md').write_text('\n'.join(lines))
    status=json.loads((root/'status.json').read_text());status['state']='analysis_complete';(root/'status.json').write_text(json.dumps(status,indent=2))
    print('ANALYSIS COMPLETE')


if __name__=='__main__':summarize()
