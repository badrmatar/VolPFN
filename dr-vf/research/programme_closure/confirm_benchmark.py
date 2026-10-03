"""Exposed-history procedure confirmation; no new competitor inference.

Freeze the inputs before corrected-checkpoint scoring. Running requires all ten
matched-prior finals, although only the five broad finals enter this comparison.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

from research.programme_closure import evaluate as E
from research.frontier_benchmark import metrics, data
from research.frontier_staged.report import validate
from research.frontier_uncertainty.run import matrix_for, resample, losses
from research.population_transfer.statistics import _calendar_counts, _basic_interval, holm

ROOT = Path(__file__).resolve().parents[2]
FRONTIER = ROOT / 'research/outputs/frontier_overnight_increment_201504_201606_v1'
DEVELOPMENT = ROOT / 'research/outputs/frontier_development_comparison_cuda_v1'
SEEDS = list(E.SEEDS)
COMPETITORS = ('chronos2', 'moirai2', 'tabpfn_ts')
METRICS = ('pinball', 'qlike_native', 'qlike_fixed_scale')
KIND = 'programme_closure_exposed_benchmark_confirmation_v1'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def freeze(training_design, out, frontier=FRONTIER, development=DEVELOPMENT, device='cpu'):
    """Lock old forecasts and exact contexts without scoring any new model."""
    E.verify_design(training_design)
    require(device in ('cpu', 'cuda'), 'Unsupported inference backend')
    frontier, development = Path(frontier).resolve(), Path(development).resolve()
    out = Path(out).resolve()
    require(not out.exists(), 'Refusing to overwrite confirmation plan')
    for protected in (frontier, development, Path(training_design).resolve()):
        require(not (out == protected or out in protected.parents or protected in out.parents),
                'Output overlaps protected evidence')
    study = E.sealed(frontier/'study.json')
    manifest = E.sealed(frontier/'bundle/manifest.json')
    require(E.sha(frontier/'bundle/manifest.json') == study['bundle_manifest_sha256'], 'Frontier manifest drift')
    require(E.sha(frontier/'bundle/inputs.npz') == manifest['files']['inputs.npz'], 'Frontier contexts drift')
    archive_hashes = read(frontier/'analysis_v1/hashes.json')
    require(E.sha(frontier/'analysis_v1/inputs.json') == archive_hashes['inputs.json'], 'Archived analysis inputs drift')
    archived_inputs = read(frontier/'analysis_v1/inputs.json')
    require(archived_inputs['study_sha256'] == E.sha(frontier/'study.json'), 'Archived study provenance mismatch')
    source_files = [Path(__file__), Path(E.__file__), Path(metrics.__file__), Path(data.__file__),
        ROOT/'research/frontier_staged/report.py', ROOT/'research/frontier_uncertainty/run.py',
        ROOT/'research/population_transfer/statistics.py', ROOT/'research/seed_replication/causal.py',
        ROOT/'volpfn.py', Path(__file__).with_name('GPU_PROTOCOL.md')]
    # The simulator has intentionally changed, but these inference/scoring paths have not.
    for name in ('volpfn.py', 'research/seed_replication/causal.py', 'research/frontier_benchmark/metrics.py'):
        path = ROOT/name
        require(E.sha(path) == study['hashes'][str(path)], 'Archived inference convention changed: '+name)
    origins = [o for o in manifest['origins'] if o['phase'] == 'evaluation']
    require(len(origins) == 8774 and len({o['market'] for o in origins}) == 28, 'Wrong fixed scoring mask')
    files = {str(p): E.sha(p) for p in [frontier/'study.json', frontier/'bundle/manifest.json',
        frontier/'bundle/inputs.npz', frontier/'analysis_v1/inputs.json', frontier/'analysis_v1/hashes.json']}
    for model in COMPETITORS:
        path, status_path = frontier/'models'/model/'predictions.json', frontier/'models'/model/'status.json'
        status = read(status_path)
        require(status['status'] == 'complete' and E.sha(path) == status['forecast_sha256']
                == archived_inputs['forecasts'][str(path)], 'Incomplete or changed saved competitor: '+model)
        require(model != 'tabpfn_ts' or status.get('bridge_passed') is True, 'Missing saved TabPFN bridge')
        files.update({str(path): E.sha(path), str(status_path): E.sha(status_path)})
    dev = E.sealed(development/'manifest.json')
    snapshot = development/'development_data.npz'
    require(E.sha(snapshot) == dev['development_snapshot_sha256'], 'Development snapshot drift')
    calibration = [o for o in dev['origins'] if o['phase'] == 'calibration']
    require(len(calibration) == 72 and dev['selected_markets'] == ['SPX', 'FTSE', 'N225'], 'Calibration population changed')
    require(len({(o['market'],o['date']) for o in calibration}) == 72 and
            all(sum(o['market']==m for o in calibration)==24 for m in dev['selected_markets']), 'Calibration mask changed')
    require(all('2012-01-01' <= o['date'] <= '2013-12-31' for o in calibration), 'Calibration dates changed')
    calibration_full, pasts = [], []
    with np.load(snapshot, allow_pickle=False) as saved:
        for o in calibration:
            dates, returns = saved[o['market']+'__date'], saved[o['market']+'__r']
            i = int(o['index'])
            require(str(dates[i]) == o['date'], 'Calibration origin misaligned')
            past = data.context(returns, i)
            require(len(past) == 1024, 'Calibration context differs from declared1024')
            pasts.append(past)
            calibration_full.append(dict(o, target=float(returns[i]), context_count=len(past),
                context_start=str(dates[i-1024]), context_end=str(dates[i-1])))
    files.update({str(development/'manifest.json'): E.sha(development/'manifest.json'), str(snapshot): E.sha(snapshot)})
    out.mkdir(parents=True, exist_ok=False)
    with np.load(frontier/'bundle/inputs.npz', allow_pickle=False) as inputs:
        keep = [i for i,o in enumerate(manifest['origins']) if o['phase'] == 'evaluation']
        evaluation = inputs['past'][keep]
        require(evaluation.shape == (8774,1024) and np.isfinite(evaluation).all(), 'Invalid evaluation context matrix')
        np.savez_compressed(out/'contexts.npz', calibration=np.asarray(pasts), evaluation=evaluation)
    plan = dict(kind=KIND, frozen_utc=datetime.now(timezone.utc).isoformat(), seeds=SEEDS,
        training_design=str(Path(training_design).resolve()), training_manifest_sha256=E.sha(Path(training_design)/'manifest.json'),
        frontier=str(frontier), development=str(development), origins=origins, calibration_origins=calibration_full,
        competitor_scales={m: study['scales'][m] for m in COMPETITORS},
        contexts_sha256=E.sha(out/'contexts.npz'), archived_files=files,
        sources={str(p.relative_to(ROOT)): E.sha(p) for p in source_files},
        inference=dict(device=device, per_origin_context=1024, precision='float32', tf32=False, deterministic=True,
                       output='native mixture second moment and native return quantiles; no stride ownership'),
        statistics=dict(main_block=10, sensitivity_blocks=[5,20], resamples=20000, seed=2026091902,
                        primary_metric='qlike_fixed_scale', holm_family=3, training_seed_resampling=False),
        calibration='Equal-market QLIKE scalar on the exact archived72 development origins, separately for each seed',
        scope='Exposed historical procedure sensitivity, conditional on five trained models and fixed competitors; no untouched-data, universal or causal-prior claim',
        original_results='Preserved separately; never select between original and corrected procedure by performance')
    E.save(out/'manifest.json', plan)
    (out/'manifest.sha256').write_text(E.sha(out/'manifest.json')+'\n')
    return plan


def verify_plan(path):
    plan = E.sealed(path)
    require(plan['kind'] == KIND and plan['seeds'] == SEEDS, 'Wrong confirmation plan')
    for name, digest in plan['archived_files'].items():
        require(E.sha(name) == digest, 'Archived evidence changed: '+name)
    for name, digest in plan['sources'].items():
        require(E.sha(ROOT/name) == digest, 'Confirmation source changed: '+name)
    require(E.sha(Path(path).parent/'contexts.npz') == plan['contexts_sha256'], 'Locked contexts changed')
    require(E.sha(Path(plan['training_design'])/'manifest.json') == plan['training_manifest_sha256'], 'Training design changed')
    return plan


def native_prediction(model, past, device):
    """The frontier native prediction formula, with explicit device transfer."""
    import torch
    from research.seed_replication import causal
    f = causal.feature_batch(np.asarray(past, float)[None])
    with torch.no_grad():
        p = model(torch.from_numpy(f['feats']).to(device), torch.from_numpy(f['base']).to(device))
    unit = float(f['output_scale'][0,-1])
    final = {k:v[0,-1].detach().cpu().numpy() for k,v in p.items()}
    logw, loc = final['logw'], final['loc']*unit
    scale, nu = np.exp(final['logscale'])*unit, final['nu']
    h = float(np.sum(np.exp(logw)*(loc**2+scale**2*nu/(nu-2))))
    require(np.isfinite(h) and h > 0, 'Invalid native mixture second moment')
    return dict(h=h, raw_quantiles=metrics.mixture_quantiles(logw,loc,scale,nu).tolist())


def score_complete(seed_rows, calibration_rows, competitor_rows, scales, settings):
    """Average score effects across seeds; never score an averaged forecast."""
    require(set(seed_rows) == set(SEEDS) == set(calibration_rows), 'All five seeds required')
    require(set(competitor_rows) == set(COMPETITORS), 'All three saved competitors required')
    reference = seed_rows[SEEDS[0]]
    key = lambda rows:[(r['market'],r['date'],r['target']) for r in rows]
    expected = key(reference)
    require(expected and len(set((m,d) for m,d,_ in expected)) == len(expected), 'Duplicate or empty scoring rows')
    require(all(key(rr) == expected for rr in list(seed_rows.values())+list(competitor_rows.values())), 'Forecast rows differ')
    dates, markets = [r['date'] for r in reference], np.array([r['market'] for r in reference])
    population = sorted(set(markets))
    equal = lambda values:float(np.mean([np.mean(values[markets==m]) for m in population]))
    descriptive, own_losses, fitted_scales = {}, {}, {}
    calibration_keys = key(calibration_rows[SEEDS[0]])
    for seed in SEEDS:
        c = calibration_rows[seed]
        require(key(c) == calibration_keys, 'Seed calibration rows differ')
        fitted_scales[seed] = metrics.fit_scale([r['target']**2 for r in c], [r['h'] for r in c],
                                               [r['market'] for r in c], [r['date'] for r in c])
        own_losses[seed] = losses(seed_rows[seed], fitted_scales[seed])
        descriptive[str(seed)] = dict(scale=fitted_scales[seed],
            mean={k:equal(v) for k,v in own_losses[seed].items()},
            per_market={m:{k:float(v[markets==m].mean()) for k,v in own_losses[seed].items()} for m in population})
    mean_losses = {metric:np.mean([own_losses[s][metric] for s in SEEDS],axis=0) for metric in METRICS}
    competitors = {m:losses(rr,scales[m]) for m,rr in competitor_rows.items()}
    results, draws_archive, counts_archive = [], {}, {}
    for block in [settings['main_block']]+settings['sensitivity_blocks']:
        group = []
        for competitor in COMPETITORS:
            gains = competitors[competitor]['qlike_fixed_scale']-mean_losses['qlike_fixed_scale']
            matrix, calendar, labels = matrix_for(dates, markets, gains)
            counts = _calendar_counts(len(calendar),settings['resamples'],block,settings['seed'])
            estimate, draws, per_market = resample(matrix,counts)
            counts_archive['block'+str(block)] = counts.astype(np.int16)
            draws_archive[competitor+'__block'+str(block)] = draws
            individual = {str(s):equal(competitors[competitor]['qlike_fixed_scale']-own_losses[s]['qlike_fixed_scale']) for s in SEEDS}
            np.testing.assert_allclose(estimate,np.mean(list(individual.values())),rtol=1e-12,atol=1e-12)
            p = (1+int(np.count_nonzero(abs(draws-estimate)>=abs(estimate))))/(settings['resamples']+1)
            group.append(dict(model=competitor,metric='qlike_fixed_scale',block=block,gain=estimate,p_raw=p,
                interval95=_basic_interval(estimate,draws,.95),family_interval95=_basic_interval(estimate,draws,1-.05/3),
                per_seed_gain=individual,seed_range=[min(individual.values()),max(individual.values())],
                positive_seeds=sum(v>0 for v in individual.values()),markets_improved=int(np.sum(per_market>0)),
                per_market_gain=dict(zip(labels,map(float,per_market)))))
        for row,p in zip(group,holm([r['p_raw'] for r in group])):
            row['p_holm3'] = float(p)
            row['verdict'] = ('supported_superiority' if row['gain']>0 and p<=.05 and row['family_interval95'][0]>0
                              else 'supported_inferiority' if row['gain']<0 and p<=.05 and row['family_interval95'][1]<0
                              else 'inconclusive')
        results.extend(group)
    return dict(contrasts=results,seeds=descriptive,seed_mean_scores={k:equal(v) for k,v in mean_losses.items()},
                competitors={m:{k:equal(v) for k,v in values.items()} for m,values in competitors.items()},
                rows=len(reference),markets=len(population),calendar_dates=len(set(dates)),
                scope='Exposed historical confirmation conditional on five trained models, fixed competitors and fitted development scales; calendar uncertainty only, excluding seed-population, calibration and research-selection uncertainty; no ensemble'),draws_archive,counts_archive


def run(plan_path, training_root, out):
    plan = verify_plan(plan_path)
    registry = E.verify_checkpoints(plan['training_design'], training_root)
    require(registry['training_manifest_sha256'] == plan['training_manifest_sha256'], 'Wrong completed battery')
    out = Path(out).resolve()
    for protected in (Path(plan_path).resolve().parent,Path(training_root).resolve(),Path(plan['frontier']),Path(plan['development'])):
        require(not(out==protected or out in protected.parents or protected in out.parents), 'Output overlaps protected inputs')
    out.mkdir(parents=True,exist_ok=False)
    E.save(out/'CHECKPOINT_REGISTRY.json',registry)
    E.save(out/'status.json',dict(status='running',plan_sha256=E.sha(plan_path)))
    try:
        import torch
        from research.seed_replication import causal
        device = plan['inference']['device']
        require(device != 'cuda' or torch.cuda.is_available(), 'Declared CUDA inference unavailable')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        seed_rows, calibration = {}, {}
        with np.load(Path(plan_path).parent/'contexts.npz',allow_pickle=False) as contexts:
            for entry in registry['entries']:
                if entry['arm'] != 'broad': continue
                seed = entry['seed']
                model = causal.load_model(entry['checkpoint'],device)
                calibration[seed],seed_rows[seed] = [],[]
                for phase,origins,rows in [('calibration',plan['calibration_origins'],calibration[seed]),
                                           ('evaluation',plan['origins'],seed_rows[seed])]:
                    for index,origin in enumerate(origins):
                        p = native_prediction(model,contexts[phase][index],device)
                        rows.append(dict(origin,**p,h_semantics='native mixture second moment',floor_activated=False))
                        if (index+1)%100==0:
                            E.save(out/'status.json',dict(status='running',seed=seed,phase=phase,completed=index+1,total=len(origins)))
                    E.save(out/('%s_%s.json'%(seed,phase)),rows)
                del model
        # No comparative scoring until every seed and all fixed rows exist.
        frontier = Path(plan['frontier'])
        old_manifest = read(frontier/'bundle/manifest.json')
        competitors = {m:validate(read(frontier/'models'/m/'predictions.json'),old_manifest['origins']) for m in COMPETITORS}
        result,draws,counts = score_complete(seed_rows,calibration,competitors,plan['competitor_scales'],plan['statistics'])
        verify_plan(plan_path)
        for entry in registry['entries']:
            require(E.sha(entry['checkpoint']) == entry['checkpoint_sha256'], 'Checkpoint changed during inference')
        np.savez_compressed(out/'draws.npz',**draws)
        np.savez_compressed(out/'counts.npz',**counts)
        E.save(out/'FORECAST_HASHES.json',{p.name:E.sha(p) for p in out.glob('81*_*.json')})
        E.save(out/'results.json',dict(result,status='complete',plan_sha256=E.sha(plan_path)))
        lines=['# Corrected procedure: exposed historical benchmark confirmation','',result['scope']+'.',
               'The original fixed-checkpoint study remains separately reported. Positive gain favours VolPFN.', '',
               '| Competitor | Seed-mean QLIKE gain | Family95% interval | Holm3 p | Positive seeds | Verdict |',
               '|---|---:|---|---:|---:|---|']
        for row in result['contrasts']:
            if row['block'] != plan['statistics']['main_block']:continue
            lo,hi=row['family_interval95']
            lines.append('| %s | %.6g | [%.6g, %.6g] | %.6g | %d/5 | %s |' %
                         (row['model'],row['gain'],lo,hi,row['p_holm3'],row['positive_seeds'],row['verdict']))
        lines+=['','All8774 fixed rows and allfive broadfinals are retained. Scoring averages losses across seeds, not predictive distributions.',
                'Per-seed results, seed ranges and block5/20 sensitivities are retained in results.json. No equivalence follows from an inconclusive contrast.']
        (out/'READOUT.md').write_text('\n'.join(lines)+'\n')
        E.save(out/'status.json',dict(status='complete',results_sha256=E.sha(out/'results.json')))
        return result
    except BaseException as exc:
        E.save(out/'status.json',dict(status='failed',error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--training-design',type=Path)
    parser.add_argument('--training-root',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.training_design,args.out,device=args.device)
    else:run(args.plan,args.training_root,args.out)
