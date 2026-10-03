"""Cross-fitted DEVELOPMENT study of ordinary learned simulator priors.

Run: ../venv/bin/python -m experiments.prior_adaptation.experiment
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from experiments.prior_adaptation.model import (
    component_paths, log_law, log_prior, make_bank, marginal_fit, predictive_fit)
from volpfn.prior import initial_scale


GROUPS = {
    'americas': 'BVSP DJI GSPTSE IXIC MXX RUT SPX'.split(),
    'europe': 'AEX BFX FCHI FTSE GDAXI IBEX OMXC20 OMXHPI OMXSPI OSEAX SMSI SSMI STOXX50E'.split(),
    'asia_pacific': 'AORD HSI KS11 N225'.split(),
    'asia_other': 'BSESN KSE NSEI SSEC'.split(),
}
DATA = Path('experiments/data/oxfordman_barde/calendar_development_v2')
BURN = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def load_market(path):
    with np.load(path, allow_pickle=False) as a:
        date = a['date'].astype('U10'); r = a['returns'].astype(float)*100.
    if (not len(date) or len(date) != len(r) or np.any(date > '2014-12-31')
            or np.any(date[1:] <= date[:-1]) or not np.isfinite(r).all()):
        raise ValueError('Expected strictly dated finite development-only returns')
    if (date <= '2007-12-31').sum() < 500:
        raise ValueError('Insufficient pre-2008 observations')
    scale = initial_scale(r[:100])
    return {'date': date, 'r': r, 'z': r/scale, 'scale': scale}


def make_blocks(market, bank, length):
    r, date = market['z'], market['date']
    phase = np.where(date <= '2007-12-31', 'fit',
                     np.where(date <= '2010-12-31', 'select', 'evaluate'))
    ewma = np.ones(len(r)); current = 1.
    for t in range(100, len(r)):
        ewma[t] = current
        current = .94*current+.06*r[t]**2
    blocks = []
    for role in ('fit', 'select', 'evaluate'):
        idx = np.flatnonzero((phase == role) & (np.arange(len(r)) >= 100))
        for offset in range(0, len(idx), length):
            chosen = idx[offset:offset+length]
            if len(chosen) <= BURN:
                continue
            if np.any(np.diff(chosen) != 1):
                raise ValueError('A block is not contiguous in the observation sequence')
            lp, _ = component_paths(r[chosen], bank, h0=ewma[chosen[0]])
            cumulative = np.cumsum(lp, axis=0)
            blocks.append({'role': role, 'index': chosen, 'cumulative': cumulative})
    return blocks, ewma


def training_arrays(records, sources):
    end, before, weights = [], [], []
    for name in sorted(sources):
        blocks = [b for b in records[name] if b['role'] == 'fit']
        if not blocks:
            raise ValueError('No training blocks')
        for b in blocks:
            end.append(b['cumulative'][-1]); before.append(b['cumulative'][BURN-1])
            weights.append(1/len(blocks))
    return np.array(end), np.array(before), np.array(weights)


def block_density(block, prior):
    marginal = logsumexp(block['cumulative']+log_prior(prior), axis=1)
    return np.diff(np.r_[0., marginal])


def validation_score(records, markets, sources, prior):
    means = []
    for name in sorted(sources):
        parts = [block_density(b, prior)[BURN:]-np.log(markets[name]['scale'])
                 for b in records[name] if b['role'] == 'select']
        means.append(np.concatenate(parts).mean())
    return float(np.mean(means))


def conventional_scores(market):
    """Causal GARCH refits; each fit receives only observations before origin."""
    from arch import arch_model
    z = market['z']; date = market['date']; scale = market['scale']
    _, ewma = make_blocks(market, make_bank()[0][:1], 512)
    ewma_score = log_law(z, ewma, np.full(len(z), 8.))-np.log(scale)
    out = np.full(len(z), np.nan); fits = []
    origins = np.flatnonzero(date >= '2011-01-01')
    if not len(origins):
        raise ValueError('Missing evaluation dates')
    for start in range(int(origins[0]), len(z), 250):
        history = z[max(0, start-1000):start]
        fit = arch_model(history, mean='Constant', vol='GARCH', p=1, q=1,
                         dist='t', rescale=False).fit(disp='off', show_warning=False)
        valid = int(fit.convergence_flag) == 0
        fits.append({'origin': str(date[start]), 'history_n': len(history),
                     'converged': valid, 'flag': int(fit.convergence_flag)})
        if not valid:
            raise RuntimeError('GARCH benchmark failed to converge; no selective date removal')
        p = fit.params; h = float(fit.conditional_volatility[-1]**2)
        for t in range(start, min(start+250, len(z))):
            h = float(p['omega']+p['alpha[1]']*(z[t-1]-p['mu'])**2+p['beta[1]']*h)
            out[t] = log_law(np.array([z[t]-p['mu']]), np.array([h]),
                            np.array([p['nu']]))[0]-np.log(scale)
    return {'garch_t': out, 'ewma_t8': ewma_score}, fits


def paired_calendar_interval(frame, candidate, control, seed=927601, reps=2000, block=60):
    pivot = frame.pivot(index='date', columns='market', values=[candidate, control])
    diff = (pivot[candidate]-pivot[control]).to_numpy()
    n = len(diff); rng = np.random.default_rng(seed); sampled = []
    for _ in range(reps//100):
        starts = rng.integers(0, n, size=(100, int(np.ceil(n/block))))
        indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(100, -1)[:, :n]
        sampled.extend(np.nanmean(np.nanmean(diff[indices], axis=1), axis=1).tolist())
    market_means = frame.groupby('market')[[candidate, control]].mean()
    delta = market_means[candidate]-market_means[control]
    return {'mean_difference': float(delta.mean()),
            'calendar_block_interval95': np.quantile(sampled, [.025, .975]).tolist(),
            'markets_improved': int((delta > 0).sum()), 'markets': len(delta),
            'seed': seed, 'resamples': len(sampled), 'block_observed_dates': block,
            'scope': 'Exploratory interval conditional on fitted models and chosen markets; '
                     'resamples dates jointly across markets, not training/fold uncertainty.'}


def run(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir(); (root/'weights').mkdir(); (root/'scores').mkdir()
    started = time.perf_counter()
    sources = [Path(__file__), Path('experiments/prior_adaptation/model.py'),
               Path('experiments/prior_adaptation/PROTOCOL.md'),
               Path('experiments/test_prior_adaptation.py'), Path('experiments/test_volpfn_causality.py'),
               Path('src/volpfn/model.py'), Path('src/volpfn/prior.py')]
    source_hash = {str(p): sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p, root/'source'/p.name)
    names = sorted(n for g in GROUPS.values() for n in g)
    if len(set(names)) != len(names):
        raise ValueError('Overlapping groups')
    bank, fixed = make_bank()
    data_hash = {n: sha(DATA/(n+'.npz')) for n in names}
    spec = {'created_utc': datetime.now(timezone.utc).isoformat(), 'groups': GROUPS,
            'block_lengths': [512, 128], 'burn': BURN, 'atoms': bank.tolist(),
            'sources': source_hash, 'data': data_hash, 'arguments': vars(args),
            'checkpoint_sha256': sha(args.checkpoint) if args.checkpoint else None,
            'primary': 'Selected learned minus selected simple mean log density, block length 512.',
            'scope': 'Cross-fitted DEVELOPMENT only, 2015+ never opened; ordinary EB baselines, no neural training.'}
    save(root/'protocol.json', spec)
    save(root/'status.json', {'status': 'running', 'pid': os.getpid(), 'phase': 'load'})
    try:
        markets = {n: load_market(DATA/(n+'.npz')) for n in names}
        classical = {}; benchmark_fit = {}; neural = {}; causal_delta = {}
        if args.checkpoint:
            import importlib.util
            import torch
            import volpfn.legacy as V
            torch.set_num_threads(args.threads)
            model = V.load_model(args.checkpoint, 'cpu')
            old_dir = Path('code_backups/20260906_prior_adaptation')
            def module(path, name):
                spec = importlib.util.spec_from_file_location(name, path)
                m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
            old = module(old_dir/'src/volpfn/model.py', '_before_causal_V')
            old.VP = module(old_dir/'src/volpfn/prior.py', '_before_causal_VP')
            for n in names:
                tick = time.perf_counter()
                r = markets[n]['r']
                repaired = V.predict_series(model, r, min_history=100, levels=())
                legacy = old.predict_series(model, r, levels=())
                neural[n] = repaired['logscore']
                mask = markets[n]['date'] >= '2011-01-01'
                change = repaired['logscore'][mask]-legacy['logscore'][mask]
                causal_delta[n] = {'mean_log_density_change': float(change.mean()),
                                   'max_abs_change': float(np.max(np.abs(change))),
                                   'observations': int(mask.sum())}
                print(f'CHECKPOINT {n}: causal-minus-legacy={change.mean():+.8f}; '
                      f'{time.perf_counter()-tick:.2f}s', flush=True)
                save(root/'status.json', {'status': 'running', 'pid': os.getpid(),
                                         'phase': 'checkpoint', 'market': n})
            save(root/'causal_checkpoint_comparison.json', causal_delta)
        for n in names:
            classical[n], benchmark_fit[n] = conventional_scores(markets[n])
        save(root/'garch_fits.json', benchmark_fit)
        summaries = {}
        for length in (512, 128):
            records = {n: make_blocks(markets[n], bank, length)[0] for n in names}
            frames = []; folds = {}
            for group, targets in GROUPS.items():
                source_markets = sorted(set(names)-set(targets))
                assert not set(source_markets) & set(targets)
                end, before, weight = training_arrays(records, source_markets)
                eb, eb_fit = marginal_fit(end, weight)
                pp, pp_fit = predictive_fit(end, before, weight,
                                            {'uniform': fixed['uniform_expanded'], 'marginal_eb': eb})
                if not eb_fit['converged'] or not pp_fit['converged']:
                    save(root/'weights'/f'{length}_{group}_failed.json',
                         {'marginal_eb': eb_fit, 'predictive_fit': pp_fit})
                    raise RuntimeError('Prior optimizer failed diagnostic; no outcome-based substitution')
                learned = {'marginal_eb': eb, 'predictive_fit': pp}
                all_priors = dict(fixed, **learned)
                scores = {k: validation_score(records, markets, source_markets, w)
                          for k, w in all_priors.items()}
                simple_name = max(sorted(fixed), key=lambda k: scores[k])
                learned_name = max(sorted(learned), key=lambda k: scores[k])
                frozen = {'source_markets': source_markets, 'target_markets': targets,
                          'selected_simple': simple_name, 'selected_learned': learned_name,
                          'source_validation_scores': scores,
                          'marginal_fit': eb_fit, 'predictive_fit': pp_fit,
                          'priors': {k: w.tolist() for k, w in all_priors.items()},
                          'frozen_utc': datetime.now(timezone.utc).isoformat()}
                save(root/'weights'/f'{length}_{group}.json', frozen)
                folds[group] = frozen
                for name in targets:
                    for b in records[name]:
                        if b['role'] != 'evaluate':
                            continue
                        idx = b['index'][BURN:]
                        values = {k: block_density(b, w)[BURN:]-np.log(markets[name]['scale'])
                                  for k, w in all_priors.items()}
                        values['selected_simple'] = values[simple_name]
                        values['selected_learned'] = values[learned_name]
                        values.update({k: x[idx] for k, x in classical[name].items()})
                        if name in neural:
                            values['volpfn_repaired'] = neural[name][idx]
                        if not all(np.isfinite(x).all() for x in values.values()):
                            raise FloatingPointError('Nonfinite scores; no selective row dropping')
                        frames.append(pd.DataFrame(dict(date=markets[name]['date'][idx],
                                                        market=name, group=group, **values)))
                print(f'FROZEN length={length} group={group}: simple={simple_name}, '
                      f'learned={learned_name}', flush=True)
                save(root/'status.json', {'status': 'running', 'pid': os.getpid(),
                                         'phase': 'priors', 'length': length, 'group': group})
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(root/'scores'/f'length_{length}.csv', index=False)
            numeric = frame.select_dtypes(include=[np.number]).columns
            per_market = frame.groupby('market')[numeric].mean()
            per_market.to_csv(root/f'per_market_{length}.csv')
            contrasts = {control: paired_calendar_interval(frame, 'selected_learned', control)
                         for control in ('selected_simple', 'garch_t', 'ewma_t8')}
            groups = {}
            for group, targets in GROUPS.items():
                delta = per_market.loc[targets, 'selected_learned']-per_market.loc[targets, 'selected_simple']
                groups[group] = float(delta.mean())
            summaries[str(length)] = {'market_equal_mean_log_density': per_market.mean().to_dict(),
                                     'contrasts': contrasts, 'group_gain_vs_simple': groups,
                                     'markets': len(per_market), 'scored_rows': len(frame)}
            save(root/f'summary_{length}.json', summaries[str(length)])
        primary = summaries['512']['contrasts']['selected_simple']
        competitive = summaries['512']['contrasts']['garch_t']['mean_difference'] > 0
        supported = primary['calendar_block_interval95'][0] > 0
        decision = ('Prior fitting warrants a separate neural mechanism study; this is still development evidence.'
                    if supported and competitive else
                    'Do not scale a prior-adaptive network from this result. Report the measured trade-offs; '
                    'learned-prior improvement is uncertain or does not beat the conventional GARCH control.')
        result = {'lengths': summaries, 'decision': decision,
                  'primary_interval_supports_improvement': supported,
                  'primary_mean_beats_garch': competitive,
                  'elapsed_seconds': time.perf_counter()-started,
                  'neural_training': False, 'reserved_outcomes_accessed': False}
        save(root/'summary.json', result)
        lines = ['# Learned-prior development study', '', decision, '',
                 'Higher log density is better; nats per scored percentage-return observation. '
                 'Equal weight per market. All results are development evidence, not market confirmation.', '',
                 '| Method | 512-observation blocks (primary) | 128-observation blocks |',
                 '|---|---:|---:|']
        for method, mean in summaries['512']['market_equal_mean_log_density'].items():
            lines.append(f'| {method} | {mean:.8f} | '
                         f'{summaries["128"]["market_equal_mean_log_density"][method]:.8f} |')
        lines += ['', '| Block length | Control | Learned minus control | Paired date-block 95% interval | Markets improved |',
                  '|---:|---|---:|---|---:|']
        for length, values in summaries.items():
            for control, c in values['contrasts'].items():
                lines.append(f'| {length} | {control} | {c["mean_difference"]:+.8f} | '
                             f'{c["calendar_block_interval95"]} | {c["markets_improved"]}/{c["markets"]} |')
        lines += ['', 'The bootstrap preserves contemporaneous dependence by sampling dates together, '
                  'but conditions on fitted priors and this market panel. It is exploratory and does not '
                  'include training uncertainty or multiple-comparison adjustment. The primary block length '
                  'was fixed in advance. Exactness is relative to the finite conditional block model.', '',
                  'The original twenty-stock Yahoo results were not reproduced: their signed input snapshot '
                  'was not found. A checkpoint comparison here uses different open-to-close index returns. '
                  'The GP-Vol implementation and tail backtests have not been revalidated by this study.', '',
                  'No new neural model was trained. See frozen fold weights, per-date scores, optimizer '
                  'diagnostics and the protocol for the information sets and limitations.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
        changed_sources = [p for p, value in source_hash.items() if sha(p) != value]
        changed_data = [n for n, value in data_hash.items() if sha(DATA/(n+'.npz')) != value]
        if changed_sources or changed_data:
            raise RuntimeError('Source or data changed during experiment')
        save(root/'verification.json', {'sources_unchanged_during_run': True,
                                      'data_unchanged': True,
                                      'checkpoint_unchanged': not args.checkpoint or sha(args.checkpoint) == spec['checkpoint_sha256']})
        save(root/'status.json', {'status': 'complete', 'elapsed_seconds': time.perf_counter()-started})
        print('COMPLETE', json.dumps({'primary': primary, 'decision': decision,
                                      'seconds': time.perf_counter()-started}), flush=True)
    except BaseException as exc:
        save(root/'status.json', {'status': 'failed', 'error': str(exc),
                                 'elapsed_seconds': time.perf_counter()-started})
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='experiments/outputs/prior_adaptation_v1')
    ap.add_argument('--checkpoint', default='out_volpfn/volpfn_base_best.pt')
    ap.add_argument('--threads', type=int, default=2)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
