"""Common frozen-mask evaluation of fixed-final ablation checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.experiment_programme_20260930 import ablations as A, programme as P
from research.experiment_programme_20260930.garch_lstm import student_t_nll


def context_hash(dates, returns):
    digest = hashlib.sha256()
    digest.update(np.asarray(dates, dtype='U10').tobytes())
    digest.update(np.asarray(returns, dtype='<f8').tobytes())
    return digest.hexdigest()


def evaluate_arm(manifest_path, ablation_manifest, training_root, arm, out, device='cuda', batch_size=8, seed=None):
    manifest = P.verify(manifest_path)
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / 'manifest.json'
    component = A.verify(ablation_manifest)
    binding = {'path': str(A._manifest_path(ablation_manifest).relative_to(P.ROOT)),
               'sha256': P.sha(A._manifest_path(ablation_manifest))}
    if manifest.get('component_design') != binding:
        raise ValueError('Evaluation component design differs from programme binding')
    if component['specification']['smoke']:
        raise ValueError('Smoke models cannot enter final empirical results')
    if arm not in manifest['volpfn_arms'] or component['specification']['seeds'] != manifest['seeds']:
        raise ValueError('Ablation arm/seed roster differs from the programme')
    if device != component['specification']['device']:
        raise ValueError('Evaluation backend differs from the registered training backend')
    A.G.configure(component['specification'])
    seeds = manifest['seeds'] if seed is None else [seed]
    if not set(seeds).issubset(manifest['seeds']):
        raise ValueError('Unregistered evaluation seed')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / 'status.json', {'status': 'running', 'arm': arm})
    all_rows = []
    try:
        mask = pd.read_csv(manifest_path.parent / manifest['mask_files']['evaluation']['path'],
                           dtype={'market': str, 'date': str}, float_precision='round_trip')
        hashes = {}
        for seed in seeds:
            run = Path(training_root) / 'runs' / (arm + '_' + str(seed))
            receipt = json.loads((run / 'status.json').read_text())
            checkpoint = run / 'final.pt'
            if receipt['status'] != 'complete' or receipt['checkpoint_sha256'] != P.sha(checkpoint):
                raise ValueError('Required fixed-final checkpoint is incomplete or changed')
            hashes[str(seed)] = P.sha(checkpoint)
            model = A.load_model(checkpoint, ablation_manifest, device)
            for market in manifest['population']:
                dates, returns, kernel = P.read_market(market)
                target_rows = mask.loc[mask.market == market].sort_values('row_index')
                indices = target_rows.row_index.to_numpy(int)
                if np.any(indices < 1024) or not np.array_equal(dates[indices], target_rows.date.to_numpy()):
                    raise ValueError('A target lacks the registered1024 completed observations or has a wrong date')
                rows = []
                for begin in range(0, len(indices), batch_size):
                    targets = indices[begin:begin + batch_size]
                    histories = [returns[index-1024:index] for index in targets]
                    observed = returns[targets]
                    prediction = A.predict_windows(model, histories, batch_size=batch_size,
                                                   device=device, targets=observed)
                    h = prediction['h']
                    if not np.isfinite(h).all() or np.any(h <= 0):
                        raise FloatingPointError('Invalid issued second moment')
                    qlike = observed ** 2 / h + np.log(h)
                    common = student_t_nll(torch.tensor(observed, dtype=torch.float64),
                        torch.zeros(len(targets), dtype=torch.float64), torch.tensor(h, dtype=torch.float64),
                        torch.tensor(8., dtype=torch.float64), reduction='none').numpy()
                    for offset, index in enumerate(targets):
                        rk = float(kernel[index])
                        row = dict(model=arm, seed=seed, market=market, date=str(dates[index]),
                            row_index=int(index), context_count=1024, context_start=str(dates[index-1024]),
                            context_end=str(dates[index-1]),
                            context_sha256=context_hash(dates[index-1024:index], histories[offset]),
                            target_sha256=context_hash(dates[index:index+1], observed[offset:offset+1]),
                            return_value=float(observed[offset]), second_moment=float(h[offset]),
                            qlike=float(qlike[offset]), common_t8_nll=float(common[offset]),
                            kernel=rk if np.isfinite(rk) and rk > 0 else None,
                            kernel_qlike=(rk / h[offset] + np.log(h[offset])) if np.isfinite(rk) and rk > 0 else None,
                            native_nll=float(prediction['native_nll'][offset]) if prediction.get('native_nll') is not None else None,
                            mean=float(prediction['mean'][offset]) if 'mean' in prediction else None,
                            variance=float(prediction['variance'][offset]) if 'variance' in prediction else None)
                        row['return'] = row.pop('return_value')
                        rows.append(row)
                    P.write_json(out / 'status.json', {'status': 'running', 'arm': arm,
                        'seed': seed, 'market': market, 'market_rows_completed': begin + len(targets)})
                table = pd.DataFrame(rows)
                table.to_csv(out / (market + '_' + str(seed) + '.csv'), index=False, float_format='%.17g')
                all_rows.extend(rows)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        frame = pd.DataFrame(all_rows)
        from research.experiment_programme_20260930.statistics import validate_forecasts
        validate_forecasts(frame, mask, seeds)
        frame.to_csv(out / 'forecasts.csv', index=False, float_format='%.17g')
        summary = frame.groupby(['seed', 'market']).qlike.mean().groupby('seed').mean()
        result = {'status': 'complete', 'arm': arm, 'seeds': seeds,
                  'scope': 'all_required_seeds' if seed is None else 'seed_shard_not_a_headline',
                  'rows_per_seed': len(mask), 'mean_equal_market_seed_qlike': float(summary.mean()),
                  'per_seed_equal_market_qlike': {str(seed): float(value) for seed, value in summary.items()},
                  'manifest_sha256': P.sha(manifest_path), 'ablation_manifest_sha256': P.sha(A._manifest_path(ablation_manifest)),
                  'checkpoint_hashes': hashes, 'forecast_sha256': P.sha(out / 'forecasts.csv'),
                  'density': 'Native mixture density where present; direct head has no native density',
                  'kernel': 'Secondary proxy, decimal² converted topercent²; target equivalence unverified'}
        P.write_json(out / 'result.json', result)
        P.write_json(out / 'status.json', {'status': 'complete', 'arm': arm})
        return result
    except BaseException as error:
        P.write_json(out / 'status.json', {'status': 'failed', 'arm': arm, 'error': repr(error),
                                          'coverage_policy': 'No incomplete seed or row is promoted'})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ablation-manifest', required=True)
    parser.add_argument('--training-root', required=True)
    parser.add_argument('--arm', choices=P.ARMS, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, help='One registered shard; requires later all-seed aggregation')
    args = parser.parse_args()
    result = evaluate_arm(args.manifest, args.ablation_manifest, args.training_root, args.arm, args.out,
                          args.device, args.batch_size, args.seed)
    print(json.dumps({'status': result['status'], 'arm': args.arm, 'out': args.out}))


if __name__ == '__main__':
    main()
