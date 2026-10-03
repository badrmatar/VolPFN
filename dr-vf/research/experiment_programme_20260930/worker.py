"""One finite matched arm/seed plus automatic scoring on an assigned GPU."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from research.experiment_programme_20260930 import ablations as A, evaluation as E, programme as P


def run(manifest_path, ablation_manifest, arm, seed, out, batch_size=8):
    manifest = P.verify(manifest_path)
    component = A.verify(ablation_manifest)
    if component['specification']['smoke'] or seed not in manifest['seeds'] or arm not in manifest['volpfn_arms']:
        raise ValueError('Worker requires a registered final arm and seed')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / 'worker_status.json', {'status': 'running', 'stage': 'train', 'arm': arm, 'seed': seed})
    try:
        trained = A.train(ablation_manifest, out / 'component_training', arm, seed)
        P.write_json(out / 'worker_status.json', {'status': 'running', 'stage': 'evaluate', 'arm': arm, 'seed': seed})
        scored = E.evaluate_arm(manifest_path, ablation_manifest, out / 'component_training', arm,
            out / 'component_evaluation' / arm / ('seed_' + str(seed)), 'cuda', batch_size, seed)
        result = {'status': 'complete', 'scope': 'one_registered_shard_not_a_complete_experiment',
                  'arm': arm, 'seed': seed, 'training': trained, 'evaluation': scored,
                  'programme_completion': 'Requires all five shards/arm, frozen-mask aggregation, uncertainty and baseline stages'}
        P.write_json(out / 'worker_result.json', result)
        P.write_json(out / 'worker_status.json', {'status': 'complete', 'arm': arm, 'seed': seed})
        return result
    except BaseException as error:
        P.write_json(out / 'worker_status.json', {'status': 'failed', 'arm': arm, 'seed': seed, 'error': repr(error)})
        (out / 'worker_failure.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ablation-manifest', required=True)
    parser.add_argument('--arm', choices=P.ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    result = run(args.manifest, args.ablation_manifest, args.arm, args.seed, args.out, args.batch_size)
    print(json.dumps({'status': result['status'], 'arm': args.arm, 'seed': args.seed}))


if __name__ == '__main__':
    main()
