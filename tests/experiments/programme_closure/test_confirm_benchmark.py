"""Synthetic/mock tests; these never read or score the market panel."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.programme_closure import confirm_benchmark as C
from experiments.frontier_benchmark import metrics


def fixture():
    def row(market,date,h,target=1.):
        return dict(market=market,date=date,target=target,h=h,
                    raw_quantiles=(np.arange(-4,5)*.2*np.sqrt(h)).tolist())
    seeds={s:[row(m,'2015-04-%02d'%d,1.+.15*i+.01*d,1.+.1*(m=='B'))
              for d in range(1,13) for m in ('A','B')] for i,s in enumerate(C.SEEDS)}
    calibration={s:[row(m,'2012-01-%02d'%d,1.) for d in range(1,4) for m in ('A','B')] for s in C.SEEDS}
    competitors={name:[row(r['market'],r['date'],2.,r['target']) for r in seeds[C.SEEDS[0]]]
                 for name in C.COMPETITORS}
    settings=dict(main_block=3,sensitivity_blocks=[2,4],resamples=99,seed=177)
    return seeds,calibration,competitors,{m:1. for m in C.COMPETITORS},settings


class ConfirmationTests(unittest.TestCase):
    def test_incomplete_battery_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new_confirmation'
            with patch.object(C,'verify_plan',return_value={'training_design':'unused'}), \
                 patch.object(C.E,'verify_checkpoints',side_effect=ValueError('incomplete battery')):
                with self.assertRaisesRegex(ValueError,'incomplete battery'):
                    C.run('unused','unused',out)
            self.assertFalse(out.exists())

    def test_all_five_seeds_required(self):
        args=list(fixture());args[0].pop(8105)
        with self.assertRaisesRegex(ValueError,'All five'):
            C.score_complete(*args)

    def test_misaligned_forecast_or_calibration_fails(self):
        args=list(fixture());args[0][8102][0]['date']='2015-04-25'
        with self.assertRaisesRegex(ValueError,'Forecast rows differ'):
            C.score_complete(*args)
        args=list(fixture());args[1][8102][0]['target']=3.
        with self.assertRaisesRegex(ValueError,'calibration rows differ'):
            C.score_complete(*args)

    def test_averages_scores_not_forecasts(self):
        args=fixture();result,draws,counts=C.score_complete(*args)
        seeds=args[0]
        direct=np.mean([np.mean(metrics.qlike(np.array([r['target']**2 for r in rows]),
                                            np.array([r['h'] for r in rows]))) for rows in seeds.values()])
        mean_h=np.mean([[r['h'] for r in seeds[s]] for s in C.SEEDS],axis=0)
        ensemble=np.mean(metrics.qlike(np.array([r['target']**2 for r in seeds[C.SEEDS[0]]]),mean_h))
        self.assertAlmostEqual(result['seed_mean_scores']['qlike_fixed_scale'],direct,places=12)
        self.assertGreater(abs(direct-ensemble),1e-4)
        self.assertEqual(len(result['contrasts']),9)
        self.assertEqual(len(counts),3)
        for row in result['contrasts']:
            self.assertEqual(len(row['per_seed_gain']),5)
            self.assertAlmostEqual(row['gain'],np.mean(list(row['per_seed_gain'].values())),places=12)

    def test_equal_market_weights_not_equal_rows(self):
        args=list(fixture())
        for collection in (args[0],args[2]):
            for identity,rows in collection.items():
                collection[identity]=[r for r in rows if r['market']=='A' or int(r['date'][-2:])%2==0]
        result,_,_=C.score_complete(*args)
        for seed in C.SEEDS:
            rows=args[0][seed]
            direct=np.mean([np.mean([float(metrics.qlike(r['target']**2,r['h'])) for r in rows if r['market']==m])
                            for m in ('A','B')])
            self.assertAlmostEqual(result['seeds'][str(seed)]['mean']['qlike_fixed_scale'],direct,places=12)

    def test_postdevelopment_calibration_rejected(self):
        args=list(fixture())
        for rows in args[1].values():rows[0]['date']='2015-01-01'
        with self.assertRaisesRegex(ValueError,'outside fixed development'):
            C.score_complete(*args)

    def test_native_prediction_matches_archived_frontier_adapter(self):
        import torch
        import volpfn
        from experiments.frontier_benchmark.lstm import predict
        torch.set_num_threads(1);torch.manual_seed(901)
        model=volpfn.VolPFN(d=16,layers=1,heads=2,K=4).eval()
        past=np.random.default_rng(99).normal(size=128)
        expected=predict(model,past)
        actual=C.native_prediction(model,past,'cpu')
        self.assertEqual(expected['h'],actual['h'])
        np.testing.assert_array_equal(expected['quantiles'],actual['raw_quantiles'])


if __name__=='__main__':unittest.main()
