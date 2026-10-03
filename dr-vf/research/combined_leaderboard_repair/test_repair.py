import unittest
import numpy as np
from research.combined_leaderboard.common import *
from research.combined_leaderboard.neural import compare as old_compare, Predictor
from research.combined_leaderboard.pipeline import verify as old_verify, cell
from .resume import compare,RTOL,ORIGINAL_RTOL,validated_old_cells

class NumericalGateTests(unittest.TestCase):
    def test_recorded_failure_is_not_erased_or_called_original_pass(self):
        f=next((OUTPUT/'failures').glob('bridge_*.json'))
        before=sha(f);failure=read(f)
        new=np.asarray(failure['values'])[:,2];old=np.asarray(failure['archived'])
        self.assertFalse(old_compare(new,old)['passed'])
        amended=compare(new,old)
        self.assertTrue(amended['passed'])
        self.assertFalse(amended['original_passed'])
        self.assertEqual(amended['original_exceedance_count'],1)
        self.assertEqual(amended['checked'],64)
        self.assertEqual(sha(f),before)

    def test_large_mismatch_is_rejected_at_every_scale(self):
        old=np.array([1e-9,1e-3,1.,1e9])
        self.assertFalse(compare(old*(1+1e-4),old)['passed'])
        self.assertTrue(compare(old*(1+2e-6),old)['passed'])
        self.assertEqual(RTOL,1e-5)
        self.assertEqual(ORIGINAL_RTOL,1e-6)

    def test_invalid_moments_and_shapes_fail_closed(self):
        for a,b in (([],[]),([1],[1,2]),([np.nan],[1]),([np.inf],[1]),
                    ([0],[1]),([-1],[1]),([1],[0]),([1],[np.nan]),([[1]],[[1]])):
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):compare(a,b)

    def test_exactly_identical_is_both_passes(self):
        result=compare([.2,7.,40.],[.2,7.,40.])
        self.assertTrue(result['passed'] and result['original_passed'])
        self.assertEqual(result['original_exceedance_count'],0)
        self.assertEqual(result['relative_errors'],[0.,0.,0.])

class PreservationTests(unittest.TestCase):
    def test_original_inventory_matches_independent_validation(self):
        self.assertEqual(len(validated_old_cells()),341)

    def test_base_freeze_and_every_saved_neural_cell_revalidate(self):
        _,sealed=old_verify(OUTPUT)
        original_id=sha(OUTPUT/'FREEZE.json');count=0
        lookups={s:{key(r):r['h'] for phase in ('calibration','evaluation')
                    for r in read(CACHE/f'{s}_{phase}.json')} for s in SEEDS}
        for path in sorted((OUTPUT/'cells/neural').rglob('forecast.json')):
            value=cell(OUTPUT,path.parent.relative_to(OUTPUT/'cells').with_suffix('.json'),original_id)
            rows=value['rows'];seed=rows[0]['seed']
            result=compare([r['second_moment'] for r in rows],[lookups[seed][key(r)] for r in rows])
            self.assertTrue(result['passed'] and result['original_passed'])
            count+=len(rows)
        self.assertEqual(count,3712)

    def test_unchanged_forecaster_reproduces_complete_failed_block(self):
        failure=read(next((OUTPUT/'failures').glob('bridge_*.json')))
        with np.load(PLAN.parent/'contexts.npz') as z:
            contexts=np.concatenate([z['calibration'],z['evaluation']])
        p=Predictor(failure['seed'],'cpu',2)
        values=np.concatenate([p.predict(contexts[i:i+1]) for i in range(failure['start'],failure['end'])])
        np.testing.assert_array_equal(values,np.asarray(failure['values']))
        self.assertTrue(compare(values[:,2],failure['archived'])['passed'])
        self.assertFalse(old_compare(values[:,2],failure['archived'])['passed'])

if __name__=='__main__':unittest.main()
