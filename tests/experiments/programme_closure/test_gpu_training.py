import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.programme_closure import gpu_training as G
from experiments.matched_prior import run as M


class TrainingGateTests(unittest.TestCase):
    def test_step_stream_is_reproducible_and_restartable(self):
        s=G.specification('cpu');s.update(steps=6,tokens=64,lengths={'32':.5,'64':.5},workers=0,prefetch_depth=0)
        all_stream=G.OrderedBatches('broad',7103,s)
        saved=[all_stream.next() for _ in range(6)]
        continuation=G.OrderedBatches('broad',7103,s,start=3)
        for expected in saved[3:]:
            got=continuation.next()
            for key in expected:
                if isinstance(expected[key],np.ndarray): np.testing.assert_array_equal(got[key],expected[key])

    def test_both_arms_have_identical_initial_weights(self):
        s=G.specification('cpu');s['cfg']=dict(d=32,layers=1,heads=2,K=3)
        G.configure(s)
        a,_,ha=G.initialize(8101,s);b,_,hb=G.initialize(8101,s)
        self.assertEqual(ha,hb)
        self.assertEqual(M.state_hash(a),M.state_hash(b))

    def test_no_budget_or_checkpoint_selection_changes(self):
        s=G.specification()
        self.assertEqual(s['steps'],20000);self.assertEqual(s['tokens'],8192)
        self.assertEqual(s['seeds'],[8101,8102,8103,8104,8105])
        self.assertIn('fixed final',s['checkpoint_selection'])
        schedule=list(M.schedule(8101,s))
        self.assertEqual(sum(batch*length for _,batch,length in schedule),163840000)

    def test_finite_update_both_priors(self):
        s=G.specification('cpu');s.update(cfg=dict(d=32,layers=1,heads=2,K=3),steps=4,tokens=128,lengths={'64':1.},workers=0)
        G.configure(s)
        for arm in G.ARMS:
            model,optimizer,_=G.initialize(8101,s)
            batch=G.generated((arm,77,2,64))
            values=G.update(model,optimizer,batch,0,s)
            self.assertTrue(np.isfinite(values).all())


if __name__=='__main__': unittest.main()
