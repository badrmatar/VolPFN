import unittest
import numpy as np
from models import ORIGINAL_PATH,COMPILED_PATH,ES,gjr_paths
from unittest.mock import patch
from calibeat.fz import empirical_var_es


class ModelTests(unittest.TestCase):
    def test_compiled_recursion_is_exact(self):
        r=np.random.default_rng(32).standard_t(5,2100)
        for b0,b1,b2 in [(-.1,.85,-.05),(-.001,.998,-.4),(-.08,.76,-.2)]:
            np.testing.assert_array_equal(ORIGINAL_PATH(r,b0,b1,b2,-2),COMPILED_PATH(r,b0,b1,b2,-2))

    def test_compiled_objective_is_exact(self):
        r=np.random.default_rng(91).standard_t(5,1000)
        for alpha in (.05,.025,.01):
            for params in [(-.1,.85,-.05,-.5),(-.001,.998,-.4,-1.)]:
                with patch.object(ES,'caviar_path',ORIGINAL_PATH):a=ES.neg_al_loglik(params,r,alpha,-2.)
                with patch.object(ES,'caviar_path',COMPILED_PATH):b=ES.neg_al_loglik(params,r,alpha,-2.)
                self.assertEqual(a,b)

    def test_es_recursion_causal(self):
        r=np.random.default_rng(92).standard_t(5,1600);changed=r.copy();changed[1300:]+=100
        for fn in (ORIGINAL_PATH,COMPILED_PATH):
            np.testing.assert_array_equal(fn(r,-.05,.9,-.05,-2)[:1301],fn(changed,-.05,.9,-.05,-2)[:1301])

    def test_gjr_and_fhs_causal(self):
        r=np.random.default_rng(80).standard_t(8,1035)
        dates=np.arange(np.datetime64('2012-04-06'),np.datetime64('2012-04-06')+len(r)).astype('U10')
        # Exactly 1000 observations precede the first fitted/scored origin.
        dates[:1000]=np.arange(np.datetime64('2015-01-01')-1000,np.datetime64('2015-01-01')).astype('U10')
        dates[1000:]=np.arange(np.datetime64('2015-01-01'),np.datetime64('2015-01-01')+35).astype('U10')
        modified=r.copy();modified[1025:]+=50
        a,sa,_=gjr_paths(r,dates,[.05,.025,.01]);b,sb,_=gjr_paths(modified,dates,[.05,.025,.01])
        self.assertTrue(all(x['status']=='complete' for x in sa.values()))
        for key in a:
            for x,y in zip(a[key],b[key]):np.testing.assert_array_equal(x[:26],y[:26])


if __name__=='__main__':unittest.main()
