import unittest
import numpy as np
import pandas as pd
from .build import evaluation_origins
from .report import combine,verdict


class ProtocolChecks(unittest.TestCase):
    def test_fixed_window_boundaries(self):
        panel={'A':dict(date=np.array(pd.date_range('2010-01-01','2016-07-01').strftime('%Y-%m-%d')))}
        mask=pd.DataFrame(dict(market=['A']*4,date=['2015-03-31','2015-04-01','2016-06-30','2016-07-01']))
        self.assertEqual([r['date'] for r in evaluation_origins(panel,mask)],['2015-04-01','2016-06-30'])

    def test_no_old_period_overlap(self):
        a=[dict(market='A',date='2015-03-31')];b=[dict(market='A',date='2015-04-01')]
        self.assertEqual(len(combine(a,b)),2)
        with self.assertRaises(ValueError):combine(a,a)
        with self.assertRaises(ValueError):combine(a,[dict(market='A',date='2016-07-01')])

    def test_three_way_conclusion(self):
        self.assertEqual(verdict(.1,.01,[.02,.18]),'supported_superiority')
        self.assertEqual(verdict(-.1,.01,[-.18,-.02]),'supported_inferiority')
        self.assertEqual(verdict(.1,.01,[-.02,.22]),'inconclusive')
        self.assertEqual(verdict(.1,.08,[.02,.18]),'inconclusive')


if __name__=='__main__':unittest.main()
