"""Meaningful guards and parity tests for the additive historical tail loader."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.audit_support.strict_tail_inputs import TailInputError, load_tail_inputs


class StrictTailInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.market, self.extract = self.root/'market.npz', self.root/'extract.npz'
        self.dates = np.array(['2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06'])
        self.r = np.array([.001, -.002, .003, -.004])
        self.split = np.array([0, 0, 1, 1], dtype=np.int8)
        self.h = np.array([np.nan, np.nan, 1.2, 1.5])
        self.write()

    def write(self, *, dates=None, r=None, split=None, h=None, extract_dates=None, extract_r=None):
        np.savez(self.market, date=self.dates if dates is None else dates,
                 returns=self.r if r is None else r, split=self.split if split is None else split)
        np.savez(self.extract, date=self.dates if extract_dates is None else extract_dates,
                 r=self.r*100 if extract_r is None else extract_r, h=self.h if h is None else h)

    def load(self, **kw):
        return load_tail_inputs(self.market, self.extract, scored_split=1, **kw)

    def test_valid_unscored_fill_and_scored_values_are_preserved(self):
        fallback = np.array([2., 3., 4., 5.])
        result = self.load(fallback_variance=fallback)
        np.testing.assert_array_equal(result.variance, [2., 3., 1.2, 1.5])
        np.testing.assert_array_equal(result.returns_percent, self.r*100)
        self.assertEqual(result.audit['scored_fallback_rows'], 0)
        self.assertEqual(result.audit['unscored_fallback_rows'], 2)
        self.assertFalse(result.variance.flags.writeable)

    def test_equal_length_shifted_calendar_is_rejected(self):
        self.write(extract_dates=np.array(['2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05']))
        with self.assertRaisesRegex(TailInputError, 'dates do not match'):
            self.load()

    def test_return_unit_or_value_mismatch_is_rejected(self):
        for wrong in (self.r, self.r*100 + .01):
            with self.subTest(wrong=wrong):
                self.write(extract_r=wrong)
                with self.assertRaisesRegex(TailInputError, 'percentage returns do not match'):
                    self.load()

    def test_every_invalid_scored_variance_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf, 0., -1.):
            with self.subTest(value=value):
                h=self.h.copy();h[2]=value;self.write(h=h)
                with self.assertRaisesRegex(TailInputError, 'scored fallback is forbidden'):
                    self.load()

    def test_missing_neural_extract_is_rejected(self):
        self.extract.unlink()
        with self.assertRaisesRegex(TailInputError, 'must exist'):
            self.load()

    def test_nonfinite_returns_and_invalid_calendar_are_rejected(self):
        bad=self.r.copy();bad[0]=np.nan;self.write(r=bad)
        with self.assertRaisesRegex(TailInputError, 'nonfinite percentage returns'):
            self.load()
        for dates in (self.dates[::-1], np.array(['2011-01-03']*4),
                      np.array(['2011-01-03T00:00:00', '2011-01-04', '2011-01-05', '2011-01-06'])):
            with self.subTest(dates=dates):
                self.write(dates=dates)
                with self.assertRaises(TailInputError):
                    self.load()

    def test_bad_lengths_split_and_fallback_are_rejected(self):
        self.write(h=self.h[:-1])
        with self.assertRaisesRegex(TailInputError, 'lengths differ'):
            self.load()
        self.write(split=np.array([0., 0., .5, 1.]))
        with self.assertRaisesRegex(TailInputError, 'integer labels'):
            self.load()
        self.write(split=np.zeros(4, dtype=int))
        with self.assertRaisesRegex(TailInputError, 'has no observations'):
            self.load()
        self.write()
        for fallback in (np.ones(3), np.array([1., np.nan, 1., 1.]), np.zeros(4)):
            with self.subTest(fallback=fallback):
                with self.assertRaisesRegex(TailInputError, 'Fallback variance'):
                    self.load(fallback_variance=fallback)


if __name__ == '__main__':
    unittest.main(verbosity=2)
