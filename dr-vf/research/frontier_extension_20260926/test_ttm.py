"""Synthetic contracts; TTM_TEST_PINNED=1 also checks cached public weights.

Run with the existing foundation_venv Python, -B, and unittest. No market
predictions, market scores, downloads, training, or external writes occur.
"""
from datetime import date, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import numpy as np
import torch

from .ttm import (ASSET_HASHES, DEFAULT_SNAPSHOT, TTMRealizedAdapter,
                  calibration_factor, causal_fill_rk, verify_snapshot)


class PointStub:
    def __init__(self, point=0.0):
        self.point = point

    def to(self, device):
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def __call__(self, past_values, freq_token):
        self.values = past_values.clone()
        self.tokens = freq_token.clone()
        return SimpleNamespace(prediction_outputs=torch.full(
            (len(past_values), 96, 1), self.point, dtype=torch.float32))


class Contracts(unittest.TestCase):
    def test_direct_variance_does_not_square_RK_or_output(self):
        x = np.linspace(1, 9, 512)
        model = PointStub()
        row = TTMRealizedAdapter(model=model).forecast_context(x)
        self.assertEqual(row["variance"], x.mean())
        self.assertEqual(row["second_moment"], x.mean())
        self.assertEqual(row["mean"], 0)
        self.assertFalse(row["checkpoint_verified"])
        self.assertEqual(tuple(model.tokens.shape), (1, 1))
        self.assertEqual(model.tokens.item(), 8)
        np.testing.assert_allclose(model.values[0, :, 0], (x-x.mean())/x.std(), atol=1e-7)

    def test_batch_scaling_is_per_context_and_scale_equivariant(self):
        x = np.linspace(1, 9, 512)
        adapter = TTMRealizedAdapter(model=PointStub(.3))
        batch = adapter.forecast_batch(np.stack([x, x*100, np.full(512, 4.)]))
        singles = [adapter.forecast_context(y) for y in (x, x*100, np.full(512, 4.))]
        for a, b in zip(batch, singles):
            self.assertEqual(a, b)
        self.assertAlmostEqual(batch[1]["variance"], 100*batch[0]["variance"])

    def test_missing_inputs_filled_from_past_without_filling_target(self):
        x = np.linspace(1, 5, 620)
        x[105:108] = [np.nan, 0., -1.]
        x[615] = np.nan
        adapter = TTMRealizedAdapter(model=PointStub())
        rows = adapter.forecast_series(x, [{"index":615, "rk":None}], batch_size=1)
        self.assertIsNone(rows[0]["rk"])
        self.assertEqual(rows[0]["imputed_context_count"], 3)
        changed = x.copy(); changed[615:] = 99999
        self.assertEqual(rows[0], adapter.forecast_series(changed, [{"index":615, "rk":None}])[0])
        # The prior value outside the 512-row window can fill its leading gap.
        y = np.ones(520); y[8:11] = np.nan
        self.assertEqual(adapter.forecast_context(y)["imputed_context_count"], 3)
        with self.assertRaises(ValueError):
            adapter.forecast_context(np.r_[np.nan, np.ones(511)])

    def test_sqrt_diagnostic_preserves_Jensen_gap_and_negative_SD_flag(self):
        x = np.linspace(1, 9, 512)
        row = TTMRealizedAdapter(transform="sqrt", model=PointStub()).forecast_context(x)
        self.assertAlmostEqual(row["variance"], np.sqrt(x).mean()**2)
        self.assertLess(row["variance"], x.mean())
        negative = TTMRealizedAdapter(transform="sqrt", model=PointStub(-100.)).forecast_context(x)
        self.assertTrue(negative["raw_nonpositive"])
        self.assertTrue(negative["floor_applied"])
        self.assertIsNone(negative["raw_h"])
        self.assertAlmostEqual(negative["variance"], 1e-8*x.mean())

    def test_nonfinite_predictions_fail_and_nonpositive_points_are_recorded(self):
        with self.assertRaises(FloatingPointError):
            TTMRealizedAdapter(model=PointStub(float("nan"))).forecast_context(np.ones(512))
        row = TTMRealizedAdapter(model=PointStub(-100.)).forecast_context(np.ones(512))
        self.assertLess(row["raw_h"], 0)
        self.assertEqual(row["variance"], 1e-8)
        self.assertTrue(row["floor_applied"])
        with self.assertRaises(ValueError):
            TTMRealizedAdapter(model=PointStub()).forecast_batch(np.ones((2, 511)))

    def test_calibration_only_accepts_the_past_development_population(self):
        rows = [dict(market=m, date=(date(2012,1,4)+timedelta(days=i)).isoformat(),
                     phase="calibration", r2=value, second_moment=2.)
                for m,value in (("FTSE",2.),("N225",4.),("SPX",6.)) for i in range(24)]
        self.assertEqual(calibration_factor(rows), 2.)
        future = [dict(r) for r in rows]; future[0]["date"] = "2019-01-01"
        with self.assertRaises(ValueError): calibration_factor(future)
        with self.assertRaises(ValueError): calibration_factor(rows[:-1])

    def test_hash_verification_rejects_changed_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ASSET_HASHES:
                (Path(directory)/name).write_bytes(b"not the pinned artifact")
            with self.assertRaises(ValueError): verify_snapshot(directory)


@unittest.skipUnless(os.environ.get("TTM_TEST_PINNED") == "1", "opt-in cached-weight engineering gate")
class CachedWeights(unittest.TestCase):
    def test_synthetic_public_model_batch_single_repeat_and_future_invariance(self):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        rng = np.random.default_rng(20260926)
        contexts = np.stack([np.exp(rng.normal(-1, .4, 512)),
                             np.exp(rng.normal(1, .8, 512)),
                             np.linspace(.01, 4., 512)])
        adapter = TTMRealizedAdapter(snapshot=DEFAULT_SNAPSHOT)
        started = time.perf_counter()
        batch = adapter.forecast_batch(contexts)
        elapsed = time.perf_counter() - started
        singles = [adapter.forecast_context(x) for x in contexts]
        repeat = adapter.forecast_batch(contexts)
        b = np.array([r["variance"] for r in batch])
        s = np.array([r["variance"] for r in singles])
        np.testing.assert_allclose(b, s, rtol=2e-5, atol=1e-7)
        self.assertEqual(batch, repeat)
        self.assertTrue(all(row["checkpoint_verified"] for row in batch))
        history = np.r_[contexts[0], np.linspace(.5, 1., 110)]
        origins = [512, 530, 600]
        series = adapter.forecast_series(history, origins, batch_size=3)
        for origin, row in zip(origins, series):
            changed = history.copy(); changed[origin:] = 1e6
            altered = adapter.forecast_series(changed, [origin])[0]
            np.testing.assert_allclose(row["variance"], altered["variance"], rtol=2e-5, atol=1e-7)
        tick = time.perf_counter()
        adapter.forecast_batch(np.tile(contexts, (22, 1)))
        warm = time.perf_counter()-tick
        print(json.dumps(dict(kind="synthetic_only", market_predictions=0,
            first_batch_seconds=elapsed, warm_66_batch_seconds=warm,
            batch_single_max_abs=float(np.max(abs(b-s))),
            batch_single_max_rel=float(np.max(abs(b-s)/s)),
            checkpoint=adapter.identity, torch=torch.__version__)), flush=True)


if __name__ == "__main__":
    unittest.main()
