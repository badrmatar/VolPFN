"""Synthetic tests of inference design, independent of real market outcomes."""
import unittest

import numpy as np
import pandas as pd

from research.seed_replication.statistics import analyze_seed_contrasts, holm


SEEDS = (8101, 8102, 8103, 8104, 8105)


def fixture(seed_offsets=None, date_effect=None, markets=("a", "b"), contrasts=("gain",)):
    dates = pd.date_range("2001-01-01", periods=240)
    offsets = np.repeat(.01, 5) if seed_offsets is None else np.asarray(seed_offsets)
    daily = np.zeros(len(dates)) if date_effect is None else np.asarray(date_effect)
    rows = [(s, m, d, c, float(offsets[j] + daily[k]))
            for j, s in enumerate(SEEDS) for m in markets for k, d in enumerate(dates) for c in contrasts]
    frame = pd.DataFrame(rows, columns=["seed", "market", "date", "contrast", "gain"])
    masks = {c: frame.loc[frame.contrast == c, ["market", "date"]].drop_duplicates() for c in contrasts}
    return frame, masks


def analyze(frame, masks, families=None, **kwargs):
    return analyze_seed_contrasts(frame, families=families or {"family": list(masks)},
                                  expected_rows=masks, expected_seeds=SEEDS,
                                  reps=999, block=20, seed=1234, **kwargs)


class SeedInferenceTests(unittest.TestCase):
    def test_equal_seed_market_mean_not_pooled_rows(self):
        frame, masks = fixture(seed_offsets=np.arange(5) * .01)
        # Unequal market lengths must not give the long market larger weight.
        frame = frame.loc[~((frame.market == "b") & (frame.date > "2001-06-30"))].copy()
        frame.loc[frame.market == "b", "gain"] += .02
        masks["gain"] = frame[["market", "date"]].drop_duplicates()
        result = analyze(frame, masks)["contrasts"]["gain"]
        self.assertAlmostEqual(result["crossed"]["mean_gain"], .03)
        np.testing.assert_allclose(list(result["per_seed_mean_gain"].values()), np.arange(5) * .01 + .01)

    def test_robust_positive_gain_survives(self):
        frame, masks = fixture()
        result = analyze(frame, masks)["contrasts"]["gain"]
        self.assertTrue(result["claim_survives"])
        self.assertTrue(result["all_seeds_positive"])

    def test_one_negative_seed_blocks_survival(self):
        frame, masks = fixture(seed_offsets=[.01, .01, .01, .01, -.0001])
        result = analyze(frame, masks)["contrasts"]["gain"]
        self.assertGreater(result["crossed"]["mean_gain"], 0)
        self.assertFalse(result["all_seeds_positive"])
        self.assertFalse(result["claim_survives"])

    def test_seed_variation_is_not_treated_as_independent_rows(self):
        frame, masks = fixture(seed_offsets=[-.02, -.01, 0, .01, .02])
        result = analyze(frame, masks)["contrasts"]["gain"]
        self.assertLess(np.ptp(result["conditional"]["interval95"]), 1e-12)
        self.assertGreater(np.ptp(result["crossed"]["interval95"]), .01)

    def test_shared_market_shocks_do_not_create_fake_sample_size(self):
        daily = .005 * np.sin(np.arange(240) / 20.)
        one, mask_one = fixture(date_effect=daily, markets=("a",))
        many, mask_many = fixture(date_effect=daily, markets=tuple("abcdefghij"))
        x, y = analyze(one, mask_one), analyze(many, mask_many)
        for method in ("crossed", "conditional"):
            np.testing.assert_allclose(x["contrasts"]["gain"][method]["interval95"],
                                       y["contrasts"]["gain"][method]["interval95"], atol=1e-12)

    def test_missing_entire_seed_never_averages_survivors(self):
        frame, masks = fixture()
        result = analyze(frame.loc[frame.seed != SEEDS[-1]], masks)
        gain = result["contrasts"]["gain"]
        self.assertEqual(gain["status"], "unavailable")
        self.assertIsNone(gain["crossed"]["mean_gain"])
        self.assertEqual(gain["crossed"]["p_null"], 1.)
        self.assertAlmostEqual(gain["per_seed_mean_gain"][str(SEEDS[0])], .01)
        self.assertIsNone(gain["per_seed_mean_gain"][str(SEEDS[-1])])

    def test_missing_or_shifted_dated_row_is_unavailable(self):
        frame, masks = fixture()
        altered = frame.copy()
        altered.loc[0, "date"] = pd.Timestamp("2000-12-31")
        for f in (frame.iloc[1:], altered):
            result = analyze(f, masks)["contrasts"]["gain"]
            self.assertEqual(result["status"], "unavailable")
            self.assertTrue(any("scoring_mask_mismatch" in x for x in result["reasons"]))

    def test_globally_missing_row_is_detected_with_frozen_mask(self):
        frame, masks = fixture()
        result = analyze(frame.loc[frame.date != pd.Timestamp("2001-01-01")], masks)
        self.assertEqual(result["contrasts"]["gain"]["status"], "unavailable")

    def test_invalid_gain_and_explicit_failure_cannot_be_hidden(self):
        frame, masks = fixture()
        bad = frame.copy()
        bad.loc[0, "gain"] = np.nan
        self.assertEqual(analyze(bad, masks)["contrasts"]["gain"]["status"], "unavailable")
        failures = [{"seed": SEEDS[0], "contrast": "gain", "failure_code": "nonconverged"}]
        result = analyze(frame, masks, model_failures=failures)
        self.assertEqual(result["contrasts"]["gain"]["status"], "unavailable")

    def test_reserved_slots_remain_in_holm_family(self):
        frame, masks = fixture()
        result = analyze(frame, masks, families={"family": ["gain", "reserved"]})
        gain, absent = result["contrasts"]["gain"], result["contrasts"]["reserved"]
        self.assertEqual(absent["crossed"]["p_null"], 1.)
        self.assertAlmostEqual(gain["holm_by_family"]["family"]["crossed_p_holm"], .002)
        self.assertFalse(result["families"]["family"]["all_slots_available"])

    def test_labels_order_does_not_affect_pairing(self):
        frame, masks = fixture(seed_offsets=[.001, .004, .006, .009, .012],
                               date_effect=.001 * np.sin(np.arange(240) / 20.))
        a = analyze(frame, masks)
        b = analyze(frame.sample(frac=1, random_state=91), masks)
        self.assertEqual(a, b)

    def test_duplicate_and_unregistered_labels_raise(self):
        frame, masks = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze(pd.concat([frame, frame.iloc[:1]]), masks)
        bad = frame.copy()
        bad.loc[0, "seed"] = 9999
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            analyze(bad, masks)

    def test_equivalence_requires_declared_precision(self):
        frame, masks = fixture(seed_offsets=[.0002] * 5)
        result = analyze(frame, masks, equivalence_contrast="gain")
        self.assertTrue(result["equivalence"]["crossed"]["equivalence_supported"])
        noisy, noisy_masks = fixture(seed_offsets=[-.01, -.005, 0, .005, .01])
        result = analyze(noisy, noisy_masks, equivalence_contrast="gain")
        self.assertFalse(result["equivalence"]["crossed"]["equivalence_supported"])
        self.assertTrue(result["equivalence"]["conditional"]["equivalence_supported"])

    def test_equivalence_missing_seed_is_unavailable(self):
        frame, masks = fixture(seed_offsets=[0] * 5)
        result = analyze(frame.loc[frame.seed != SEEDS[0]], masks, equivalence_contrast="gain")
        self.assertEqual(result["equivalence"]["crossed"]["status"], "unavailable")
        self.assertEqual(result["equivalence"]["crossed"]["p_tost"], 1.)

    def test_common_resampling_across_contrasts(self):
        frame, masks = fixture(date_effect=.01 * np.sin(np.arange(240) / 20.), contrasts=("a", "b"))
        result = analyze(frame, masks)["contrasts"]
        self.assertEqual(result["a"]["crossed"], result["b"]["crossed"])
        self.assertEqual(result["a"]["conditional"], result["b"]["conditional"])

    def test_holm_retains_family_multiplicity(self):
        np.testing.assert_allclose(holm([.01, .02, 1., .04]), [.04, .06, 1., .08])

    def test_missing_frozen_masks_are_disclosed(self):
        frame, _ = fixture()
        result = analyze_seed_contrasts(frame, families={"f": ["gain"]}, reps=99, block=20)
        self.assertFalse(result["all_scoring_masks_supplied"])
        self.assertIn("globally missing rows", result["scoring_mask_sources"]["gain"])


if __name__ == "__main__":
    unittest.main()
