"""Material source/runtime/label gates, without MATLAB or scientific surrogates."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from research.experiment_programme_20260930 import rech


class RECHGates(unittest.TestCase):
    def test_primary_unavailable_even_with_verified_source_and_matlab(self):
        with patch.object(rech, "inspect_source", return_value={"verified": True}), patch.object(
            rech, "inspect_runtime", return_value={"matlab": "/licensed/matlab", "octave": None}
        ):
            receipt = rech.availability("source")
        self.assertFalse(receipt["available"])
        self.assertFalse(receipt["common_daily_available"])
        self.assertTrue(receipt["native_matlab_replay_candidate"])
        self.assertIn("nu <= 2", " ".join(receipt["reasons"]))
        self.assertFalse(receipt["forecast_arrays_created"])

    def test_api_cannot_emit_mle_or_cached_forecast_fallback(self):
        with patch.object(rech, "availability", return_value={"reasons": ["Bayesian adapter unavailable"]}):
            model = rech.RECH2022()
        for method in (model.fit, model.forward, model.forecast):
            with self.assertRaises(rech.RECHUnavailable):
                method([0.1, -0.2])

    def test_source_drift_or_missing_content_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_bytes(b"pinned")
            pins = {"README.md": hashlib.sha256(b"pinned").hexdigest(), "archive.zip": "missing"}
            with patch.object(rech, "PINNED_FILES", pins):
                self.assertFalse(rech.inspect_source(root)["verified"])
                del pins["archive.zip"]
                self.assertTrue(rech.inspect_source(root)["verified"])
                (root / "README.md").write_bytes(b"changed")
                self.assertFalse(rech.inspect_source(root)["verified"])

    def test_missing_runtime_does_not_create_replay_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "replay"
            with patch.object(rech, "availability", return_value={"source": {"verified": True}, "runtime": {"matlab": None}}):
                with self.assertRaises(rech.RECHUnavailable):
                    rech.run_author_source("source", output, 8101)
            self.assertFalse(output.exists())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "replay"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("important")
            with patch.object(rech, "availability", return_value={"source": {"verified": True}, "runtime": {"matlab": "/matlab"}}):
                with self.assertRaises(FileExistsError):
                    rech.run_author_source("source", output, 8101)
            self.assertEqual(marker.read_text(), "important")

    def test_archive_traversal_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside", "bad")
            with self.assertRaises(ValueError):
                rech._extract_zip(archive, root / "inside")
            self.assertFalse((root / "outside").exists())

    def test_matlab_paths_escaped_without_shell_interpolation(self):
        self.assertEqual(rech._matlab_literal("/tmp/O'Brien/$literal"), "'/tmp/O''Brien/$literal'")


if __name__ == "__main__":
    unittest.main()
