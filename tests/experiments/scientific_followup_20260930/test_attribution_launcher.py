"""Safety and supported-module dispatch for the historical workflow launcher."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.scientific_followup_20260930.attribution import ROOT, parse_args


class LauncherTests(unittest.TestCase):
    def test_module_help_dispatch(self):
        result = subprocess.run(
            [sys.executable, "-m", "experiments.scientific_followup_20260930.attribution", "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--out", result.stdout)

    def test_existing_output_refused_before_scientific_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "original.txt"
            marker.write_text("preserved")
            with self.assertRaises(SystemExit):
                parse_args(["--out", directory])
            self.assertEqual(marker.read_text(), "preserved")

    def test_defaults_preserve_historical_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--out", str(Path(directory) / "new")])
            self.assertEqual(args.maxfev, 800)
            self.assertFalse(args.skip_likelihood)
            self.assertEqual(Path(args.extract), ROOT / "experiments/outputs/attribution_extract_v1")
            self.assertEqual(Path(args.reference), ROOT / "experiments/outputs/prior_adaptation_mechanism_v1/scores.csv")


if __name__ == "__main__":
    unittest.main()
