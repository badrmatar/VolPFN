"""Supported launcher for the preserved exploratory gain-surgery workflow.

Run from the repository root with ``python -m
experiments.scientific_followup_20260930.attribution --out NEW_DIRECTORY``.
This fixes command dispatch only. It does not certify the exploratory optimizer
fits or turn the retained checkpoint experiment into a final-model experiment.
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", default=str(ROOT / "experiments/outputs/attribution_extract_v1"))
    parser.add_argument("--reference", default=str(ROOT / "experiments/outputs/prior_adaptation_mechanism_v1/scores.csv"))
    parser.add_argument("--out", required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--garch-common", default=str(ROOT / "experiments/outputs/attribution_garch_common_v1/garch_common.csv"))
    parser.add_argument("--maxfev", type=int, default=800)
    parser.add_argument("--skip-likelihood", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error("--out must be a new directory; historical evidence is never overwritten")
    if args.maxfev < 1:
        parser.error("--maxfev must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Import by module from the repository root, rather than the failing direct
    # historical script invocation. No historical source is edited or patched.
    from experiments.attribution.gain_surgery import main as archived_main
    return archived_main(args)


if __name__ == "__main__":
    main()
