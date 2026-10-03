"""Versioned exact Gaussian log-likelihood algebra for the full RAPCF filter.

The original file is loaded into a private module namespace. Only that private
namespace's observation likelihood is replaced. All priors, particle counts,
GP calculations, resampling, clipping policies and RNG calls remain those of
the original source. Importing this module never changes the original module.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_PATH = ROOT / "research/gpvol_full/gpvol_rapcf.py"
VERSION = "gaussian_loglikelihood_algebra_v1_20260930"
_LOG_2 = math.log(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def log_normal_obs(x, v):
    """Log N(x; 0, exp(v)), without squaring x or clipping exp(-v).

    x must be a finite scalar and v finite log variances (any array shape).
    The quadratic contribution is exp(2 log|x| - v - log 2). Computing the
    half contribution directly also handles a representable log density when
    the unhalved ratio would overflow. A mathematically unrepresentable
    negative log density returns -inf; invalid parameters raise explicitly.
    Zero x has exactly zero quadratic contribution. Floating underflow of the
    half contribution is permitted: it is below float64 resolution.
    """
    x_arr = np.asarray(x, dtype=float)
    values = np.asarray(v, dtype=float)
    if x_arr.ndim != 0 or not np.isfinite(x_arr):
        raise ValueError("Observation must be a finite scalar")
    if not np.isfinite(values).all():
        raise ValueError("Log variances must be finite")
    x_scalar = float(x_arr)
    normalizer = -_HALF_LOG_2PI - 0.5 * values
    if x_scalar == 0.0:
        return normalizer
    log_half_ratio = 2.0 * math.log(abs(x_scalar)) - values - _LOG_2
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return normalizer - np.exp(log_half_ratio)


def load_filter_module():
    """Return a fresh, explicit repaired RAPCF binding in a private namespace."""
    spec = importlib.util.spec_from_file_location("_gpvol_likelihood_v1_private", ORIGINAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.log_normal_obs = log_normal_obs
    module.SUPPORTED_LIKELIHOOD_VERSION = VERSION
    return module


def provenance():
    """Bind the explicit source dependency and replacement to their file hashes."""
    return dict(version=VERSION, original_path=str(ORIGINAL_PATH),
                original_sha256=hashlib.sha256(ORIGINAL_PATH.read_bytes()).hexdigest(),
                replacement_path=str(Path(__file__).resolve()),
                replacement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                changed_binding="private module log_normal_obs only",
                full_primary_replay_complete=False)
