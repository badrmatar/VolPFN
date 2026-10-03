"""Blend VolPFN's forecast with realised-kernel models, using saved forecasts only.

Question: does VolPFN (daily returns only) carry information the RK models lack?
If a causal blend of VolPFN + Log-HAR beats Log-HAR alone, VolPFN adds something.

Blend (per proxy, per VolPFN seed):  log h = a + w*log(c1 h_VolPFN) + (1-w)*log(c2 h_X)
c1, c2 = frozen 72-origin calibration scalars. (a, w) minimise pooled QLIKE over
ALL PAST evaluation dates only (expanding window, all markets), refitted every 20
union dates; the first 60 dates use a=0, w=0.5. Weights for a proxy are fitted on
that proxy's own past targets. Same 22,026 primary rows, same 20,000-draw
calendar-block bootstrap (seed 2026092601, blocks 60/20/120) as the frozen run.

Blends: volpfn+loghar (main), volpfn+ms_har, volpfn+ttm_rk.
Controls: loghar+ms_har (does ANY second forecast help Log-HAR?) and
gjr_garch_t+loghar (does a strong GARCH add as much as VolPFN?).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/gpvol_full"))
import analyse_addendum as A  # noqa
from research.combined_leaderboard.statistics import _equal_market, _panel, _resample  # noqa
from research.population_transfer.statistics import _calendar_counts, holm, _basic_interval  # noqa

OUT = ROOT / "research/outputs/blend_rk_20260926_v1"
REFIT, BURN, RESAMPLES, SEED = 20, 60, 20000, 2026092601


def load(model):
    rows = A.load_cells(A.FRONTIER, model)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    res = {}
    for s, rr in by_seed.items():
        ids = [(r["phase"], r["market"], r["date"]) for r in rr]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate forecast origins for " + model)
        bundle = json.loads((A.FRONTIER / "bundle.json").read_text())
        expected = {(r["phase"], r["market"], r["date"]) for r in bundle["rows"]}
        if set(ids) != expected:
            raise ValueError("Incomplete or extra calibration/evaluation origins for " + model)
        targets = {(r["phase"], r["market"], r["date"]): r for r in bundle["rows"]}
        for row in rr:
            wanted = targets[row["phase"], row["market"], row["date"]]
            if any(row.get(k) != wanted.get(k) for k in ("r2", "rk")):
                raise ValueError("Calibration/evaluation target mismatch for " + model)
            if row.get("forecast_failed") or not np.isfinite([row["mean"], row["variance"], row["second_moment"]]).all():
                raise ValueError("Failed or nonfinite source forecast for " + model)
            if row["variance"] <= 0 or row["second_moment"] <= 0 or not np.isclose(
                    row["second_moment"], row["variance"] + row["mean"] ** 2, rtol=1e-7, atol=1e-10):
                raise ValueError("Incoherent source moments for " + model)
        cal = [r for r in rr if r["phase"] == "calibration"]
        if not cal:
            raise ValueError("Missing calibration origins for " + model)
        ratios = np.array([r["r2"] / r["second_moment"] for r in cal])
        if not np.isfinite(ratios).all() or np.any(ratios < 0):
            raise ValueError("Invalid calibration ratios for " + model)
        c, _ = _equal_market(ratios, np.array([r["market"] for r in cal]))
        ev = {(r["market"], r["date"]): r for r in rr if r["phase"] == "evaluation"}
        res[s] = (c, ev)
    return res


def qlike(y, h):
    y, h = np.broadcast_arrays(np.asarray(y, float), np.asarray(h, float))
    if not np.isfinite(y).all() or np.any(y < 0) or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("QLIKE requires finite nonnegative targets and positive forecasts")
    with np.errstate(over="ignore", invalid="ignore"):
        answer = y / h + np.log(h)
    if not np.isfinite(answer).all():
        raise ValueError("Nonfinite blend loss; forecasts cannot be dropped")
    return answer


def fit(la, lb, y):
    la, lb, y = map(lambda v: np.asarray(v, float), (la, lb, y))
    if la.ndim != 1 or la.shape != lb.shape or la.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned blend fitting vectors required")
    if not np.isfinite(la).all() or not np.isfinite(lb).all() or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Finite fitting inputs and nonnegative targets required")
    def f(p):
        lh = p[0] + p[1] * la + (1 - p[1]) * lb
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.mean(y * np.exp(-lh) + lh)
        return float(value) if np.isfinite(value) else float("inf")
    # Preserve the historical unconstrained optimizer and its tolerances.
    # This is a log-linear combination: w is not restricted to a percentage.
    result = minimize(f, x0=[0.0, 0.5], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("Blend optimizer failed: " + str(result.message))
    return result.x


def blend_series(keys, dates_idx, ha, hb, y):
    dates_idx = np.asarray(dates_idx)
    ha, hb, y = map(lambda v: np.asarray(v, float), (ha, hb, y))
    if y.ndim != 1 or not len(y) or any(v.shape != y.shape for v in (ha, hb, dates_idx)) or len(keys) != len(y):
        raise ValueError("Nonempty aligned blend histories required")
    if len(set(keys)) != len(keys) or not np.issubdtype(dates_idx.dtype, np.integer) or np.any(dates_idx < 0):
        raise ValueError("Unique origins and nonnegative integer calendar indices required")
    if not np.isfinite(ha).all() or not np.isfinite(hb).all() or np.any(ha <= 0) or np.any(hb <= 0):
        raise ValueError("Finite positive component forecasts required")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Finite nonnegative targets required")
    la, lb = np.log(ha), np.log(hb)
    out = np.empty(len(y)); weights = []
    params = np.array([0.0, 0.5])
    n_dates = dates_idx.max() + 1
    for d0 in range(0, n_dates, REFIT):
        if d0 >= BURN:
            past = dates_idx < d0
            params = fit(la[past], lb[past], y[past])
        sel = (dates_idx >= d0) & (dates_idx < d0 + REFIT)
        out[sel] = np.exp(params[0] + params[1] * la[sel] + (1 - params[1]) * lb[sel])
        weights.append(dict(from_date=int(d0), a=float(params[0]), w=float(params[1])))
    if not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError("Nonfinite or nonpositive blend forecast")
    return out, weights


def validate_sources(models, bundle):
    """Require full frozen coverage and shared targets before any blend fit."""
    expected = {(r["market"], r["date"]): r for r in bundle["rows"] if r["phase"] == "evaluation"}
    if len(expected) != sum(r["phase"] == "evaluation" for r in bundle["rows"]):
        raise ValueError("Duplicate bundle origins")
    for model, seeds in models.items():
        wanted = set(bundle["model_specs"][model]["seeds"])
        if set(seeds) != wanted:
            raise ValueError("Incomplete retained seed set for " + model)
        for scalar, rows in seeds.values():
            if not np.isfinite(scalar) or scalar <= 0 or set(rows) != set(expected):
                raise ValueError("Incomplete coverage or invalid calibration for " + model)
            for key, row in rows.items():
                for field in ("r2", "rk"):
                    if row.get(field) != expected[key].get(field):
                        raise ValueError("Target mismatch for " + model + " at " + str(key))


def prepare_output(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    return out


def main(out):
    out = prepare_output(out)
    bundle = json.loads((A.FRONTIER / "bundle.json").read_text())
    keys = sorted((r["market"], r["date"]) for r in bundle["rows"]
                  if r["phase"] == "evaluation" and r["rk"] is not None and np.isfinite(r["rk"]) and r["rk"] > 0)
    markets = np.array([k[0] for k in keys]); dates = np.array([k[1] for k in keys])
    calendar = sorted(set(dates)); population = sorted(set(markets))
    didx = np.array([calendar.index(d) for d in dates]) if False else np.searchsorted(calendar, dates)
    models = {m: load(m) for m in ("volpfn", "loghar", "ms_har", "ttm_rk", "gjr_garch_t")}
    validate_sources(models, bundle)
    ref_rows = models["loghar"][None][1]
    targets = {"r2": np.array([ref_rows[k]["r2"] for k in keys]), "rk": np.array([ref_rows[k]["rk"] for k in keys])}

    def fc(model, seed, proxy):
        c, ev = models[model][seed]
        field = "second_moment" if proxy == "r2" else "variance"
        return c * np.array([ev[k][field] for k in keys])

    losses, weights = {}, {}
    for proxy in ("r2", "rk"):
        y = targets[proxy]
        L = {}
        for m in ("loghar", "ms_har", "ttm_rk", "gjr_garch_t"):
            L[m] = qlike(y, fc(m, None, proxy))
        L["volpfn"] = np.mean([qlike(y, fc("volpfn", s, proxy)) for s in models["volpfn"]], axis=0)
        per_seed = {}
        for other in ("loghar", "ms_har", "ttm_rk"):
            name = f"volpfn+{other}"
            seed_losses = []
            for s in sorted(models["volpfn"]):
                h, wts = blend_series(keys, didx, fc("volpfn", s, proxy), fc(other, None, proxy), y)
                seed_losses.append(qlike(y, h)); weights[f"{proxy}|{name}|{s}"] = wts
            L[name] = np.mean(seed_losses, axis=0); per_seed[name] = seed_losses
        for a_, b_ in (("ms_har", "loghar"), ("gjr_garch_t", "loghar")):
            name = f"{a_}+{b_}"
            h, wts = blend_series(keys, didx, fc(a_, None, proxy), fc(b_, None, proxy), y)
            L[name] = qlike(y, h); weights[f"{proxy}|{name}"] = wts
        losses[proxy] = (L, per_seed)

    # contrasts: reference = volpfn+loghar blend; also each blend vs loghar
    pairs = [("loghar", "volpfn+loghar"), ("ms_har", "volpfn+loghar"), ("ttm_rk", "volpfn+loghar"),
             ("volpfn", "volpfn+loghar"), ("ms_har+loghar", "volpfn+loghar"), ("gjr_garch_t+loghar", "volpfn+loghar"),
             ("loghar", "volpfn+ms_har"), ("loghar", "volpfn+ttm_rk"),
             ("loghar", "ms_har+loghar"), ("loghar", "gjr_garch_t+loghar")]
    result = dict(scope=__doc__, leaderboard={}, contrasts=[], final_weights={})
    for proxy in ("r2", "rk"):
        L, per_seed = losses[proxy]
        result["leaderboard"][proxy] = {m: _equal_market(v, markets)[0] for m, v in L.items()}
    for block in (60, 20, 120):
        counts = _calendar_counts(len(calendar), RESAMPLES, block, SEED)
        rows = []
        for proxy in ("r2", "rk"):
            L, per_seed = losses[proxy]
            for comp, ref in pairs:
                gains = L[comp] - L[ref]
                est, draws, mm = _resample(_panel(dates, markets, gains, calendar, population), counts)
                p = (1 + np.count_nonzero(draws - est >= est)) / (RESAMPLES + 1)
                seeds_pos = None
                if ref in per_seed:
                    seeds_pos = int(sum(_equal_market(L[comp] - sl, markets)[0] > 0 for sl in per_seed[ref]))
                rows.append(dict(block=block, proxy=proxy, comparator=comp, reference=ref, gain=est,
                                 interval95=_basic_interval(est, draws, .95), p_raw=float(p),
                                 markets_improved=int(np.sum(mm > 0)), reference_seeds_positive=seeds_pos))
        for r, ph in zip(rows, holm([r["p_raw"] for r in rows])):
            r["p_holm"] = float(ph); r["family_slots"] = len(rows)
        result["contrasts"] += rows
    for k, v in weights.items():
        result["final_weights"][k] = v[-1]
    (out / "results.json").write_text(json.dumps(result, indent=1, allow_nan=False))
    (out / "weights.json").write_text(json.dumps(weights, allow_nan=False))
    sources = [Path(__file__), Path(A.__file__), A.FRONTIER / "bundle.json"]
    sources.extend(p for model in models for p in sorted((A.FRONTIER / "cells" / model).rglob("forecast.json")))
    (out / "provenance.json").write_text(json.dumps({
        "kind": "fresh_audit_replay_of_exploratory_blends",
        "scientific_settings": {"refit_dates": REFIT, "burn_dates": BURN, "resamples": RESAMPLES, "seed": SEED,
                                "weights": "unconstrained; separately fitted for each proxy and retained seed"},
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    }, indent=2))
    print("LEADERBOARD (calibrated QLIKE; lower better)")
    for proxy in ("r2", "rk"):
        print(proxy, {m: round(v, 5) for m, v in sorted(result["leaderboard"][proxy].items(), key=lambda x: x[1])})
    print("CONTRASTS block 60 (gain = comparator - reference; positive favours reference)")
    for r in result["contrasts"]:
        if r["block"] == 60:
            print(f"{r['proxy']} {r['comparator']:>20s} vs {r['reference']:<14s} {r['gain']:+.5f} "
                  f"[{r['interval95'][0]:+.5f},{r['interval95'][1]:+.5f}] holm={r['p_holm']:.4f} mk={r['markets_improved']}/28 seeds+={r['reference_seeds_positive']}")
    print("final weights on VolPFN (w):", {k: round(v["w"], 3) for k, v in result["final_weights"].items() if "volpfn+loghar" in k})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory; never overwrites historical results")
    main(parser.parse_args().out)
