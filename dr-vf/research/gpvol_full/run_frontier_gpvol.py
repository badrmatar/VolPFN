"""Run full GP-Vol on the frozen frontier-extension origins (addendum; the frozen run is untouched).

Reads research/outputs/frontier_extension_20260926_v1/bundle.json for origins,
targets and market files, and writes forecast cells in the same row format to
research/outputs/frontier_gpvol_20260926_v1/cells/<model>/<seed>/<market>/.

Per market and phase (evaluation, and calibration for FTSE/N225/SPX) one RAPCF
chain is run over the contiguous session block from `burn_in` sessions before
the first origin to the last origin, so every intervening session is filtered.
Returns are standardised with the mean and sd of the 1,000 sessions before the
first origin of the block (causal; the paper used full-sample standardisation).

usage:
  python3 run_frontier_gpvol.py --procs 4 --threads 2            # all markets
  python3 run_frontier_gpvol.py --markets SPX FTSE --seed 1
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRONTIER = ROOT / "research/outputs/frontier_extension_20260926_v1"
DEFAULT_OUT = ROOT / "research/outputs/frontier_gpvol_20260926_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan():
    bundle = json.loads((FRONTIER / "bundle.json").read_text())
    rows = {}
    for r in bundle["rows"]:
        rows.setdefault(r["market"], []).append(r)
    return bundle, rows


def claim(cell, kind):
    """Atomically claim a market cell for one runner kind ('native' or 'chunk'). Returns holder."""
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / "claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(kind + " " + datetime.now(timezone.utc).isoformat() + "\n")
        return kind
    except FileExistsError:
        return path.read_text().split()[0]


def prepare_segments(market, rows, market_path, settings):
    import numpy as np
    d = np.load(ROOT / market_path, allow_pickle=True)
    r = d["r"].astype(float)
    dates = d["date"].astype(str)
    rk = d["rk"].astype(float)
    segs = []
    for phase in ("calibration", "evaluation"):
        prow = sorted((x for x in rows if x["phase"] == phase), key=lambda x: x["index"])
        if not prow:
            continue
        idx = np.array([x["index"] for x in prow])
        for x in prow:
            assert dates[x["index"]] == x["date"], (market, x)
            assert abs(r[x["index"]] ** 2 - x["r2"]) <= 1e-9 * max(1, x["r2"]), (market, x)
            if x["rk"] is not None and np.isfinite(x["rk"]):
                assert abs(rk[x["index"]] - x["rk"]) <= 1e-9 * max(1, x["rk"]), (market, x)
        first, last = int(idx.min()), int(idx.max())
        B = settings["burn_in"]
        start = first - B
        if start < 0:
            raise ValueError(f"{market} {phase}: not enough history for burn-in")
        std_lo = max(0, first - settings["standardise_window"])
        mu_hat = float(np.mean(r[std_lo:first]))
        sd_hat = float(np.std(r[std_lo:first], ddof=1))
        seg_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}|{market}|{phase}".encode()).digest()[:4], "little")
        segs.append(dict(phase=phase, prow=prow, first=first, last=last, start=start, std_lo=std_lo,
                         mu_hat=mu_hat, sd_hat=sd_hat, seg_seed=seg_seed,
                         x=(r[start:last + 1] - mu_hat) / sd_hat, start_date=str(dates[start])))
    return segs


def make_filter(seg, settings):
    sys.path.insert(0, str(HERE))
    import gpvol_rapcf as G
    return G.RAPCF(seg["x"], n_particles=settings["n_particles"], burn_in=settings["burn_in"], lam=settings["lam"],
                   seed=seg["seg_seed"], code_indexing=settings["code_indexing"], chunk=settings["chunk"])


def finalize(market, segs, outs, seconds, settings, out_dir, market_path, runner):
    import numpy as np
    sys.path.insert(0, str(HERE))
    import gpvol_rapcf as G
    model, seed = settings["model"], settings["seed"]
    cell = out_dir / "cells" / model / str(seed) / market
    result_rows, audit_segments, diag = [], [], {}
    B = settings["burn_in"]
    for seg in segs:
        out = outs[seg["phase"]]
        mu_hat, sd_hat, start = seg["mu_hat"], seg["sd_hat"], seg["start"]
        for xrow in seg["prow"]:
            t = xrow["index"] - start
            var = sd_hat ** 2 * float(out["h"][t])
            ok = bool(np.isfinite(var) and var > 0)
            result_rows.append(dict(xrow, model=model, seed=seed, mean=mu_hat, variance=var,
                                    second_moment=var + mu_hat ** 2, substituted=False, floor_applied=False,
                                    forecast_failed=not ok,
                                    log_score=float(out["logdens"][t] - np.log(sd_hat)),
                                    ess_first=float(out["ess_first"][t]), ess_second=float(out["ess_second"][t]),
                                    gp_failures=int(out["gp_failures"][t])))
        steps = slice(B, len(seg["x"]))
        audit_segments.append(dict(
            phase=seg["phase"], first_origin=seg["first"], last_origin=seg["last"], block_start=start,
            block_start_date=seg["start_date"], steps_filtered=len(seg["x"]) - B, origins=len(seg["prow"]),
            standardise=dict(window_start=seg["std_lo"], window_end_exclusive=seg["first"], mean=mu_hat, sd=sd_hat),
            rng_seed=seg["seg_seed"], seconds=round(seconds.get(seg["phase"], float("nan")), 1),
            ess_first_min=float(np.nanmin(out["ess_first"][steps])),
            ess_first_median=float(np.nanmedian(out["ess_first"][steps])),
            ess_second_median=float(np.nanmedian(out["ess_second"][steps])),
            gp_failures_total=int(np.nansum(out["gp_failures"][steps])),
            weight_collapse_steps=int(np.nansum(out["weight_collapse"][steps] > 0)),
            final_param_mean=dict(zip(G.PARAM_NAMES, np.round(out["param_mean"][-1], 5).tolist())),
            final_param_sd=dict(zip(G.PARAM_NAMES, np.round(out["param_sd"][-1], 5).tolist()))))
        for k in ("h", "logdens", "apf_loglik", "ess_first", "ess_second", "gp_failures", "weight_collapse",
                  "param_mean", "param_sd", "v_mean"):
            diag[f"{seg['phase']}__{k}"] = out[k]
        diag[f"{seg['phase']}__block_start"] = np.array(start)
    np.savez_compressed(cell / "diagnostics.npz", **diag)
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), rows=result_rows,
                   audit=dict(model=model, seed=seed, market=market, settings=settings, runner=runner,
                              segments=audit_segments, elapsed_seconds=round(sum(seconds.values()), 1),
                              source=dict(gpvol_rapcf_sha256=sha(HERE / "gpvol_rapcf.py"),
                                          runner_sha256=sha(Path(__file__)),
                                          market_file=market_path, market_file_sha256=sha(ROOT / market_path),
                                          bundle_sha256=sha(FRONTIER / "bundle.json"))))
    tmp = cell / "forecast.json.tmp"
    tmp.write_text(json.dumps(payload, allow_nan=False))
    tmp.rename(cell / "forecast.json")
    (cell / "forecast.sha256").write_text(sha(cell / "forecast.json") + "\n")
    return payload


def run_market(market, rows, market_path, settings, out_dir):
    model, seed = settings["model"], settings["seed"]
    cell = out_dir / "cells" / model / str(seed) / market
    target = cell / "forecast.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old["audit"]["settings"] == settings:
            return {"market": market, "status": "exists"}
        raise RuntimeError(f"{target} exists with different settings; use a new --out or model name")
    holder = claim(cell, "native")
    if holder != "native":
        return {"market": market, "status": f"claimed_by_{holder}"}
    segs = prepare_segments(market, rows, market_path, settings)
    outs, seconds = {}, {}
    for seg in segs:
        t0 = time.time()
        outs[seg["phase"]] = make_filter(seg, settings).run().result()
        seconds[seg["phase"]] = time.time() - t0
    payload = finalize(market, segs, outs, seconds, settings, out_dir, market_path, "native")
    return {"market": market, "status": "done", "seconds": payload["audit"]["elapsed_seconds"]}


def settings_from(a):
    model = "gpvol_codeidx" if a.code_indexing else "gpvol"
    return dict(model=model, seed=a.seed, n_particles=a.n_particles, burn_in=a.burn_in, lam=a.lam,
                standardise_window=a.standardise_window, code_indexing=a.code_indexing, chunk=a.chunk,
                prior="N(init, I), init=[0.98,-0.1,0,0,0,log 0.1] (authors' demo.m)",
                resampling="systematic", jitter="diagonal unweighted sd * sqrt(1-lam^2)")


def _worker(args):
    market, rows, path, settings, out_dir, threads = args
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        return run_market(market, rows, path, settings, Path(out_dir))
    except Exception as exc:  # recorded, never silently dropped
        return {"market": market, "status": "failed", "error": repr(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--markets", nargs="*")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--standardise-window", type=int, default=1000)
    p.add_argument("--code-indexing", action="store_true", help="diagnostic: reproduce released-code test input")
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    bundle, rows = load_plan()
    markets = a.markets or bundle["markets"]
    # largest blocks first so parallel workers finish together
    markets = sorted(markets, key=lambda m: -len(rows[m]))
    settings = settings_from(a)
    model = settings["model"]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, rows[m], bundle["market_paths"][m], settings, str(a.out), a.threads) for m in markets]
    log = a.out / f"run_{model}_seed{a.seed}.log"
    t0 = time.time()
    if a.procs > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.procs) as pool:
            for res in pool.imap_unordered(_worker, jobs):
                res["wall"] = round(time.time() - t0, 1)
                print(json.dumps(res), flush=True)
                with log.open("a") as f:
                    f.write(json.dumps(res) + "\n")
    else:
        for job in jobs:
            res = _worker(job)
            res["wall"] = round(time.time() - t0, 1)
            print(json.dumps(res), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
