"""Dated, outcome-independent data acquisition for the transfer experiment.

Run ``python -m research.population_transfer.data freeze`` before ``acquire``.
The population and data rules are immutable once frozen. No model is fitted here.
Yahoo's dated observations are vendor records, not certified exchange endpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "research/outputs/population_transfer_v1"
STOCKS = "AAPL AMD BAC BBY CVX GE HD JNJ JPM KO LLY MRK MSFT PEP PFE PG RRC UNH WMT XOM".split()
FX = "EUR GBP JPY CHF CAD AUD NZD SEK NOK DKK MXN ZAR BRL INR KRW SGD HKD PLN HUF CZK".split()
DIRECT_FX = {"EUR", "GBP", "AUD", "NZD"}
START = "2005-01-01"
END = "2022-02-25"
END_EXCLUSIVE = "2022-02-26"
DEV_START = "2011-01-01"
EVAL_START = "2015-01-01"
WARMUP = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def population_spec():
    rows = [{"market": s, "population": "stocks", "vendor_symbol": s,
             "vendor_quote": "USD per share", "return_sign": 1,
             "prior_exposure": "Legacy VolPFN scores inspected; reconstructed target differs."}
            for s in STOCKS]
    for c in FX:
        direct = c in DIRECT_FX
        symbol = c + "USD=X" if direct else c + "=X"
        rows.append({"market": c, "population": "fx", "vendor_symbol": symbol,
                     "vendor_quote": "USD per " + c if direct else c + " per USD",
                     "canonical_quote": "USD per " + c, "return_sign": 1 if direct else -1,
                     "prior_exposure": ("Legacy VolPFN saved scores located."
                        if c in {"AUD", "BRL"} else
                        "No saved VolPFN scores located in targeted inventory; absence does not prove untouched data.")})
    return {"version": 1, "stocks_primary": True, "fx_secondary_target_change": True,
            "source": "Yahoo Finance through yfinance", "start_inclusive": START,
            "end_inclusive": END, "vendor_end_exclusive": END_EXCLUSIVE,
            "development_start": DEV_START, "evaluation_start": EVAL_START,
            "warmup_observations": WARMUP,
            "coverage_minimum_after_warmup": {"predevelopment": 200, "development": 500, "evaluation": 500},
            "request": {"interval": "1d", "auto_adjust": False, "back_adjust": False,
                        "repair": False, "actions": True, "keepna": True, "rounding": False},
            "stock_target": "100 * log(Close / Open), vendor same-basis OHLC; no cash-dividend or total-return adjustment.",
            "fx_target": "return_sign * 100 * log(Close_t / Close_previous_observed_row); calendar gaps retained and reported.",
            "quality_rules": [
                "Preserve the vendor date label without UTC conversion; sort dates; duplicate dates fail the series.",
                "Reject rows with nonfinite or nonpositive Open/High/Low/Close, or inconsistent OHLC order (tolerance 1e-8 relative).",
                "Stock returns use valid current OHLC rows. FX needs valid current and immediately preceding raw OHLC rows; no bridging invalid rows.",
                "Keep exactly zero returns. No future fills, clipping, price repairs, outcome filters, ticker replacements, or unsigned legacy reconstruction.",
                "Retain all forty inventory entries including failures; insufficient coverage is unavailable, not replaced.",
                "The first 100 valid returns initialize forecasts and are excluded from scoring and coverage minima."],
            "endpoint_limitation": "Daily vendor dates and OHLC are not independent proof of historical exchange endpoint or delivery alignment.",
            "members": rows}


def freeze_population(out=DEFAULT_OUT):
    out = Path(out)
    path = out / "population.json"
    digest = out / "population.json.sha256"
    spec = population_spec()
    if path.exists():
        if not digest.exists() or digest.read_text().strip() != sha(path):
            raise ValueError("Population manifest hash mismatch")
        if json.loads(path.read_text()) != spec:
            raise ValueError("Population specification differs from the frozen manifest")
    else:
        if any((out / d).exists() for d in ("raw", "data")):
            raise ValueError("Cannot first freeze a population after raw/data artifacts exist")
        save_json(path, spec)
        digest.write_text(sha(path) + "\n")
    return spec


def prepare_observations(frame, member):
    """Return dated percent returns, explicit row exclusions, and quality counts."""
    required = ["Open", "High", "Low", "Close"]
    if frame.empty:
        raise ValueError("empty_download")
    if not all(c in frame.columns for c in required):
        raise ValueError("missing_ohlc_columns")
    frame = frame.copy()
    # strftime on an aware index retains its local/vendor calendar date.
    labels = np.asarray(pd.DatetimeIndex(frame.index).strftime("%Y-%m-%d"), dtype="U10")
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("duplicate_vendor_dates")
    order = np.argsort(labels)
    frame, labels = frame.iloc[order], labels[order]
    inside = (labels >= START) & (labels <= END)
    frame, labels = frame.iloc[np.flatnonzero(inside)], labels[inside]
    prices = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(prices).all(axis=1)
    positive = (prices > 0).all(axis=1)
    o, h, l, c = prices.T
    tolerance = 1e-8 * np.maximum(np.max(np.abs(prices), axis=1), 1.)
    ordered = (h + tolerance >= np.maximum(o, c)) & (l - tolerance <= np.minimum(o, c)) & (h + tolerance >= l)
    valid = finite & positive & ordered
    reason = np.full(len(labels), "", dtype=object)
    reason[~ordered] = "inconsistent_ohlc"
    reason[~positive] = "nonpositive_ohlc"
    reason[~finite] = "nonfinite_ohlc"
    r = np.full(len(labels), np.nan)
    gaps = np.zeros(len(labels), dtype=int)
    if member["population"] == "stocks":
        r[valid] = 100. * np.log(c[valid] / o[valid])
    else:
        dates = labels.astype("datetime64[D]")
        gaps[1:] = np.diff(dates).astype(int)
        pair_valid = valid.copy()
        pair_valid[0] = False
        pair_valid[1:] &= valid[:-1]
        reason[valid & ~pair_valid] = "missing_valid_previous_raw_row"
        ix = np.flatnonzero(pair_valid)
        r[ix] = member["return_sign"] * 100. * np.log(c[ix] / c[ix - 1])
        valid = pair_valid
    valid &= np.isfinite(r)
    reason[(reason == "") & ~valid] = "nonfinite_return"
    exclusions = [{"date": str(date), "reason": str(code)} for date, code in zip(labels[~valid], reason[~valid])]
    result = {"date": labels[valid], "r": r[valid], "population": np.asarray(member["population"]),
              "calendar_gap_days": gaps[valid]}
    score_eligible = np.arange(valid.sum()) >= WARMUP
    dates = result["date"]
    counts = {"predevelopment": int(np.sum(score_eligible & (dates < DEV_START))),
              "development": int(np.sum(score_eligible & (dates >= DEV_START) & (dates < EVAL_START))),
              "evaluation": int(np.sum(score_eligible & (dates >= EVAL_START) & (dates <= END)))}
    failures = [k for k, minimum in population_spec()["coverage_minimum_after_warmup"].items() if counts[k] < minimum]
    quality = {"raw_rows_in_range": len(labels), "valid_returns": int(valid.sum()),
               "zero_returns_retained": int(np.sum(result["r"] == 0)), "excluded_rows": len(exclusions),
               "counts_after_warmup": counts, "coverage_failures": failures,
               "first_date": str(dates[0]) if len(dates) else None,
               "last_date": str(dates[-1]) if len(dates) else None,
               "multi_day_return_intervals": int(np.sum(gaps[valid] > 1)),
               "maximum_calendar_gap_days": int(gaps[valid].max()) if valid.any() else None}
    return result, exclusions, quality


def write_snapshot(path, frame):
    """Deterministic gzip containing all returned columns and vendor timestamps."""
    content = frame.to_csv(index=True, index_label="vendor_timestamp", float_format="%.17g").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
            zipped.write(content)


def index_inventory():
    entries = []
    for path in sorted((ROOT / "research/outputs/session_information_pilot_v1/forecasts").glob("*.npz")):
        entries.append({"market": path.stem, "path": str(path), "sha256": sha(path),
                        "role": "Read-only dated index returns for the matched equity-index fit; cached model forecasts are not fitting inputs."})
    return entries


def acquire(out=DEFAULT_OUT):
    import yfinance as yf
    out = Path(out)
    spec = freeze_population(out)
    cache = out / "vendor_cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    inventory_path = out / "inventory.json"
    if inventory_path.exists():
        raise FileExistsError("Acquisition inventory already exists; no silent re-pull or overwrite")
    inventory = {"population_sha256": sha(out / "population.json"), "data_code_sha256": sha(__file__),
                 "retrieval_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "versions": {p: importlib.metadata.version(p) for p in ["yfinance", "pandas", "numpy"]},
                 "status": "acquiring", "members": [], "index_sources": index_inventory()}
    # Write the inventory before the first request, and after each completed series.
    save_json(inventory_path, inventory)
    for member in spec["members"]:
        name = member["market"]
        entry = dict(member)
        entry["status"] = "unavailable"
        print("DATA %s %s: retrieving" % (member["population"], name), flush=True)
        try:
            ticker = yf.Ticker(member["vendor_symbol"])
            frame = ticker.history(start=START, end=END_EXCLUSIVE, timeout=30, raise_errors=True, **spec["request"])
            if frame.empty:
                raise ValueError("empty_download")
            raw = out / "raw" / (name + ".csv.gz")
            write_snapshot(raw, frame)
            metadata = ticker.get_history_metadata()
            save_json(out / "raw" / (name + ".metadata.json"), {
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": member["vendor_symbol"], "request": spec["request"],
                "history_metadata": metadata, "population_sha256": inventory["population_sha256"]})
            entry.update(raw_path=str(raw), raw_sha256=sha(raw),
                         metadata_sha256=sha(out / "raw" / (name + ".metadata.json")))
            arrays, excluded, quality = prepare_observations(frame, member)
            entry.update(quality)
            save_json(out / "raw" / (name + ".excluded.json"), excluded)
            entry["exclusions_sha256"] = sha(out / "raw" / (name + ".excluded.json"))
            if quality["coverage_failures"]:
                entry["failure_code"] = "insufficient_coverage"
            else:
                destination = out / "data" / (name + ".npz")
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **arrays)
                entry.update(status="available", data_path=str(destination), data_sha256=sha(destination))
        except Exception as exc:
            entry.update(failure_code=type(exc).__name__ + ":" + str(exc), exception_type=type(exc).__name__)
        inventory["members"].append(entry)
        save_json(inventory_path, inventory)
        print("DATA %s: %s %s" % (name, entry["status"], entry.get("failure_code", entry.get("counts_after_warmup"))), flush=True)
    inventory["status"] = "complete"
    inventory["retrieval_completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    inventory["available_per_population"] = {p: sum(m["population"] == p and m["status"] == "available" for m in inventory["members"]) for p in ["stocks", "fx"]}
    save_json(inventory_path, inventory)
    (out / "inventory.json.sha256").write_text(sha(inventory_path) + "\n")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "acquire"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze_population(args.out)
        print("Frozen population: " + str(args.out / "population.json"))
    else:
        result = acquire(args.out)
        print(json.dumps(result["available_per_population"]))


if __name__ == "__main__":
    main()
