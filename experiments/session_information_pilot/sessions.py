"""Session availability and cross-market features without inferred calendars.

This module consumes an independently established UTC session table.  It does
not turn the Oxford--Man ``open_time``/``close_time`` numbers into session times:
the local archive does not establish those fields' timezone or exchange-session
semantics.  A valid timestamp schema is not evidence of historical availability.

The latest and calendar-lagged features use the *same donor markets*.  A donor
enters both only when each variant has a usable session in the age window.  This
prevents a timing contrast from also being a donor-coverage contrast.  Missing
features are zero with explicit count flags, never a forward-filled surprise.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd


SESSION_COLUMNS = ("market", "date", "open_utc", "close_utc", "available_utc")
FEATURE_COLUMNS = (
    "market", "date", "forecast_utc", "latest_x", "lagged_x",
    "timing_increment", "donor_count", "latest_eligible_count",
    "lagged_eligible_count", "new_session_count", "latest_max_available_utc",
    "donor_records_json",
)


def _utc(value, field):
    """Require explicit, zero-offset timestamps; never guess a timezone."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: invalid timestamp {value!r}") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{field}: an explicit timezone-aware UTC timestamp is required")
    if stamp.utcoffset() != timedelta(0):
        raise ValueError(f"{field}: convert the established timestamp to UTC explicitly")
    return stamp.tz_convert("UTC")


def _session_date(value):
    # A trading date is an external label, not the date after UTC conversion.
    if isinstance(value, datetime):
        raise ValueError("date: use a local session date, not a timestamp")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date: an ISO YYYY-MM-DD local session label is required")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"date: invalid session label {value!r}") from exc
    return value


def _market(value):
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError("market: a non-empty identifier without surrounding spaces is required")
    return value


@dataclass(frozen=True)
class Session:
    """One externally verified session, including the data availability time."""

    market: str
    date: str
    open_utc: pd.Timestamp
    close_utc: pd.Timestamp
    available_utc: pd.Timestamp
    surprise: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "market", _market(self.market))
        object.__setattr__(self, "date", _session_date(self.date))
        for field in ("open_utc", "close_utc", "available_utc"):
            object.__setattr__(self, field, _utc(getattr(self, field), field))
        if self.close_utc <= self.open_utc:
            raise ValueError("close_utc must be strictly after open_utc")
        if self.available_utc < self.close_utc:
            raise ValueError("available_utc cannot precede close_utc")

    def as_record(self):
        return asdict(self)


def _validate_keys(frame):
    out = frame.copy()
    if not {"market", "date", "open_utc"}.issubset(out.columns):
        raise ValueError("Required target columns: market, date, open_utc")
    out["market"] = out["market"].map(_market)
    out["date"] = out["date"].map(_session_date)
    if out.duplicated(["market", "date"]).any():
        raise ValueError("Duplicate market/date session keys")
    out["open_utc"] = pd.to_datetime(
        [_utc(value, "open_utc") for value in out["open_utc"]], utc=True
    )
    return out


def validate_sessions(frame):
    """Validate an explicit UTC table, preserving additional data columns.

    Callers must already have established dates, actual session boundaries and
    the availability buffer.  Neither offsets nor a buffer are inferred here.
    No assertion is made about source provenance by this schema validation.
    """
    missing = set(SESSION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing session columns: {sorted(missing)}")
    out = _validate_keys(frame)
    for field in ("close_utc", "available_utc"):
        out[field] = pd.to_datetime([_utc(value, field) for value in out[field]], utc=True)
    if (out["close_utc"] <= out["open_utc"]).any():
        raise ValueError("close_utc must be strictly after open_utc")
    if (out["available_utc"] < out["close_utc"]).any():
        raise ValueError("available_utc cannot precede close_utc")
    return out


def build_session_features(
    sessions,
    targets=None,
    *,
    donor_markets: Optional[Sequence[str]] = None,
    surprise_column="surprise",
    max_age_hours=120.0,
    clip=3.0,
):
    """Return matched-donor latest and calendar-lagged features.

    ``sessions`` requires market/date/open_utc/close_utc/available_utc and a
    numeric surprise column.  Missing surprises are permitted and excluded.
    ``targets`` optionally specifies only market/date/open_utc, allowing scoring
    on a subset while drawing donor records from the full supplied history.

    All records satisfy available_utc < target open and age since donor close
    <= max_age_hours.  A lagged record additionally has its supplied local
    session label < the target's supplied local session label.  For each donor,
    the most recently *closed* eligible session is selected.  The most recently
    delivered record is not necessarily the most recently closed session.

    The two means average clipped surprises over the same donor intersection;
    an empty intersection gives zeros and donor_count=0.  ``new_session_count``
    counts donors whose selected session differs between the two variants.
    Extra eligibility counts expose observations excluded by the matching rule.
    Metadata records every contributing date and availability time for audit.
    """
    if not np.isfinite(max_age_hours) or max_age_hours <= 0:
        raise ValueError("max_age_hours must be finite and positive")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    source = validate_sessions(sessions)
    if surprise_column not in source.columns:
        raise ValueError(f"Missing surprise column: {surprise_column}")
    # Missing donor labels are absence of information, not an imputed surprise.
    surprise = pd.to_numeric(source[surprise_column], errors="raise").to_numpy(dtype=float)
    usable = np.isfinite(surprise)
    source = source.loc[usable].copy()
    source["_surprise"] = np.clip(surprise[usable], -clip, clip)
    target = _validate_keys(sessions if targets is None else targets)
    if donor_markets is None:
        names = sorted(sessions["market"].unique())
    else:
        names = sorted({_market(name) for name in donor_markets})
    banks = {}
    for name, rows in source.groupby("market", sort=False):
        rows = rows.sort_values(["close_utc", "date"], kind="stable")
        banks[name] = {
            "close": np.array([stamp.value for stamp in rows["close_utc"]], dtype=np.int64),
            "available": np.array([stamp.value for stamp in rows["available_utc"]], dtype=np.int64),
            "dates": rows["date"].to_numpy(),
            "surprise": rows["_surprise"].to_numpy(dtype=float),
        }
    max_age_ns = int(pd.Timedelta(hours=float(max_age_hours)).value)
    result = []
    for row in target.itertuples(index=False):
        now, day = row.open_utc.value, row.date
        newest, lagged = {}, {}
        for name in names:
            if name == row.market or name not in banks:
                continue
            bank = banks[name]
            # Strict close cutoff follows from available >= close and available < now.
            index = int(np.searchsorted(bank["close"], now, side="left")) - 1
            while index >= 0 and bank["close"][index] >= now - max_age_ns:
                if bank["available"][index] < now:
                    if name not in newest:
                        newest[name] = index
                    if bank["dates"][index] < day:
                        lagged[name] = index
                        break
                index -= 1
        shared = sorted(newest.keys() & lagged.keys())
        latest_values, lagged_values, details, availabilities = [], [], [], []
        changed = 0
        for name in shared:
            bank = banks[name]
            new_index, old_index = newest[name], lagged[name]
            latest_values.append(bank["surprise"][new_index])
            lagged_values.append(bank["surprise"][old_index])
            changed += int(new_index != old_index)
            availabilities.append(bank["available"][new_index])
            details.append({
                "market": name,
                "latest_date": bank["dates"][new_index],
                "lagged_date": bank["dates"][old_index],
                "latest_close_utc": pd.Timestamp(bank["close"][new_index], tz="UTC").isoformat(),
                "latest_available_utc": pd.Timestamp(bank["available"][new_index], tz="UTC").isoformat(),
                "lagged_close_utc": pd.Timestamp(bank["close"][old_index], tz="UTC").isoformat(),
                "lagged_available_utc": pd.Timestamp(bank["available"][old_index], tz="UTC").isoformat(),
                "latest_surprise": float(bank["surprise"][new_index]),
                "lagged_surprise": float(bank["surprise"][old_index]),
            })
        latest_x = float(np.mean(latest_values)) if shared else 0.0
        lagged_x = float(np.mean(lagged_values)) if shared else 0.0
        result.append({
            "market": row.market, "date": day, "forecast_utc": row.open_utc.isoformat(),
            "latest_x": latest_x, "lagged_x": lagged_x,
            "timing_increment": latest_x - lagged_x,
            "donor_count": len(shared), "latest_eligible_count": len(newest),
            "lagged_eligible_count": len(lagged), "new_session_count": changed,
            "latest_max_available_utc": (
                pd.Timestamp(max(availabilities), tz="UTC").isoformat() if availabilities else None
            ),
            "donor_records_json": json.dumps(details, sort_keys=True, allow_nan=False),
        })
    return pd.DataFrame(result, columns=FEATURE_COLUMNS)
