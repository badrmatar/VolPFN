"""Strict, read-only loading for the retained historical tail study.

This support module does not replace or monkeypatch the frozen sibling evaluator.
It reproduces that evaluator's percent-return and unscored EWMA-fill conventions,
while refusing misaligned inputs or invalid variance on the declared scored split.
"""
from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
import argparse
import hashlib
import json
from typing import Optional

import numpy as np
from scipy.signal import lfilter


class TailInputError(ValueError):
    """An input cannot safely represent the declared historical evaluation."""


@dataclass(frozen=True)
class TailInputs:
    date: np.ndarray
    returns_percent: np.ndarray
    split: np.ndarray
    scored: np.ndarray
    variance: np.ndarray
    unscored_fallback: np.ndarray
    audit: dict


def _vector(archive, name, path):
    if name not in archive:
        raise TailInputError(f"{path}: missing array {name!r}")
    value = archive[name]
    if value.ndim != 1 or not len(value):
        raise TailInputError(f"{path}: {name!r} must be a nonempty one-dimensional array")
    return value


def _numeric(value, name, path):
    if value.dtype.kind not in 'fiu':
        raise TailInputError(f"{path}: {name!r} must be a numerical array")
    return value.astype(float, copy=True)


def _dates(value, path):
    if value.dtype.kind not in 'US':
        raise TailInputError(f"{path}: date must contain ISO calendar-date strings")
    dates = value.astype(str)
    for item in dates:
        try:
            parsed = calendar_date.fromisoformat(item)
        except ValueError as exc:
            raise TailInputError(f"{path}: invalid ISO calendar date {item!r}") from exc
        if parsed.isoformat() != item:
            raise TailInputError(f"{path}: date {item!r} is not canonical YYYY-MM-DD")
    if np.any(dates[1:] <= dates[:-1]):
        raise TailInputError(f"{path}: dates must be strictly increasing and unique")
    return dates


def _historical_ewma(returns_percent):
    """Exact market.py:80-82 convention, including its unscored initialization."""
    r = returns_percent
    x = np.empty(len(r))
    x[0] = max(float(np.mean(r[:10] ** 2)), 1e-4)
    x[1:] = (1 - .94) * r[:-1] ** 2
    return lfilter([1.0], [1.0, -.94], x)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tail_inputs(market_file, extract_file, *, scored_split,
                     fallback_variance: Optional[np.ndarray] = None):
    """Validate and load one explicitly declared historical tail cohort.

    The market stores decimal returns; the extract's ``r`` stores percentage
    returns. Exact equality after conversion is required because these retained
    extracts were produced from the same saved arrays. Invalid ``h`` is permitted
    only outside ``scored_split`` and filled exactly as in the frozen evaluator.
    The supplied/default fallback is never used for a scored variance.
    """
    if scored_split not in (1, 2):
        raise TailInputError('scored_split must explicitly select development (1) or later (2)')
    market_file, extract_file = Path(market_file), Path(extract_file)
    if not market_file.is_file() or not extract_file.is_file():
        raise TailInputError('Both market and neural-extract files must exist; omission is not a fallback')
    with np.load(market_file, allow_pickle=False) as market:
        dates = _dates(_vector(market, 'date', market_file), market_file)
        returns = _numeric(_vector(market, 'returns', market_file), 'returns', market_file) * 100.
        split = _vector(market, 'split', market_file)
        if split.dtype.kind not in 'iu' or not np.isin(split, [0, 1, 2]).all():
            raise TailInputError(f'{market_file}: split must contain only integer labels 0, 1, 2')
        split = split.astype(int, copy=True)
    if len(returns) != len(dates) or len(split) != len(dates):
        raise TailInputError(f'{market_file}: date, returns and split lengths differ')
    if not np.isfinite(returns).all():
        raise TailInputError(f'{market_file}: nonfinite percentage returns')
    scored = split == scored_split
    if not scored.any():
        raise TailInputError(f'{market_file}: declared scored split {scored_split} has no observations')
    with np.load(extract_file, allow_pickle=False) as extract:
        extracted_dates = _dates(_vector(extract, 'date', extract_file), extract_file)
        extracted_returns = _numeric(_vector(extract, 'r', extract_file), 'r', extract_file)
        variance = _numeric(_vector(extract, 'h', extract_file), 'h', extract_file)
    if not (len(extracted_dates) == len(extracted_returns) == len(variance) == len(dates)):
        raise TailInputError(f'{extract_file}: neural extract and market array lengths differ')
    if not np.array_equal(extracted_dates, dates):
        raise TailInputError(f'{extract_file}: neural extract dates do not match the market calendar')
    if not np.isfinite(extracted_returns).all() or not np.array_equal(extracted_returns, returns):
        raise TailInputError(f'{extract_file}: neural extract percentage returns do not match market returns')
    invalid = ~np.isfinite(variance) | (variance <= 0)
    if np.any(invalid & scored):
        first = int(np.flatnonzero(invalid & scored)[0])
        raise TailInputError(f'{extract_file}: invalid scored neural variance at {dates[first]} '
                             f'(index {first}); scored fallback is forbidden')
    fallback = (_historical_ewma(returns) if fallback_variance is None
                else np.asarray(fallback_variance, dtype=float))
    if fallback.shape != variance.shape or not np.isfinite(fallback).all() or np.any(fallback <= 0):
        raise TailInputError('Fallback variance must have matching shape and be finite and strictly positive')
    variance[invalid] = fallback[invalid]
    audit = dict(market_file=str(market_file.resolve()), extract_file=str(extract_file.resolve()),
        market_sha256=_sha(market_file), extract_sha256=_sha(extract_file),
        scored_split=int(scored_split), total_rows=len(dates), scored_rows=int(scored.sum()),
        first_scored_date=str(dates[scored][0]), last_scored_date=str(dates[scored][-1]),
        date_alignment='exact', percentage_return_alignment='exact',
        scored_fallback_rows=0, unscored_fallback_rows=int(invalid.sum()),
        fallback='frozen EWMA lambda .94' if fallback_variance is None else 'caller-supplied validated path',
        evaluator_connection='support loader only; frozen historical evaluator is unchanged')
    for value in (dates, returns, split, scored, variance, invalid):
        value.setflags(write=False)
    return TailInputs(dates, returns, split, scored, variance, invalid, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market-file', required=True, type=Path)
    parser.add_argument('--extract-file', required=True, type=Path)
    parser.add_argument('--scored-split', required=True, type=int, choices=(1, 2))
    parser.add_argument('--report', type=Path, help='Optional new JSON receipt; existing files are never replaced')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(f'Preserve existing receipt: {args.report}')
    result = load_tail_inputs(args.market_file, args.extract_file, scored_split=args.scored_split)
    rendered = json.dumps(result.audit, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
