"""Date parsing and default window utilities."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

from .constants import TimeWindowType
from .errors import ValidationError
from .types import TimeWindow


def parse_date(value: str) -> date:
    """
    Parse a date string into a datetime.date.
    Accepts ISO formats like 'YYYY-MM-DD'. You can extend later if needed.
    """
    try:
        return datetime.fromisoformat(value).date()
    except Exception as e:
        raise ValidationError(
            f"Invalid date '{value}'. Expected ISO format YYYY-MM-DD."
        ) from e


def today_utc_date() -> date:
    # Using UTC date for determinism in logs; you can align to a business timezone later.
    return datetime.utcnow().date()


def compute_yesterday(ref: Optional[date] = None) -> Tuple[date, date]:
    d = ref or today_utc_date()
    y = d - timedelta(days=1)
    return y, y


def compute_last_n_days(n_days: int, ref: Optional[date] = None) -> Tuple[date, date]:
    if n_days <= 0:
        raise ValidationError("n_days must be positive.")
    d = ref or today_utc_date()
    end = d - timedelta(days=1)  # last complete day by default
    start = end - timedelta(days=n_days - 1)
    return start, end


def normalize_time_window(tw: TimeWindow, ref: Optional[date] = None) -> TimeWindow:
    """
    Returns a TimeWindow with concrete start/end for downstream execution.
    Convention: inclusive start and inclusive end.
    """
    if tw.type == TimeWindowType.YESTERDAY:
        s, e = compute_yesterday(ref=ref)
        return replace(tw, start=s, end=e, n_days=1)

    if tw.type == TimeWindowType.LAST_N_DAYS:
        if tw.n_days is None:
            raise ValidationError("TimeWindow LAST_N_DAYS requires n_days.")
        s, e = compute_last_n_days(tw.n_days, ref=ref)
        return replace(tw, start=s, end=e)

    if tw.type == TimeWindowType.RANGE:
        if tw.start is None or tw.end is None:
            raise ValidationError("TimeWindow RANGE requires start and end.")
        if tw.start > tw.end:
            raise ValidationError("TimeWindow RANGE start must be <= end.")
        return tw

    raise ValidationError(f"Unsupported TimeWindowType: {tw.type}")
