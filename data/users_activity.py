from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from cost_agent_mvp.core.errors import ValidationError

_DOW_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def dow_key(d: dt.date) -> str:
    return _DOW_KEYS[d.weekday()]


def month_days(year: int, month: int) -> list[dt.date]:
    n = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, day) for day in range(1, n + 1)]


def sample_biased_day_in_month(
    rng: np.random.Generator,
    year: int,
    month: int,
    *,
    alpha: float,
    activity_cfg: dict[str, Any],
) -> dt.date:
    """
    Sample a day in month with first-half bias (exp decay),
    modulated by weekend + month multipliers.
    """
    days = month_days(year, month)

    dow_mult = activity_cfg["dow_multiplier"]
    month_mult = activity_cfg["month_multiplier"]

    weights = []
    for d in days:
        w = float(np.exp(-alpha * (d.day - 1)))  # front-load month
        w *= float(dow_mult[dow_key(d)])  # weekend dip
        w *= float(month_mult[int(d.month)])  # summer dip
        weights.append(w)

    w = np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        raise ValidationError("Adoption day weights sum to 0. Check multipliers.")
    w = w / w_sum

    return rng.choice(days, p=w)


def make_monotonic_survival(r30: float, r60: float, r90: float) -> tuple[float, float, float]:
    """
    Ensure non-decreasing survival probability with time.
    Real survival curves should not increase over time.
    """
    r60 = max(r60, r30)
    r90 = max(r90, r60)
    return r30, r60, r90


def daily_churn_from_survival(s0: float, s1: float, days: int) -> float:
    """
    Convert survival drop from s0 -> s1 over `days` into per-day churn p.
    Uses constant daily churn within the window.

    If s1 >= s0 => churn=0 (no loss).
    """
    if days <= 0:
        raise ValidationError("days must be positive.")
    if s0 <= 0:
        # nothing survives anyway, but keep safe
        return 1.0
    if s1 >= s0:
        return 0.0

    p_stay = (s1 / s0) ** (1.0 / days)
    p_churn = 1.0 - p_stay
    return float(np.clip(p_churn, 0.0, 1.0))


def sample_churn_date_piecewise(
    rng: np.random.Generator,
    *,
    adopt: dt.date,
    horizon_end: dt.date,
    r30: float,
    r60: float,
    r90: float,
    post90_daily_churn: float,
    trial_days: int = 0,
) -> dt.date | None:
    """
    Sample churn date using piecewise constant daily churn derived from r30/r60/r90.

    Returns:
      - churn date (last active day) OR
      - None if the account survives through horizon_end
    """
    r30, r60, r90 = make_monotonic_survival(r30, r60, r90)

    p0_30 = daily_churn_from_survival(1.0, r30, 30)
    p30_60 = daily_churn_from_survival(r30, r60, 30) if r30 > 0 else 0.0
    p60_90 = daily_churn_from_survival(r60, r90, 30) if r60 > 0 else 0.0
    p90p = float(np.clip(post90_daily_churn, 0.0, 1.0))

    d = adopt
    age = 0
    while d <= horizon_end:
        # during trial: cannot churn
        if age < trial_days:
            d += dt.timedelta(days=1)
            age += 1
            continue

        # churn probabilities after trial
        if age < 30:
            p = p0_30
        elif age < 60:
            p = p30_60
        elif age < 90:
            p = p60_90
        else:
            p = p90p

        if rng.random() < p:
            return d

        d += dt.timedelta(days=1)
        age += 1

    return None


@dataclass(frozen=True)
class ServiceUserState:
    """
    Per-account per-service schedule (adopt + churn) and intensity multiplier.
    """

    adoption_date: dt.date
    churn_date: dt.date | None
    intensity: float


def build_service_user_state(
    accounts: list[int],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, dict[int, ServiceUserState]]:
    """
    Returns:
      state[service][account_id] = ServiceUserState(...)
    For accounts that never adopt a service, the account_id is absent in state[service].
    """
    sm = cfg["services_model"]
    activity_cfg = cfg["activity"]
    trial_days = int(sm.get("trial_days", 0))

    horizon_start = dt.date.fromisoformat(cfg["dataset"]["start_date"])
    horizon_end = dt.date.fromisoformat(cfg["dataset"]["end_date"])

    launch = {
        "tasks": dt.date.fromisoformat(sm["launch_dates"]["tasks"]),
        "classifications": dt.date.fromisoformat(sm["launch_dates"]["classifications"]),
        "amocrm_call": dt.date.fromisoformat(sm["launch_dates"]["amocrm_call"]),
        # If you also want a "dialog" schedule, we can add it later (usually always-on).
    }

    targets = sm["new_adopters_per_month"]  # tasks/classifications/amocrm_call
    alpha = float(sm["adoption_day_bias"]["alpha"])

    retention = sm["retention_targets"]
    post90 = sm["post90_daily_churn"]

    # intensity per service per account (heterogeneity)
    sigma_int = float(sm.get("intensity_lognorm_sigma", 0.6))

    state: dict[str, dict[int, ServiceUserState]] = {
        s: {} for s in ["tasks", "classifications", "amocrm_call"]
    }

    for service in state.keys():
        remaining = set(accounts)

        # schedule adoptions month-by-month, respecting launch date
        for month in range(1, 13):
            month_start = dt.date(2025, month, 1)
            launch_month_start = dt.date(launch[service].year, launch[service].month, 1)
            if month_start < launch_month_start:
                continue
            if not remaining:
                break

            k = int(targets[service])
            k = min(k, len(remaining))
            if k <= 0:
                continue

            chosen = rng.choice(list(remaining), size=k, replace=False)
            for acc in chosen.tolist():
                day = sample_biased_day_in_month(
                    rng, 2025, month, alpha=alpha, activity_cfg=activity_cfg
                )

                if day < launch[service]:
                    day = launch[service]
                if day < horizon_start:
                    day = horizon_start
                if day > horizon_end:
                    continue

                r = retention[service]
                p_convert = float(sm.get("p_convert_after_trial", {}).get(service, 1.0))

                if rng.random() > p_convert:
                    # churn at end of trial (last active day = adopt + trial_days - 1)
                    churn = min(horizon_end, day + dt.timedelta(days=max(trial_days - 1, 0)))
                else:
                    r = retention[service]
                    churn = sample_churn_date_piecewise(
                        rng,
                        adopt=day,
                        horizon_end=horizon_end,
                        r30=float(r["r30"]),
                        r60=float(r["r60"]),
                        r90=float(r["r90"]),
                        post90_daily_churn=float(post90[service]),
                        trial_days=trial_days,
                    )

                # heterogeneity multiplier; median ~ 1.0
                intensity = float(rng.lognormal(mean=0.0, sigma=sigma_int))

                state[service][int(acc)] = ServiceUserState(
                    adoption_date=day,
                    churn_date=churn,
                    intensity=intensity,
                )
                remaining.remove(int(acc))

    return state


def is_active_on_day(s: ServiceUserState, day: dt.date) -> bool:
    """
    True if service is active on this calendar day (inclusive).
    """
    if day < s.adoption_date:
        return False
    if s.churn_date is not None and day > s.churn_date:
        return False
    return True
