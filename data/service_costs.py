from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from cost_agent_mvp.core.errors import ValidationError
from data.users_activity import ServiceUserState


def _nbinom_sample(rng: np.random.Generator, mean: float, k: float) -> int:
    """
    Sample from Negative Binomial using mean/dispersion parameterization.
    Var = mean + mean^2 / k. Larger k => closer to Poisson.
    """
    mean = float(max(mean, 0.0))
    k = float(max(k, 1e-6))
    if mean == 0.0:
        return 0
    # NB with parameters (n=k, p=k/(k+mean))
    p = k / (k + mean)
    return int(rng.negative_binomial(n=k, p=p))


def _sample_days_weighted(
    rng: np.random.Generator,
    days: list[dt.date],
    weights: np.ndarray,
    k: int,
) -> set[dt.date]:
    if k <= 0 or not days:
        return set()
    k = min(k, len(days))
    w = weights.astype(float)
    s = float(w.sum())
    if s <= 0:
        # fallback: uniform
        idx = rng.choice(len(days), size=k, replace=False)
    else:
        w = w / s
        idx = rng.choice(len(days), size=k, replace=False, p=w)
    return {days[int(i)] for i in np.asarray(idx).tolist()}


def build_usage_calendar(
    accounts: list[int],
    cfg: dict[str, Any],
    user_state: dict[str, dict[int, ServiceUserState]],
    horizon_start: dt.date,
    horizon_end: dt.date,
    rng: np.random.Generator,
) -> dict[tuple[int, dt.date], set[str]]:
    """
    Precompute which account uses which services on which days.

    Returns:
      calendar[(account_id, day)] = {"tasks", "classifications", ...}
    """
    usage_cfg = cfg["services_usage_days"]
    activity_cfg = cfg["activity"]
    print("services_usage_days.mean_active_days =", usage_cfg.get("mean_active_days"))
    print("services_usage_days.dispersion_k =", usage_cfg.get("dispersion_k"))

    def season_weight(day: dt.date) -> float:
        dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day.weekday()]
        return float(activity_cfg["dow_multiplier"][dow]) * float(
            activity_cfg["month_multiplier"][int(day.month)]
        )

    # Precompute all days in horizon
    all_days = []
    d = horizon_start
    while d <= horizon_end:
        all_days.append(d)
        d += dt.timedelta(days=1)

    calendar_map: dict[tuple[int, dt.date], set[str]] = {}

    # Services to schedule via user_state
    for service in ["tasks", "classifications", "amocrm_call"]:
        mean_days = float(usage_cfg["mean_active_days"][service])
        k_disp = float(usage_cfg["dispersion_k"][service])

        # DEBUG: window length distribution per service
        win_lens = []
        for acc in accounts:
            s = user_state.get(service, {}).get(acc)
            if s is None:
                continue
            start = max(horizon_start, s.adoption_date)
            end = horizon_end if s.churn_date is None else min(horizon_end, s.churn_date)
            if end < start:
                continue
            win_lens.append((end - start).days + 1)

        if win_lens:
            arr = np.array(win_lens)
            print(
                f"[{service}] window_len median={np.median(arr)} mean={arr.mean():.1f} min={arr.min()} p25={np.percentile(arr, 25)} p75={np.percentile(arr, 75)}"
            )

        for acc in accounts:
            s = user_state.get(service, {}).get(acc)
            if s is None:
                continue

            # Available window is intersection of (adopt..churn) with horizon
            start = max(horizon_start, s.adoption_date)
            end = horizon_end if s.churn_date is None else min(horizon_end, s.churn_date)
            if end < start:
                continue

            # Candidate days
            days = []
            weights = []
            dd = start
            while dd <= end:
                days.append(dd)
                # intensity makes active days more likely for high-intensity accounts
                weights.append(season_weight(dd) * float(s.intensity))
                dd += dt.timedelta(days=1)
            w = np.asarray(weights, dtype=float)

            # DEBUG small sample: first 3 accounts per service
            if acc in set(accounts[:3]):
                window_len = len(days)
                print(
                    f"[{service}] acc={acc} window_len={window_len} intensity={float(s.intensity):.3f} mean_days={mean_days}"
                )

            # Interpret mean_active_days as "expected number of active days over this user's available window"
            mean_i = mean_days * float(s.intensity)
            n_active = _nbinom_sample(rng, mean=mean_i, k=k_disp)
            n_active = min(n_active, len(days))

            if acc in set(accounts[:3]):
                print(f"    -> sampled n_active={n_active}")

            chosen_days = _sample_days_weighted(rng, days, w, n_active)
            for day in chosen_days:
                key = (acc, day)
                calendar_map.setdefault(key, set()).add(service)

    # Dialog (optional): not in user_state, schedule separately
    if "dialog" in usage_cfg:
        mean_days = float(usage_cfg["dialog"]["mean_active_days"])
        k_disp = float(usage_cfg["dialog"]["dispersion_k"])
        for acc in accounts:
            # whole horizon as availability
            w = np.asarray([season_weight(day) for day in all_days], dtype=float)
            n_active = _nbinom_sample(rng, mean=mean_days, k=k_disp)
            n_active = min(n_active, len(all_days))
            chosen_days = _sample_days_weighted(rng, all_days, w, n_active)
            for day in chosen_days:
                key = (acc, day)
                calendar_map.setdefault(key, set()).add("dialog")

    return calendar_map


def _to_date_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series of 'YYYY-MM-DD' strings or datetime-like to datetime.date.
    """
    if s.dtype == object:
        # assumes ISO strings
        return pd.to_datetime(s, errors="raise").dt.date
    # already date/datetime; normalize
    return pd.to_datetime(s, errors="raise").dt.date


def _seasonality_multiplier(day: dt.date, activity_cfg: dict[str, Any]) -> float:
    # same semantics as in your generator: weekend + summer multipliers
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day.weekday()]
    return float(activity_cfg["dow_multiplier"][dow]) * float(
        activity_cfg["month_multiplier"][int(day.month)]
    )


def add_services_and_costs_v2(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    user_state: dict[str, dict[int, ServiceUserState]],
) -> pd.DataFrame:
    """
    Assign service flags + costs to chat-day rows using a PRECOMPUTED usage calendar.

    Key change vs previous version:
      - We DO NOT use per-day Bernoulli p_use anymore.
      - Instead, we precompute which (account_id, day) uses which services
        using `build_usage_calendar(...)` (mean active days + dispersion).
      - `has_*` flags are ROW-LEVEL: only the chat row(s) selected as "used"
        get `has_* = 1` so your usage_count analysis works as intended.

    Inputs:
      - df: output of simulate_active_chat_days(cfg) with columns: account_id, chat_id, chat_type, date
      - user_state: output of build_service_user_state(accounts, cfg, rng)

    Output:
      - df with required schema columns added:
        cost_dialog, cost_task, total_cost_tasks, cost_classification, cost_qc, cost_check_list,
        total_cost_classifications, cost_amocrm_call, total_cost,
        has_tasks, has_classifications, has_amocrm_call, num_services, has_both
    """
    out = df.copy()

    if "account_id" not in out.columns or "date" not in out.columns:
        raise ValidationError("df must contain at least: account_id, date")

    # Parse dates once for grouping + calendar lookup
    out["_date"] = _to_date_series(out["date"])

    rng = np.random.default_rng(int(cfg["dataset"]["seed"]) + 123)

    # ---- initialize required output columns
    out["cost_dialog"] = 0.0
    out["cost_task"] = 0.0
    out["total_cost_tasks"] = 0.0
    out["cost_classification"] = 0.0
    out["cost_qc"] = 0.0
    out["cost_check_list"] = 0.0
    out["total_cost_classifications"] = 0.0
    out["cost_amocrm_call"] = 0.0
    out["total_cost"] = 0.0

    out["has_tasks"] = 0
    out["has_classifications"] = 0
    out["has_amocrm_call"] = 0
    out["num_services"] = 0
    out["has_both"] = 0
    out["n_tasks"] = 0
    out["n_classifications"] = 0
    out["n_amocrm_calls"] = 0

    evt_cfg = cfg["services_event_volume"]
    evt_mean = evt_cfg["mean_events_per_active_day"]
    evt_k = evt_cfg["dispersion_k"]

    def sample_events(service: str) -> int:
        return _nbinom_sample(
            rng,
            mean=float(evt_mean[service]),
            k=float(evt_k[service]),
        )

    # ---- horizon + calendar (Option 1)
    horizon_start = dt.date.fromisoformat(cfg["dataset"]["start_date"])
    horizon_end = dt.date.fromisoformat(cfg["dataset"]["end_date"])

    accounts = sorted(out["account_id"].unique().tolist())
    usage_calendar = build_usage_calendar(
        accounts=accounts,
        cfg=cfg,
        user_state=user_state,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        rng=rng,
    )
    svc_counts = Counter()
    for svcs in usage_calendar.values():
        for s in svcs:
            svc_counts[s] += 1

    print("usage_calendar account-days:", len(usage_calendar))
    print("per-service account-days:", svc_counts)
    print(
        "paid-only days:",
        svc_counts.get("tasks", 0)
        + svc_counts.get("classifications", 0)
        + svc_counts.get("amocrm_call", 0),
    )
    print("dialog-only days:", sum(1 for svcs in usage_calendar.values() if svcs == {"dialog"}))

    # ---- cost sampler (still the same; later swapped to events×duration)
    cost_when_used = cfg.get("services", {}).get("cost_when_used", {})

    def positive_cost(service: str) -> float:
        """
        Positive cost sampler using shifted lognormal + clipping.
        Expects cost_when_used[service] has min/max/median/sigma.
        """
        c = cost_when_used.get(service)
        if not c:
            return 0.0
        cmin = float(c["min"])
        cmax = float(c["max"])
        median = float(c["median"])
        sigma = float(c["sigma"])

        # guard: median may equal min in your stats -> still allow right tail
        base_median = max(median - cmin, 0.01)
        mu = float(np.log(base_median))
        val = cmin + float(rng.lognormal(mean=mu, sigma=sigma, size=1)[0])
        return float(np.clip(val, cmin, cmax))

    def pick_one(idx_arr: np.ndarray) -> int:
        return int(rng.choice(idx_arr))

    # ---- group by account-day and assign costs/flags
    groups = out.groupby(["account_id", "_date"]).indices  # (acc, date) -> ndarray indices

    for (acc, day), idx in tqdm(groups.items(), total=len(groups), desc="Assigning service costs"):
        acc = int(acc)
        day = day  # dt.date
        idx_arr = np.asarray(idx, dtype=int)

        # Which services are used on this (account, day)?
        used_services = usage_calendar.get((acc, day), set())

        used_dialog = "dialog" in used_services
        used_tasks = "tasks" in used_services
        used_cls = "classifications" in used_services
        used_call = "amocrm_call" in used_services

        # Generate costs only if used
        cost_dialog = positive_cost("dialog") if used_dialog else 0.0
        cost_task = positive_cost("tasks") if used_tasks else 0.0
        cost_cls = positive_cost("classifications") if used_cls else 0.0
        cost_call = positive_cost("amocrm_call") if used_call else 0.0

        # Pick one row to store account-day summary fields
        chosen_row = pick_one(idx_arr)

        # Row-level flags + per-service costs: mark only the row that "used" the service
        # (so your `feature_df.groupby("account_id").size()` proxy makes sense)
        if used_tasks:
            r = pick_one(idx_arr)
            out.loc[r, "has_tasks"] = 1
            n = sample_events("tasks")
            out.loc[r, "n_tasks"] = n

            # optional: scale cost by n (quick proxy)
            cost_task = positive_cost("tasks") * max(1, n / float(evt_mean["tasks"]))
            out.loc[r, "cost_task"] = cost_task
            out.loc[r, "total_cost_tasks"] = cost_task
        if used_cls:
            r = pick_one(idx_arr)
            out.loc[r, "has_classifications"] = 1
            n = sample_events("classifications")
            out.loc[r, "n_classifications"] = n

            cost_cls = positive_cost("classifications") * max(
                1, n / float(evt_mean["classifications"])
            )
            out.loc[r, "cost_classification"] = cost_cls
            out.loc[r, "total_cost_classifications"] = cost_cls
        if used_call:
            r = pick_one(idx_arr)
            out.loc[r, "has_amocrm_call"] = 1
            n = sample_events("amocrm_call")
            out.loc[r, "n_amocrm_calls"] = n

            cost_call = positive_cost("amocrm_call") * max(1, n / float(evt_mean["amocrm_call"]))
            out.loc[r, "cost_amocrm_call"] = cost_call

        # Dialog cost is written once per account-day (does not affect your has_* usage_count)
        out.loc[chosen_row, "cost_dialog"] = cost_dialog

        # Account-day summary metrics (store once to avoid duplication)
        num_services = int(used_tasks) + int(used_cls) + int(used_call)
        has_both = int(num_services >= 2)
        out.loc[chosen_row, "num_services"] = num_services
        out.loc[chosen_row, "has_both"] = has_both

        # Total cost stored once per account-day
        total_cost = float(cost_dialog + cost_task + cost_cls + cost_call)
        out.loc[chosen_row, "total_cost"] = total_cost

    out.drop(columns=["_date"], inplace=True)
    return out
