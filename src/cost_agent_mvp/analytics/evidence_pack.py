from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from cost_agent_mvp.analytics.kpi_definitions import (
    active_account_days,
    active_users,
    avg_cost_per_account_diluted,
    avg_cost_per_active_account_day,
    avg_cost_per_active_service_account,
    component_costs_sum,
    delta_abs,
    delta_pct,
    dials_analyzed,
    distribution_stats,
    histogram_table,
    rows_per_account_day_stats,
    service_mix_share,
    top_accounts,
    total_cost_sum,
)
from cost_agent_mvp.core.errors import ValidationError

EvidencePack = dict[str, pd.DataFrame]


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns for evidence pack: {missing}")


def _filter_day(df: pd.DataFrame, day: date) -> pd.DataFrame:
    _require_columns(df, ["date"])
    return df[df["date"] == day]


def _filter_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    _require_columns(df, ["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _kpi_row(report_day: date, cur: dict[str, float], prev: dict[str, float]) -> pd.DataFrame:
    """Build a single-row KPI table with current, previous, and deltas."""
    row: dict[str, Any] = {"date": report_day}

    for name, cur_val in cur.items():
        prev_val = float(prev.get(name, 0.0))
        row[name] = float(cur_val)
        row[f"{name}_prev"] = prev_val
        row[f"{name}_delta_abs"] = float(delta_abs(cur_val, prev_val))
        row[f"{name}_delta_pct"] = float(delta_pct(cur_val, prev_val))

    return pd.DataFrame([row])


def build_standard_daily_evidence(
    df_all: pd.DataFrame,
    report_day: date,
    trend_days: int = 7,
    top_n: int = 10,
    cap: int | None = None,
) -> EvidencePack:
    """
    Build a deterministic Evidence Pack for the "Standard Daily Report".

    Required v0 outputs (dict of DataFrames):
      - kpis: single-row table with current + previous + deltas
      - trend_daily: date-level trend for last N days
      - service_breakdown: component/service cost mix for report_day
      - top_accounts: top N accounts by total_cost for report_day

    Notes:
      - df_all must be type-normalized (date column is datetime.date).
      - If report_day is missing entirely from df_all (no rows), raises ValidationError.
      - If previous day has no rows, previous KPIs evaluate to 0 deterministically.
      - cap is optional and is used only for rows-per-account-day diagnostics.
    """
    if trend_days < 2:
        raise ValidationError("trend_days must be >= 2")
    if top_n <= 0:
        raise ValidationError("top_n must be positive")

    prev_day = report_day - timedelta(days=1)
    trend_start = report_day - timedelta(days=trend_days - 1)

    df_today = _filter_day(df_all, report_day)
    if df_today.empty:
        raise ValidationError(f"No data for report_day={report_day} (no rows with this date).")

    df_prev = _filter_day(df_all, prev_day)
    df_trend = _filter_range(df_all, trend_start, report_day)

    # -------------------------
    # KPIs (today vs previous)
    # -------------------------
    # Keep this compact & narrative-friendly (8–12 KPIs).
    cur_kpis = {
        "total_cost": float(total_cost_sum(df_today)),
        "active_users": float(active_users(df_today)),
        "active_account_days": float(active_account_days(df_today)),
        "dials_analyzed": float(dials_analyzed(df_today)),
        "avg_cost_per_account": float(avg_cost_per_account_diluted(df_today)),
        "avg_cost_per_active_account_day": float(avg_cost_per_active_account_day(df_today)),
        "avg_cost_per_active_service_account": float(avg_cost_per_active_service_account(df_today)),
    }

    prev_kpis = {
        "total_cost": float(total_cost_sum(df_prev)),
        "active_users": float(active_users(df_prev)),
        "active_account_days": float(active_account_days(df_prev)),
        "dials_analyzed": float(dials_analyzed(df_prev)),
        "avg_cost_per_account": float(avg_cost_per_account_diluted(df_prev)),
        "avg_cost_per_active_account_day": float(avg_cost_per_active_account_day(df_prev)),
        "avg_cost_per_active_service_account": float(avg_cost_per_active_service_account(df_prev)),
    }

    # Add cap / row-density diagnostics as KPI fields (flattened)
    rows_stats_today = rows_per_account_day_stats(df_today, cap=cap)
    rows_stats_prev = (
        rows_per_account_day_stats(df_prev, cap=cap)
        if not df_prev.empty
        else dict.fromkeys(rows_stats_today.keys(), 0.0)
    )

    # We only include the most interpretable row-density metrics in the KPI row
    for key in ["p95", "max", "n_over_cap", "pct_over_cap"]:
        cur_kpis[f"rows_per_account_day_{key}"] = float(rows_stats_today.get(key, 0.0))
        prev_kpis[f"rows_per_account_day_{key}"] = float(rows_stats_prev.get(key, 0.0))

    kpis_df = _kpi_row(report_day, cur_kpis, prev_kpis)

    # -------------------------
    # Trend table (last N days)
    # -------------------------
    if df_trend.empty:
        trend_df = pd.DataFrame(
            columns=[
                "date",
                "total_cost",
                "active_users",
                "active_account_days",
                "dials_analyzed",
            ]
        )
    else:
        daily_total = df_trend.groupby("date", as_index=False)["total_cost"].sum()

        au_rows = []
        aad_rows = []
        dials_rows = []
        for d, part in df_trend.groupby("date"):
            au_rows.append({"date": d, "active_users": active_users(part)})
            aad_rows.append({"date": d, "active_account_days": active_account_days(part)})
            dials_rows.append({"date": d, "dials_analyzed": dials_analyzed(part)})

        au_df = pd.DataFrame(au_rows)
        aad_df = pd.DataFrame(aad_rows)
        dials_df = pd.DataFrame(dials_rows)

        trend_df = (
            daily_total.merge(au_df, on="date", how="left")
            .merge(aad_df, on="date", how="left")
            .merge(dials_df, on="date", how="left")
            .sort_values("date", ascending=True)
            .reset_index(drop=True)
        )

    # -------------------------
    # Service breakdown (today)
    # -------------------------

    mix = service_mix_share(df_today)  # component, cost, share
    if mix.empty:
        service_df = pd.DataFrame(columns=["date", "component", "cost", "share_of_total"])
    else:
        service_df = mix.rename(columns={"share": "share_of_total"}).copy()
        service_df.insert(0, "date", report_day)
        service_df = service_df.sort_values("cost", ascending=False).reset_index(drop=True)

    # -------------------------
    # Top accounts (today)
    # -------------------------
    top_df = top_accounts(df_today, n=top_n)  # rank, account_id, total_cost, share_of_total_cost
    if top_df.empty:
        top_accounts_df = pd.DataFrame(
            columns=["date", "rank", "account_id", "total_cost", "share_of_total_cost"]
        )
    else:
        top_accounts_df = top_df.copy()
        top_accounts_df.insert(0, "date", report_day)

    # -------------------------
    # Optional extras (useful later)
    # -------------------------

    # Extra: component absolute totals (not shares) – sometimes handy

    comps = component_costs_sum(df_today)
    service_totals_df = pd.DataFrame(
        [{"date": report_day, "component": k, "cost": float(v)} for k, v in sorted(comps.items())],
        columns=["date", "component", "cost"],
    )

    # Extra: distribution for total_cost_per_account (hist + stats)
    # Using per-account totals from top_accounts_df isn't complete; compute from df_today instead.
    # For histogram: we approximate per-account by grouping total_cost by account_id and summing.
    if "account_id" in df_today.columns and "total_cost" in df_today.columns:
        per_acc = df_today.groupby("account_id", as_index=False)["total_cost"].sum()["total_cost"]
    else:
        per_acc = pd.Series([], dtype=float)

    dist_hist = histogram_table(per_acc, bins=20)
    dist_hist["metric"] = "total_cost_per_account"
    dist_stats_df = pd.DataFrame([{"date": report_day, **distribution_stats(per_acc)}])

    # Placeholder for later rule-based exceptions (kept from M1 style)
    exceptions = pd.DataFrame(
        columns=[
            "date",
            "level",
            "entity_id",
            "rule_id",
            "severity",
            "metric_value",
            "baseline_value",
            "delta_abs",
            "delta_pct",
            "supporting_notes",
        ]
    )

    return {
        # REQUIRED v0 keys:
        "kpis": kpis_df,
        "trend_daily": trend_df,
        "service_breakdown": service_df,
        "top_accounts": top_accounts_df,
        # OPTIONAL extras:
        "service_totals": service_totals_df,
        "distribution_hist": dist_hist,
        "distribution_stats": dist_stats_df,
        "exceptions_queue": exceptions,
    }
