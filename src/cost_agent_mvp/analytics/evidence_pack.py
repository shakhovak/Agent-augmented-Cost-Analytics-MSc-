"""Build standard tables for daily report."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from src.analytics.kpi_definitions import (
    active_users,
    avg_cost_per_account_diluted,
    avg_cost_per_account_non_diluted,
    component_costs_sum,
    delta,
    dials_analyzed,
    distribution_stats,
    histogram_table,
    per_account_total_cost,
    total_cost_sum,
)
from src.core.errors import ValidationError

# Evidence pack is a dict[str, DataFrame]
EvidencePack = dict[str, pd.DataFrame]


def _require_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns for evidence pack: {missing}")


def _filter_day(df: pd.DataFrame, day: date) -> pd.DataFrame:
    _require_columns(df, ["date"])
    return df[df["date"] == day]


def _filter_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    _require_columns(df, ["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)]


# -----------------------------
# Standard Daily Evidence Pack
# -----------------------------


def build_standard_daily_evidence(
    df_all: pd.DataFrame,
    report_day: date,
    trend_days: int = 7,
    top_n: int = 10,
) -> EvidencePack:
    """
    Build a deterministic Evidence Pack for the "Standard Daily Report".

    Inputs:
      - df_all: full dataset (already type-normalized; date is datetime.date)
      - report_day: the target day (e.g., yesterday)
      - trend_days: number of days to include in trend tables
      - top_n: for top account tables

    Outputs: dict of named DataFrames:
      - kpi_today_vs_yesterday
      - service_cost_breakdown
      - top_accounts_by_total_cost
      - top_accounts_by_delta
      - trend_7d
      - distribution_avg_cost_per_account (histogram bins)
      - exceptions_queue (empty placeholder in Step 1)
    """
    if trend_days <= 1:
        raise ValidationError("trend_days must be >= 2")
    if top_n <= 0:
        raise ValidationError("top_n must be positive")

    prev_day = report_day - timedelta(days=1)
    trend_start = report_day - timedelta(days=trend_days - 1)

    df_today = _filter_day(df_all, report_day)
    df_prev = _filter_day(df_all, prev_day)
    df_trend = _filter_range(df_all, trend_start, report_day)

    # --- KPI table (today vs yesterday) ---
    tot_today = total_cost_sum(df_today)
    tot_prev = total_cost_sum(df_prev)
    tot_d_abs, tot_d_pct = delta(tot_today, tot_prev)

    au_today = active_users(df_today)
    au_prev = active_users(df_prev)
    au_d_abs, au_d_pct = delta(float(au_today), float(au_prev))

    dials_today = dials_analyzed(df_today)
    dials_prev = dials_analyzed(df_prev)
    dials_d_abs, dials_d_pct = delta(float(dials_today), float(dials_prev))

    avg_dil_today = avg_cost_per_account_diluted(df_today)
    avg_dil_prev = avg_cost_per_account_diluted(df_prev)
    avg_dil_d_abs, avg_dil_d_pct = delta(avg_dil_today, avg_dil_prev)

    avg_nd_today = avg_cost_per_account_non_diluted(df_today)
    avg_nd_prev = avg_cost_per_account_non_diluted(df_prev)
    avg_nd_d_abs, avg_nd_d_pct = delta(avg_nd_today, avg_nd_prev)

    kpi = pd.DataFrame(
        [
            {
                "date": report_day,
                "total_cost": tot_today,
                "total_cost_prev_day": tot_prev,
                "total_cost_delta_abs": tot_d_abs,
                "total_cost_delta_pct": tot_d_pct,
                "active_users": au_today,
                "active_users_prev_day": au_prev,
                "active_users_delta_abs": au_d_abs,
                "active_users_delta_pct": au_d_pct,
                "dials_analyzed": dials_today,
                "dials_analyzed_prev_day": dials_prev,
                "dials_analyzed_delta_abs": dials_d_abs,
                "dials_analyzed_delta_pct": dials_d_pct,
                "avg_cost_per_account": avg_dil_today,
                "avg_cost_per_account_prev_day": avg_dil_prev,
                "avg_cost_per_account_delta_abs": avg_dil_d_abs,
                "avg_cost_per_account_delta_pct": avg_dil_d_pct,
                "avg_cost_per_account_non_diluted": avg_nd_today,
                "avg_cost_per_account_non_diluted_prev_day": avg_nd_prev,
                "avg_cost_per_account_non_diluted_delta_abs": avg_nd_d_abs,
                "avg_cost_per_account_non_diluted_delta_pct": avg_nd_d_pct,
            }
        ]
    )

    # --- Service / component breakdown (today) ---
    comps = component_costs_sum(df_today)
    breakdown_rows = []
    for comp_name, comp_cost in comps.items():
        breakdown_rows.append(
            {
                "date": report_day,
                "component": comp_name,
                "cost": float(comp_cost),
            }
        )
    service_breakdown = pd.DataFrame(breakdown_rows)
    if not service_breakdown.empty:
        total = float(service_breakdown["cost"].sum())
        service_breakdown["share_of_total"] = (
            service_breakdown["cost"] / total if total > 0 else 0.0
        )
        service_breakdown = service_breakdown.sort_values("cost", ascending=False).reset_index(
            drop=True
        )
    else:
        service_breakdown = pd.DataFrame(columns=["date", "component", "cost", "share_of_total"])

    # --- Top accounts by total cost (today) ---
    per_acc_today = per_account_total_cost(df_today)
    if not per_acc_today.empty:
        per_acc_today = per_acc_today.sort_values("total_cost", ascending=False).reset_index(
            drop=True
        )
        total = float(per_acc_today["total_cost"].sum())
        per_acc_today["share_of_total"] = per_acc_today["total_cost"] / total if total > 0 else 0.0
        top_accounts_cost = per_acc_today.head(top_n).copy()
    else:
        top_accounts_cost = pd.DataFrame(columns=["account_id", "total_cost", "share_of_total"])

    # --- Top accounts by delta vs previous day ---
    per_acc_prev = per_account_total_cost(df_prev).rename(
        columns={"total_cost": "total_cost_prev_day"}
    )
    per_acc_today2 = (
        per_acc_today.rename(columns={"total_cost": "total_cost_today"})
        if not per_acc_today.empty
        else pd.DataFrame(columns=["account_id", "total_cost_today"])
    )
    merged = per_acc_today2.merge(per_acc_prev, on="account_id", how="outer").fillna(0.0)

    if not merged.empty:
        merged["delta_abs"] = merged["total_cost_today"] - merged["total_cost_prev_day"]
        merged["delta_pct"] = merged.apply(
            lambda r: (
                (r["delta_abs"] / r["total_cost_prev_day"])
                if r["total_cost_prev_day"] > 0
                else (1.0 if r["total_cost_today"] > 0 else 0.0)
            ),
            axis=1,
        )
        top_accounts_delta = (
            merged.sort_values("delta_abs", ascending=False).head(top_n).reset_index(drop=True)
        )
    else:
        top_accounts_delta = pd.DataFrame(
            columns=[
                "account_id",
                "total_cost_today",
                "total_cost_prev_day",
                "delta_abs",
                "delta_pct",
            ]
        )

    # --- 7-day trend table ---
    if df_trend.empty:
        trend = pd.DataFrame(
            columns=[
                "date",
                "total_cost",
                "active_users",
                "dials_analyzed",
                "total_cost_tasks",
                "total_cost_classifications",
            ]
        )
    else:
        # Aggregate per day deterministically
        daily_cost = df_trend.groupby("date", as_index=False)["total_cost"].sum()

        # Usage proxies per day
        au_rows = []
        dials_rows = []
        for d, part in df_trend.groupby("date"):
            au_rows.append({"date": d, "active_users": active_users(part)})
            dials_rows.append({"date": d, "dials_analyzed": dials_analyzed(part)})

        au_df = pd.DataFrame(au_rows)
        dials_df = pd.DataFrame(dials_rows)

        # Major buckets
        cols_optional = []
        if "total_cost_tasks" in df_trend.columns:
            cols_optional.append("total_cost_tasks")
        if "total_cost_classifications" in df_trend.columns:
            cols_optional.append("total_cost_classifications")

        if cols_optional:
            buckets = df_trend.groupby("date", as_index=False)[cols_optional].sum()
        else:
            buckets = pd.DataFrame({"date": daily_cost["date"]})

        trend = (
            daily_cost.merge(au_df, on="date", how="left")
            .merge(dials_df, on="date", how="left")
            .merge(buckets, on="date", how="left")
        )
        trend = trend.sort_values("date", ascending=True).reset_index(drop=True)

    # --- Distribution (today) ---
    # For histogram we use per-account daily totals (this can be many accounts but the histogram is compact).
    per_acc_vals = (
        per_acc_today["total_cost"] if not per_acc_today.empty else pd.Series([], dtype=float)
    )
    dist_hist = histogram_table(per_acc_vals, bins=20)
    # Put a conventional column name for plotting configs
    # (hist can also be plotted from bin_left/bin_right/count)
    dist_hist["metric"] = "total_cost_per_account"

    # Extra stats table (optional but very useful for narratives)
    dist_stats = distribution_stats(per_acc_vals)
    dist_stats_df = pd.DataFrame([{"date": report_day, **dist_stats}])

    # --- Exceptions queue placeholder (Step 1) ---
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
        "kpi_today_vs_yesterday": kpi,
        "service_cost_breakdown": service_breakdown,
        "top_accounts_by_total_cost": top_accounts_cost,
        "top_accounts_by_delta": top_accounts_delta,
        "trend_7d": trend,
        "distribution_avg_cost_per_account": dist_hist,
        "distribution_stats": dist_stats_df,  # optional but helpful
        "exceptions_queue": exceptions,
    }
