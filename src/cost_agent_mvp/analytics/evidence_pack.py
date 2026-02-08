from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from cost_agent_mvp.analytics.kpi_definitions import (
    build_daily_series,
    per_account_day_costs,  # NEW (distribution foundation)
    service_combo_counts,
    user_churn,
)
from cost_agent_mvp.viz.chart_specs import SERVICE_COST_COLUMNS, SERVICE_LABELS

EvidencePack = dict[str, pd.DataFrame]


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def build_standard_daily_evidence(
    df_all: pd.DataFrame,
    report_day: date,
    *,
    trend_days: int = 7,
    top_accounts_n: int = 10,
    hist_bins: int = 30,
) -> EvidencePack:
    """
    Evidence pack tailored to the dashboard:
    - excludes dialog from "services"
    - excludes rows where the included service_total == 0
    """
    df_all = _ensure_date(df_all)

    today = df_all[df_all["date"] == report_day].copy()
    yesterday = df_all[df_all["date"] == (report_day - timedelta(days=1))].copy()

    # -------- Trend window
    start = report_day - timedelta(days=trend_days - 1)
    days = [start + timedelta(days=i) for i in range(trend_days)]
    trend_daily = build_daily_series(df_all, days)

    # -------- KPIs (single-row table)
    churn = user_churn(today, yesterday)
    kpi_row = {
        "date": report_day,
        "active_users": float(
            trend_daily.loc[trend_daily["date"] == report_day, "active_users"].iloc[0]
        ),
        "active_users_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "active_users",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "new_users": float(churn.new_users),
        "churned_users": float(churn.churned_users),
        "net_user_change": float(churn.net_change),
        "total_cost_services": float(
            trend_daily.loc[trend_daily["date"] == report_day, "total_cost_services"].iloc[0]
        ),
        "avg_cost_per_active_account_day": float(
            trend_daily.loc[
                trend_daily["date"] == report_day, "avg_cost_per_active_account_day"
            ].iloc[0]
        ),
        "avg_cost_per_active_account_day_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "avg_cost_per_active_account_day",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "p95_cost_per_account_day": float(
            trend_daily.loc[trend_daily["date"] == report_day, "p95_cost_per_account_day"].iloc[0]
        ),
        "p95_cost_per_account_day_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "p95_cost_per_account_day",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "pct_accounts_above_cap": float(
            trend_daily.loc[trend_daily["date"] == report_day, "pct_accounts_above_cap"].iloc[0]
        ),
        "pct_accounts_above_cap_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "pct_accounts_above_cap",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "avg_margin_per_account_day": float(
            trend_daily.loc[trend_daily["date"] == report_day, "avg_margin_per_account_day"].iloc[0]
        ),
        "avg_margin_per_account_day_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "avg_margin_per_account_day",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "total_cost_services_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "total_cost_services",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
        "dials_analyzed": float(
            trend_daily.loc[trend_daily["date"] == report_day, "dials_analyzed"].iloc[0]
        ),
        "dials_analyzed_prev": (
            float(
                trend_daily.loc[
                    trend_daily["date"] == (report_day - timedelta(days=1)),
                    "dials_analyzed",
                ].iloc[0]
            )
            if (report_day - timedelta(days=1)) in set(trend_daily["date"])
            else 0.0
        ),
    }

    # deltas
    def _delta_abs(a: float, b: float) -> float:
        return float(a - b)

    def _delta_pct(a: float, b: float) -> float:
        return float((a - b) / b) if b else 0.0

    for base in [
        "active_users",
        "avg_cost_per_active_account_day",
        "p95_cost_per_account_day",
        "pct_accounts_above_cap",
        "avg_margin_per_account_day",
        "total_cost_services",
        "dials_analyzed",
    ]:
        kpi_row[f"{base}_delta_abs"] = _delta_abs(kpi_row[base], kpi_row[f"{base}_prev"])
        kpi_row[f"{base}_delta_pct"] = _delta_pct(kpi_row[base], kpi_row[f"{base}_prev"])
    kpis = pd.DataFrame([kpi_row])

    # -------- Service breakdown (mix) for report_day
    # Use service costs only
    service_totals = {}
    for col in SERVICE_COST_COLUMNS:
        service_totals[col] = float(today[col].sum())

    total_services = float(sum(service_totals.values()))
    breakdown_rows = []
    for col in SERVICE_COST_COLUMNS:
        cost = service_totals[col]
        breakdown_rows.append(
            {
                "date": report_day,
                "component": col,
                "label": SERVICE_LABELS.get(col, col),
                "cost": cost,
                "share_of_total": (float(cost / total_services) if total_services else 0.0),
            }
        )
    service_breakdown = (
        pd.DataFrame(breakdown_rows).sort_values("cost", ascending=False).reset_index(drop=True)
    )

    # -------- Service usage totals (for plot E)
    service_usage_totals = service_breakdown[["date", "label", "cost"]].copy()

    # -------- Top accounts by total service cost
    if today.empty:
        top_accounts = pd.DataFrame(
            columns=[
                "date",
                "rank",
                "account_id",
                "total_cost_services",
                "share_of_total_cost",
            ]
        )
    else:
        tmp = today.copy()
        tmp["total_cost_services_row"] = tmp[list(SERVICE_COST_COLUMNS)].sum(axis=1)
        tmp = tmp[tmp["total_cost_services_row"] > 0]
        per_acc = (
            tmp.groupby("account_id", as_index=False)["total_cost_services_row"]
            .sum()
            .rename(columns={"total_cost_services_row": "total_cost_services"})
        )
        per_acc = (
            per_acc.sort_values("total_cost_services", ascending=False)
            .head(top_accounts_n)
            .reset_index(drop=True)
        )
        per_acc["rank"] = np.arange(1, len(per_acc) + 1)
        per_acc["date"] = report_day
        per_acc["share_of_total_cost"] = (
            per_acc["total_cost_services"] / total_services if total_services else 0.0
        )
        top_accounts = per_acc[
            ["date", "rank", "account_id", "total_cost_services", "share_of_total_cost"]
        ]

    # -------- Distribution (plot A)
    dist_series = per_account_day_costs(today)
    if dist_series.empty:
        distribution_hist = pd.DataFrame(columns=["bin_left", "bin_right", "count"])
        distribution_stats = pd.DataFrame(
            [{"median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}]
        )
    else:
        counts, edges = np.histogram(dist_series.values, bins=hist_bins)
        distribution_hist = pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "count": counts,
            }
        )
        distribution_stats = pd.DataFrame(
            [
                {
                    "median": float(np.median(dist_series.values)),
                    "min": float(np.min(dist_series.values)),
                    "max": float(np.max(dist_series.values)),
                    "std": (
                        float(np.std(dist_series.values, ddof=1)) if len(dist_series) > 1 else 0.0
                    ),
                    "n": int(len(dist_series)),
                }
            ]
        )

    # -------- Service combinations (plot D)
    combo = service_combo_counts(today)
    service_combinations = pd.DataFrame(
        [{"date": report_day, "label": k, "active_users": int(v)} for k, v in combo.items()]
    )

    # -------- Dials & cost trends (plot F) — reuse trend_daily with relevant columns
    dials_cost_trend = trend_daily[["date", "dials_analyzed", "total_cost_services"]].copy()

    return {
        "kpis": kpis,
        "trend_daily": trend_daily,
        "service_breakdown": service_breakdown,
        "service_totals": service_usage_totals,
        "top_accounts": top_accounts,
        "distribution_hist": distribution_hist,
        "distribution_stats": distribution_stats,
        "service_combinations": service_combinations,
        "dials_cost_trend": dials_cost_trend,
        "exceptions_queue": pd.DataFrame(columns=["message"]),  # placeholder, keep contract stable
    }
