from __future__ import annotations

from datetime import date

import pandas as pd

from cost_agent_mvp.analytics import kpi_definitions as kd


def _make_df() -> pd.DataFrame:
    """Small fixture spanning two days.

    Note: rows are considered "active" only if at least one INCLUDED service column
    has cost > 0 (tasks/classifications/amocrm-call).
    """
    return pd.DataFrame(
        [
            # report day (2025-02-01)
            {
                "date": "2025-02-01",
                "account_id": 1,
                "chat_id": "c1",
                "chat_type": "telegram",
                "total_cost_tasks": 10.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                # extra columns are OK; KPI module ignores them
                "total_cost": 10.0,
            },
            {
                "date": "2025-02-01",
                "account_id": 2,
                "chat_id": "c2",
                "chat_type": "whatsapp",
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 6.0,
                "cost_amocrm_call": 0.0,
                "total_cost": 6.0,
            },
            # previous day (2025-01-31)
            {
                "date": "2025-01-31",
                "account_id": 1,
                "chat_id": "c1",
                "chat_type": "telegram",
                "total_cost_tasks": 8.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                "total_cost": 8.0,
            },
        ]
    )


def _day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    day_d = pd.to_datetime(day).date()
    return out[out["date"] == day_d].reset_index(drop=True)


def test_kpis_golden_bundle_small_fixture() -> None:
    df = _make_df()

    today = _day(df, "2025-02-01")
    prev = _day(df, "2025-01-31")

    # Core KPIs
    k_active = kd.active_users(today)
    k_active_prev = kd.active_users(prev)
    d_active_abs, d_active_pct = kd.delta(k_active, k_active_prev)

    k_cost = kd.total_cost_services(today)
    k_cost_prev = kd.total_cost_services(prev)
    d_cost_abs, d_cost_pct = kd.delta(k_cost, k_cost_prev)

    k_dials = kd.dials_analyzed(today)
    k_avg_non_diluted = kd.avg_cost_per_account_non_diluted(today)

    # New/expanded bundle metrics
    k_avg_diluted = kd.avg_cost_per_active_account_day(today)
    k_p95 = kd.pxx_cost_per_account_day(today, 0.95)
    k_pct_above_cap = kd.pct_accounts_above_cost_cap(today)  # default cap
    k_margin = kd.avg_margin_per_account_day(today)  # default price

    # Expected values
    assert k_active == 2.0
    assert k_active_prev == 1.0
    assert d_active_abs == 1.0
    assert d_active_pct == 1.0  # (2-1)/1

    assert k_cost == 16.0  # 10 + 6
    assert k_cost_prev == 8.0
    assert d_cost_abs == 8.0
    assert d_cost_pct == 1.0

    assert k_dials == 2.0  # 2 active rows on "today"
    assert k_avg_non_diluted == 16.0  # 10 (tasks) + 6 (classifications)

    # Diluted avg reconciles with total cost: 16 / 2 = 8
    assert k_avg_diluted == 8.0

    # Per-account-day totals are {1: 10, 2: 6} => p95 with 2 points = 9.8 (linear interp)
    assert k_p95 == 9.8

    # Default cap is far above 10/6, so no accounts above cap
    assert k_pct_above_cap == 0.0

    # Default price is 100 => margin = 100 - 8 = 92
    assert k_margin == 92.0

    # Sanity: per-account-day costs are stable and exclude zero-cost accounts
    per_acc_day = kd.per_account_day_costs(today)
    assert per_acc_day.to_dict() == {2: 6.0, 1: 10.0}


def test_service_combo_counts_and_daily_series_are_consistent() -> None:
    df = _make_df()

    today = _day(df, "2025-02-01")
    prev = _day(df, "2025-01-31")

    assert kd.service_combo_counts(today) == {
        "Has 1 Service": 2,
        "Has 2 Services": 0,
        "All 3 Services": 0,
    }
    assert kd.service_combo_counts(prev) == {
        "Has 1 Service": 1,
        "Has 2 Services": 0,
        "All 3 Services": 0,
    }

    series = kd.build_daily_series(df, dates=[date(2025, 1, 31), date(2025, 2, 1)])
    assert list(series.columns) == [
        "date",
        "total_cost_services",
        "dials_analyzed",
        "active_users",
        "avg_cost_per_active_account_day",
        "p95_cost_per_account_day",
        "pct_accounts_above_cap",
        "avg_margin_per_account_day",
    ]

    got = series.set_index("date").to_dict(orient="index")

    # 2025-01-31
    assert got[date(2025, 1, 31)]["total_cost_services"] == 8.0
    assert got[date(2025, 1, 31)]["dials_analyzed"] == 1.0
    assert got[date(2025, 1, 31)]["active_users"] == 1.0
    assert got[date(2025, 1, 31)]["avg_cost_per_active_account_day"] == 8.0
    assert got[date(2025, 1, 31)]["p95_cost_per_account_day"] == 8.0
    assert got[date(2025, 1, 31)]["pct_accounts_above_cap"] == 0.0
    assert got[date(2025, 1, 31)]["avg_margin_per_account_day"] == 92.0

    # 2025-02-01
    assert got[date(2025, 2, 1)]["total_cost_services"] == 16.0
    assert got[date(2025, 2, 1)]["dials_analyzed"] == 2.0
    assert got[date(2025, 2, 1)]["active_users"] == 2.0
    assert got[date(2025, 2, 1)]["avg_cost_per_active_account_day"] == 8.0
    assert got[date(2025, 2, 1)]["p95_cost_per_account_day"] == 9.8
    assert got[date(2025, 2, 1)]["pct_accounts_above_cap"] == 0.0
    assert got[date(2025, 2, 1)]["avg_margin_per_account_day"] == 92.0
