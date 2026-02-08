from __future__ import annotations

from datetime import date

import pandas as pd

from cost_agent_mvp.analytics import kpi_definitions as kd


def _df_day() -> pd.DataFrame:
    """
    Small, explicit fixture for one day.
    We include:
      - multiple rows per account (to test "dials" counting rows)
      - zero-cost rows (must be excluded from 'active' logic)
      - all three service columns expected by KPI logic
    """
    d = date(2025, 1, 1)
    df = pd.DataFrame(
        [
            # account 1: one active row (tasks), one zero row
            {
                "account_id": 1,
                "date": d,
                "total_cost_tasks": 10.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
            },
            {
                "account_id": 1,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
            },
            # account 2: two active rows (classifications + amocrm)
            {
                "account_id": 2,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 20.0,
                "cost_amocrm_call": 0.0,
            },
            {
                "account_id": 2,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 5.0,
            },
            # account 3: only zero rows -> should NOT be counted as active user
            {
                "account_id": 3,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
            },
        ]
    )
    # total_cost is often present in your pipeline; set it for completeness
    df["total_cost"] = (
        df["total_cost_tasks"] + df["total_cost_classifications"] + df["cost_amocrm_call"]
    )
    return df


def test_active_users_unique_accounts_excludes_zero_cost() -> None:
    df = _df_day()
    assert kd.active_users(df) == 2  # accounts 1 and 2 only


def test_dials_analyzed_counts_nonzero_service_lines() -> None:
    df = _df_day()
    # active rows are: (acc1 row1), (acc2 row3), (acc2 row4) => 3 rows
    assert kd.dials_analyzed(df) == 3


def test_avg_cost_per_account_non_diluted_logic() -> None:
    """
    Expected non-diluted average definition:
      - For each service separately:
          filter rows where service_cost > 0
          groupby account_id -> mean (per account)
          mean across accounts (average of account-means)
      - Then sum across services
    Using _df_day():
      tasks: only acc1 has >0 => mean_tasks = 10
      classifications: only acc2 has >0 => mean_classif = 20
      amocrm: only acc2 has >0 => mean_amocrm = 5
      total = 35
    """
    df = _df_day()
    assert kd.avg_cost_per_account_non_diluted(df) == 35.0


def test_total_cost_services_sums_service_columns() -> None:
    df = _df_day()
    assert kd.total_cost_services(df) == 35.0


def test_per_account_day_costs_excludes_zero_accounts_and_sums_rows() -> None:
    df = _df_day()
    # account 1: 10; account 2: 20 + 5 = 25; account 3 excluded
    s = kd.per_account_day_costs(df)
    assert s.to_dict() == {1: 10.0, 2: 25.0} or s.to_dict() == {2: 25.0, 1: 10.0}
    assert set(s.index.tolist()) == {1, 2}


def test_avg_cost_per_active_account_day_is_diluted_total_over_active_users() -> None:
    df = _df_day()
    # total_cost_services = 35, active_users = 2 => 17.5
    assert kd.avg_cost_per_active_account_day(df) == 17.5


def test_pxx_cost_and_cap_and_margin_on_small_fixture() -> None:
    df = _df_day()
    # per-account totals are [10, 25]; p95 = 10 + 0.95*(25-10) = 24.25
    assert kd.pxx_cost_per_account_day(df, 0.95) == 24.25

    # with cap=20: one of two accounts (25) is above => 0.5
    assert kd.pct_accounts_above_cost_cap(df, cap_usd=20.0) == 0.5

    # margin uses default price=100: 100 - 17.5 = 82.5
    assert kd.avg_margin_per_account_day(df) == 82.5


def test_user_churn_new_churned_net() -> None:
    d = date(2025, 1, 2)

    # Today: accounts 1,2 active
    today = pd.DataFrame(
        [
            {
                "account_id": 1,
                "date": d,
                "total_cost_tasks": 1.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
            },
            {
                "account_id": 2,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 2.0,
                "cost_amocrm_call": 0.0,
            },
        ]
    )
    today["total_cost"] = (
        today["total_cost_tasks"] + today["total_cost_classifications"] + today["cost_amocrm_call"]
    )

    # Yesterday: accounts 2,3 active
    yest = pd.DataFrame(
        [
            {
                "account_id": 2,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 2.0,
                "cost_amocrm_call": 0.0,
            },
            {
                "account_id": 3,
                "date": d,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 3.0,
            },
        ]
    )
    yest["total_cost"] = (
        yest["total_cost_tasks"] + yest["total_cost_classifications"] + yest["cost_amocrm_call"]
    )

    churn = kd.user_churn(today, yest)
    assert churn.new_users == 1  # account 1
    assert churn.churned_users == 1  # account 3
    assert churn.net_change == 0  # 2 active today - 2 active yesterday
