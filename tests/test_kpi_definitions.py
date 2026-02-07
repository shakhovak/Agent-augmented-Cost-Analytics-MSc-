import pandas as pd
import pytest

from cost_agent_mvp.core.errors import ValidationError
from src.cost_agent_mvp.analytics.kpi_definitions import (
    active_account_days,
    active_users,
    avg_cost_per_account_diluted,
    avg_cost_per_account_non_diluted,
    avg_cost_per_active_account_day,
    avg_cost_per_active_service_account,
    component_costs_sum,
    delta,
    delta_abs,
    delta_pct,
    dials_analyzed,
    distribution_stats,
    histogram_table,
    per_account_total_cost,
    rows_per_account_day_stats,
    service_mix_share,
    top_accounts,
    total_cost_sum,
)


def _df_small() -> pd.DataFrame:
    # Two accounts, two dates. Totals are present only on some rows,
    # plus extra “empty” chat rows to simulate chat-level grain.
    return pd.DataFrame(
        [
            # account 1, day 1: one cost row + one empty row
            {
                "account_id": 1,
                "date": "2025-01-01",
                "chat_id": "c1",
                "chat_type": "telegram",
                "total_cost": 10.0,
                "cost_dialog": 2.0,
                "total_cost_tasks": 3.0,
                "total_cost_classifications": 5.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 1,
                "has_classifications": 1,
            },
            {
                "account_id": 1,
                "date": "2025-01-01",
                "chat_id": "c2",
                "chat_type": "telegram",
                "total_cost": 0.0,
                "cost_dialog": 0.0,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 0,
                "has_classifications": 0,
            },
            # account 1, day 2: only empty rows (no paid activity)
            {
                "account_id": 1,
                "date": "2025-01-02",
                "chat_id": "c3",
                "chat_type": "whatsapp",
                "total_cost": 0.0,
                "cost_dialog": 0.0,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 0,
                "has_classifications": 0,
            },
            # account 2, day 1: one cost row (different component mix)
            {
                "account_id": 2,
                "date": "2025-01-01",
                "chat_id": "c4",
                "chat_type": "whatsapp",
                "total_cost": 4.0,
                "cost_dialog": 1.0,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 3.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 0,
                "has_classifications": 1,
            },
            # account 2, day 1: extra empty rows to test rows_per_account_day_stats
            {
                "account_id": 2,
                "date": "2025-01-01",
                "chat_id": "c5",
                "chat_type": "whatsapp",
                "total_cost": 0.0,
                "cost_dialog": 0.0,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 0,
                "has_classifications": 0,
            },
            {
                "account_id": 2,
                "date": "2025-01-01",
                "chat_id": "c6",
                "chat_type": "whatsapp",
                "total_cost": 0.0,
                "cost_dialog": 0.0,
                "total_cost_tasks": 0.0,
                "total_cost_classifications": 0.0,
                "cost_amocrm_call": 0.0,
                "has_tasks": 0,
                "has_classifications": 0,
            },
        ]
    )


def test_total_cost_sum():
    df = _df_small()
    assert total_cost_sum(df) == 14.0


def test_active_users():
    df = _df_small()
    assert active_users(df) == 2


def test_active_account_days():
    df = _df_small()
    # (1, 2025-01-01) and (2, 2025-01-01) have total_cost > 0
    assert active_account_days(df) == 2


def test_avg_cost_per_active_account_day():
    df = _df_small()
    # total_cost_sum = 14, active_account_days = 2
    assert avg_cost_per_active_account_day(df) == 7.0


def test_dials_analyzed():
    df = _df_small()
    # unique (account_id, chat_id, chat_type) among rows with total_cost>0: c1 and c4
    assert dials_analyzed(df) == 2


def test_component_costs_sum():
    df = _df_small()
    out = component_costs_sum(df)
    assert out["cost_dialog"] == 3.0
    assert out["total_cost_tasks"] == 3.0
    assert out["total_cost_classifications"] == 8.0
    # present in df, should be included
    assert out["cost_amocrm_call"] == 0.0


def test_service_mix_share():
    df = _df_small()
    mix = service_mix_share(df)
    assert set(mix.columns) == {"component", "cost", "share"}
    # shares should sum to 1 when total > 0 (floating tolerance)
    assert abs(float(mix["share"].sum()) - 1.0) < 1e-9
    # classification share = 8/11? careful: mix total is sum(component costs), not total_cost
    # component totals: dialog=3, tasks=3, cls=8, calls=0 => total=14
    cls_share = float(mix.loc[mix["component"] == "total_cost_classifications", "share"].iloc[0])
    assert cls_share == 8.0 / 14.0
    assert "cost_classification" not in set(mix["component"])


def test_avg_cost_per_account_diluted():
    df = _df_small()
    assert avg_cost_per_account_diluted(df) == 7.0  # 14 / 2 active users


def test_avg_cost_per_account_non_diluted_and_alias():
    df = _df_small()
    # tasks_mean: accounts with has_tasks==1 -> only account 1 -> sum(tasks)=3 => mean=3
    # cls_mean: accounts with has_classifications==1 -> account 1 sum=5, account 2 sum=3 => mean=4
    expected = 7.0
    assert avg_cost_per_account_non_diluted(df) == expected
    assert avg_cost_per_active_service_account(df) == expected


def test_per_account_total_cost():
    df = _df_small()
    out = per_account_total_cost(df).sort_values("account_id").reset_index(drop=True)
    assert list(out["account_id"]) == [1, 2]
    assert list(out["total_cost"]) == [10.0, 4.0]


def test_top_accounts():
    df = _df_small()
    out = top_accounts(df, n=2)
    assert list(out.columns) == [
        "rank",
        "account_id",
        "total_cost",
        "share_of_total_cost",
    ]
    assert list(out["rank"]) == [1, 2]
    assert list(out["account_id"]) == [1, 2]
    assert list(out["total_cost"]) == [10.0, 4.0]
    assert out.loc[0, "share_of_total_cost"] == 10.0 / 14.0


def test_rows_per_account_day_stats():
    df = _df_small()
    stats = rows_per_account_day_stats(df, cap=2)
    # Account-days present: (1,01-01)=2 rows, (1,01-02)=1 row, (2,01-01)=3 rows => max=3
    assert stats["max"] == 3.0
    assert stats["n_account_days"] == 3.0
    assert stats["n_over_cap"] == 1.0  # only (2,01-01) has 3>2
    assert stats["pct_over_cap"] == 1.0 / 3.0


def test_distribution_stats():
    s = pd.Series([0.0, 10.0, 20.0, 20.0])
    stats = distribution_stats(s)
    assert stats["mean"] == 12.5
    assert stats["median"] == 15.0
    assert stats["min"] == 0.0
    assert stats["max"] == 20.0


def test_histogram_table_counts_sum():
    s = pd.Series([0, 1, 1, 2, 2, 2], dtype=float)
    hist = histogram_table(s, bins=3)
    assert set(hist.columns) == {"bin_left", "bin_right", "count"}
    assert int(hist["count"].sum()) == 6


def test_delta_helpers():
    assert delta_abs(10, 7) == 3.0
    assert delta_pct(10, 5) == 1.0
    assert delta_pct(0, 0) == 0.0
    assert delta_pct(2, 0) == 1.0
    d_abs, d_pct = delta(10, 7)
    assert d_abs == 3.0
    assert abs(d_pct - (3.0 / 7.0)) < 1e-12


def test_missing_columns_raise():
    df = pd.DataFrame([{"account_id": 1}])
    with pytest.raises(ValidationError):
        total_cost_sum(df)
    with pytest.raises(ValidationError):
        active_account_days(df)
    with pytest.raises(ValidationError):
        rows_per_account_day_stats(df)


def test_empty_df_ok():
    df = pd.DataFrame(columns=["account_id", "date", "chat_id", "chat_type", "total_cost"])
    assert active_users(df) == 0
    assert active_account_days(df) == 0
    assert total_cost_sum(df) == 0.0
    assert dials_analyzed(df) == 0
