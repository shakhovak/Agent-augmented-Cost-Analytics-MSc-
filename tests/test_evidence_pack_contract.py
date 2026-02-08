from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import load_csv
from cost_agent_mvp.viz.chart_specs import SERVICE_COST_COLUMNS

# Contract ...
REQUIRED_TABLES: dict[str, set[str]] = {
    "kpis": {
        "date",
        "active_users",
        "active_users_prev",
        "active_users_delta_abs",
        "active_users_delta_pct",
        "new_users",
        "churned_users",
        "net_user_change",
        "avg_cost_per_active_account_day",
        "avg_cost_per_active_account_day_prev",
        "avg_cost_per_active_account_day_delta_abs",
        "avg_cost_per_active_account_day_delta_pct",
        "p95_cost_per_account_day",
        "p95_cost_per_account_day_prev",
        "p95_cost_per_account_day_delta_abs",
        "p95_cost_per_account_day_delta_pct",
        "pct_accounts_above_cap",
        "pct_accounts_above_cap_prev",
        "pct_accounts_above_cap_delta_abs",
        "pct_accounts_above_cap_delta_pct",
        "avg_margin_per_account_day",
        "avg_margin_per_account_day_prev",
        "avg_margin_per_account_day_delta_abs",
        "avg_margin_per_account_day_delta_pct",
        "total_cost_services",
        "total_cost_services_prev",
        "total_cost_services_delta_abs",
        "total_cost_services_delta_pct",
        "dials_analyzed",
        "dials_analyzed_prev",
        "dials_analyzed_delta_abs",
        "dials_analyzed_delta_pct",
    },
    "trend_daily": {
        "date",
        "active_users",
        "total_cost_services",
        "dials_analyzed",
        "avg_cost_per_active_account_day",
        "p95_cost_per_account_day",
        "pct_accounts_above_cap",
        "avg_margin_per_account_day",
    },
    "service_breakdown": {
        "date",
        "component",
        "label",
        "cost",
        "share_of_total",
    },
    "service_totals": {
        "date",
        "label",
        "cost",
    },
    "top_accounts": {
        "date",
        "rank",
        "account_id",
        "total_cost_services",
        "share_of_total_cost",
    },
    "distribution_hist": {
        "bin_left",
        "bin_right",
        "count",
    },
    "distribution_stats": {
        "median",
        "min",
        "max",
        "std",
        "n",
    },
    "service_combinations": {
        "date",
        "label",
        "active_users",
    },
    "dials_cost_trend": {
        "date",
        "dials_analyzed",
        "total_cost_services",
    },
    "exceptions_queue": {
        "message",
    },
}


def _pick_report_day(df: pd.DataFrame):
    svc_total = df[list(SERVICE_COST_COLUMNS)].sum(axis=1).fillna(0)
    df_nonzero = df[svc_total > 0]
    if not df_nonzero.empty:
        return df_nonzero["date"].max()
    return df["date"].max()


def _assert_contract(evidence: dict[str, pd.DataFrame]) -> None:
    missing_tables = [k for k in REQUIRED_TABLES if k not in evidence]
    assert not missing_tables, f"Evidence pack missing required tables: {missing_tables}"

    missing_cols: dict[str, list[str]] = {}
    for table_key, required_cols in REQUIRED_TABLES.items():
        df = evidence[table_key]
        cols = set(df.columns)
        miss = sorted(required_cols - cols)
        if miss:
            missing_cols[table_key] = miss

    assert not missing_cols, f"Evidence pack missing required columns: {missing_cols}"


def test_evidence_pack_contract_enforced():
    path = Path("data/samples/joint_info_sample.csv")
    df = load_csv(str(path))
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df,
        report_day=report_day,
        trend_days=7,
        top_accounts_n=10,
        hist_bins=30,
    )

    _assert_contract(evidence)


def test_contract_failure_message_is_explicit():
    # Build a valid pack, then simulate drift by removing a required column
    path = Path("data/samples/joint_info_sample.csv")
    df = load_csv(str(path))
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df,
        report_day=report_day,
        trend_days=7,
        top_accounts_n=10,
        hist_bins=30,
    )

    # Simulate schema drift:
    evidence["kpis"] = evidence["kpis"].drop(columns=["total_cost_services"], errors="ignore")

    with pytest.raises(AssertionError) as e:
        _assert_contract(evidence)

    msg = str(e.value)
    assert "kpis" in msg and "total_cost_services" in msg
