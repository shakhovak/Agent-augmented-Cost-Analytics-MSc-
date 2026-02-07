from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import load_csv

REQUIRED_TABLES: dict[str, set[str]] = {
    "kpis": {
        "date",
        "total_cost",
        "total_cost_prev",
        "total_cost_delta_abs",
        "total_cost_delta_pct",
    },
    "trend_daily": {
        "date",
        "total_cost",
        "active_users",
        "active_account_days",
        "dials_analyzed",
    },
    "service_breakdown": {
        "date",
        "component",
        "cost",
        "share_of_total",
    },
    "top_accounts": {
        "date",
        "rank",
        "account_id",
        "total_cost",
        "share_of_total_cost",
    },
}


def _pick_report_day(df: pd.DataFrame):
    df_nonzero = df[df["total_cost"].fillna(0) > 0]
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
        top_n=10,
        cap=50,
    )

    _assert_contract(evidence)


def test_contract_failure_message_is_explicit():
    # Build a valid pack, then simulate drift by removing a required column
    path = Path("data/samples/joint_info_sample.csv")
    df = load_csv(str(path))
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df, report_day=report_day, trend_days=7, top_n=10
    )

    # Simulate schema drift:
    evidence["kpis"] = evidence["kpis"].drop(columns=["total_cost"], errors="ignore")

    with pytest.raises(AssertionError) as e:
        _assert_contract(evidence)

    msg = str(e.value)
    assert "kpis" in msg and "total_cost" in msg
