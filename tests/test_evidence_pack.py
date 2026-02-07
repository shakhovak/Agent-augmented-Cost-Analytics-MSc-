from __future__ import annotations

import pandas as pd
import pytest

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.core.errors import ValidationError
from cost_agent_mvp.data.schema import load_csv  # or CsvBackend if you prefer


def _pick_report_day(df: pd.DataFrame):
    df_nonzero = df[df["total_cost"].fillna(0) > 0]
    if not df_nonzero.empty:
        return df_nonzero["date"].max()
    return df["date"].max()


def test_evidence_pack_contract_small_sample(small_sample_csv_path: str):
    df = load_csv(small_sample_csv_path)
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df,
        report_day=report_day,
        trend_days=7,
        top_n=5,
        cap=50,
    )

    # Required v0 tables
    for k in ["kpis", "trend_daily", "service_breakdown", "top_accounts"]:
        assert k in evidence, f"Missing required evidence table: {k}"

    # Column contracts
    kpis = evidence["kpis"]
    assert len(kpis) == 1
    for col in [
        "date",
        "total_cost",
        "total_cost_prev",
        "total_cost_delta_abs",
        "total_cost_delta_pct",
    ]:
        assert col in kpis.columns

    trend = evidence["trend_daily"]
    for col in ["date", "total_cost", "active_users"]:
        assert col in trend.columns

    svc = evidence["service_breakdown"]
    for col in ["date", "component", "cost", "share_of_total"]:
        assert col in svc.columns

    top = evidence["top_accounts"]
    for col in ["date", "rank", "account_id", "total_cost", "share_of_total_cost"]:
        assert col in top.columns


def test_kpi_total_cost_matches_day_sum(small_sample_csv_path: str):
    df = load_csv(small_sample_csv_path)
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df, report_day=report_day, trend_days=7, top_n=5
    )
    reported = float(evidence["kpis"]["total_cost"].iloc[0])
    true = float(df[df["date"] == report_day]["total_cost"].sum())
    assert abs(reported - true) < 1e-9


def test_missing_report_day_raises(small_sample_csv_path: str):
    df = load_csv(small_sample_csv_path)

    with pytest.raises(ValidationError):
        build_standard_daily_evidence(
            df_all=df,
            report_day=pd.to_datetime("2030-01-01").date(),
            trend_days=7,
            top_n=5,
        )
