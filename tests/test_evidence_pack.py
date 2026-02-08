from __future__ import annotations

from datetime import timedelta

import pandas as pd

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import load_csv  # or CsvBackend if you prefer
from cost_agent_mvp.viz.chart_specs import SERVICE_COST_COLUMNS


def _pick_report_day(df: pd.DataFrame):
    """Pick a day that is likely to have non-zero *service* cost."""
    svc_total = df[list(SERVICE_COST_COLUMNS)].sum(axis=1).fillna(0)
    df_nonzero = df[svc_total > 0]
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
        top_accounts_n=5,
        hist_bins=30,
    )

    # Required v1 tables
    for k in [
        "kpis",
        "trend_daily",
        "service_breakdown",
        "service_totals",
        "top_accounts",
        "distribution_hist",
        "distribution_stats",
        "service_combinations",
        "dials_cost_trend",
        "exceptions_queue",
    ]:
        assert k in evidence, f"Missing required evidence table: {k}"

    # Column contracts
    kpis = evidence["kpis"]
    assert len(kpis) == 1
    for col in [
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
    ]:
        assert col in kpis.columns

    trend = evidence["trend_daily"]
    for col in [
        "date",
        "active_users",
        "total_cost_services",
        "dials_analyzed",
        "avg_cost_per_active_account_day",
        "p95_cost_per_account_day",
        "pct_accounts_above_cap",
        "avg_margin_per_account_day",
    ]:
        assert col in trend.columns

    svc = evidence["service_breakdown"]
    for col in ["date", "component", "label", "cost", "share_of_total"]:
        assert col in svc.columns

    top = evidence["top_accounts"]
    for col in [
        "date",
        "rank",
        "account_id",
        "total_cost_services",
        "share_of_total_cost",
    ]:
        assert col in top.columns


def test_kpi_total_cost_services_matches_day_sum(small_sample_csv_path: str):
    df = load_csv(small_sample_csv_path)
    report_day = _pick_report_day(df)

    evidence = build_standard_daily_evidence(
        df_all=df, report_day=report_day, trend_days=7, top_accounts_n=5
    )
    reported = float(evidence["kpis"]["total_cost_services"].iloc[0])
    true = float(df[df["date"] == report_day][list(SERVICE_COST_COLUMNS)].sum().sum())
    assert abs(reported - true) < 1e-9


def test_future_report_day_returns_zeroed_kpis_when_no_data(small_sample_csv_path: str):
    df = load_csv(small_sample_csv_path)
    report_day = df["date"].max() + timedelta(days=365)

    evidence = build_standard_daily_evidence(
        df_all=df, report_day=report_day, trend_days=7, top_accounts_n=5
    )

    # We still get a valid pack; KPI values for that day should be zero.
    kpis = evidence["kpis"].iloc[0]
    assert float(kpis["active_users"]) == 0.0
    assert float(kpis["total_cost_services"]) == 0.0
    assert float(kpis["dials_analyzed"]) == 0.0
