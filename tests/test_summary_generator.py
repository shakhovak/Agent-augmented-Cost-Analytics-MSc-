from __future__ import annotations

import pandas as pd

from cost_agent_mvp.reports.summary_generator import generate_daily_summary


def _make_evidence_pack_fixture() -> dict[str, pd.DataFrame]:
    """
    Minimal Evidence Pack v0 fixture for summary generation.

    Keys expected by the updated summary_generator:
      - "kpis"
      - "service_breakdown"
      - "top_accounts"
      - "trend_daily"
    """
    kpis = pd.DataFrame(
        [
            {
                "date": "2025-07-20",
                "total_cost_services": 5815.661900058649,
                "total_cost_services_delta_pct": 0.10,  # +10.0% vs prev day
                "active_users": 479,
                "active_users_delta_pct": 0.05,  # +5.0%
                "new_users": 180,
                "churned_users": 220,
                "net_user_change": -40,
                "dials_analyzed": 1234,
                "dials_analyzed_delta_pct": -0.02,  # -2.0%
            }
        ]
    )

    service_breakdown = pd.DataFrame(
        [
            {
                "label": "Classifications",
                "cost": 3000.0,
                "share_of_total": 3000.0 / 5815.661900058649,
            },
            {
                "label": "Tasks",
                "cost": 2000.0,
                "share_of_total": 2000.0 / 5815.661900058649,
            },
            {
                "label": "AmoCRM Calls",
                "cost": 815.661900058649,
                "share_of_total": 815.661900058649 / 5815.661900058649,
            },
        ]
    )

    top_accounts = pd.DataFrame(
        [
            {
                "account_id": 101,
                "total_cost_services": 150.0,
                "share_of_total_cost": 150.0 / 5815.661900058649,
            },
            {
                "account_id": 202,
                "total_cost_services": 120.0,
                "share_of_total_cost": 120.0 / 5815.661900058649,
            },
            {
                "account_id": 303,
                "total_cost_services": 90.0,
                "share_of_total_cost": 90.0 / 5815.661900058649,
            },
        ]
    )

    trend_daily = pd.DataFrame(
        [
            {"date": "2025-07-14", "avg_cost_per_active_account_day": 10.0},
            {"date": "2025-07-15", "avg_cost_per_active_account_day": 10.5},
            {"date": "2025-07-16", "avg_cost_per_active_account_day": 11.0},
            {"date": "2025-07-17", "avg_cost_per_active_account_day": 11.2},
            {"date": "2025-07-18", "avg_cost_per_active_account_day": 11.4},
            {"date": "2025-07-19", "avg_cost_per_active_account_day": 11.8},
            {"date": "2025-07-20", "avg_cost_per_active_account_day": 12.0},
        ]
    )

    return {
        "kpis": kpis,
        "service_breakdown": service_breakdown,
        "top_accounts": top_accounts,
        "trend_daily": trend_daily,
    }


def test_generate_daily_summary_contains_required_sections_and_values() -> None:
    evidence = _make_evidence_pack_fixture()

    text = generate_daily_summary(evidence=evidence, top_n=3)

    # --- Key sections / headings
    assert "Daily Cost Monitoring Summary" in text
    assert "Total cost (services):" in text
    assert "Top 3 services by cost:" in text
    assert "Top 3 accounts by cost:" in text
    assert "7-day trend" in text

    # --- Specific known values (formatted)
    # total cost formatted with thousands separator and 2 decimals
    assert "Total cost (services): 5,815.66" in text
    # delta pct is fraction -> percent string
    assert "10.0% vs previous day" in text

    # active users and churn snippet (optional but present in fixture)
    assert "Active users: 479" in text
    assert "User churn: new=180, churned=220, net=-40" in text

    # dials analyzed snippet
    assert "Dials analyzed: 1,234" in text
    assert "(-2.0% vs previous day)" in text

    # top services include labels + costs (2 decimals)
    assert "- Classifications: 3,000.00" in text
    assert "- Tasks: 2,000.00" in text
    assert "- AmoCRM Calls: 815.66" in text

    # top accounts include IDs + costs
    assert "- account 101: 150.00" in text
    assert "- account 202: 120.00" in text
    assert "- account 303: 90.00" in text

    # trend note should be "up" and show start→end values
    assert "7-day trend (avg_cost_per_active_account_day): up" in text
    assert "(10.00 → 12.00)" in text


def test_generate_daily_summary_empty_kpis_is_non_empty_and_explains() -> None:
    evidence = {
        "kpis": pd.DataFrame(),
        "service_breakdown": pd.DataFrame(),
        "top_accounts": pd.DataFrame(),
        "trend_daily": pd.DataFrame(),
    }

    text = generate_daily_summary(evidence=evidence, top_n=3)

    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert "KPI table is empty" in text
