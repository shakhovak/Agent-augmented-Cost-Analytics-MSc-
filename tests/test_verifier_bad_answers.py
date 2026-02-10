from __future__ import annotations

import pandas as pd

from cost_agent_mvp.agent.verifier import verify_numeric_fidelity


def _evidence_fixture(*, delta_pct: float) -> dict[str, pd.DataFrame]:
    """
    Evidence fixture aligned to current pipeline:
    table name: 'kpis'
    expected delta column: total_cost_services_delta_pct
    """
    kpis = pd.DataFrame(
        [
            {
                "date": "2025-07-20",
                "total_cost_services": 1000.0,
                "total_cost_services_prev": 800.0,
                "total_cost_services_delta_abs": 200.0,
                "total_cost_services_delta_pct": delta_pct,  # 0.25 => 25.0%
                "active_users": 100.0,
                "active_users_prev": 90.0,
                "active_users_delta_pct": 0.111111,
            }
        ]
    )

    service_breakdown = pd.DataFrame(
        [
            {"label": "Tasks", "cost": 100.0, "share_of_total": 0.10},
            {"label": "Classifications", "cost": 700.0, "share_of_total": 0.70},
            {"label": "AmoCRM Calls", "cost": 200.0, "share_of_total": 0.20},
        ]
    )

    top_accounts = pd.DataFrame(
        [
            {"account_id": 1, "total_cost_services": 55.0, "share_of_total": 0.055},
            {"account_id": 2, "total_cost_services": 50.0, "share_of_total": 0.050},
        ]
    )

    return {
        "kpis": kpis,
        "service_breakdown": service_breakdown,
        "top_accounts": top_accounts,
    }


def test_verifier_fails_on_hallucinated_number() -> None:
    evidence = _evidence_fixture(delta_pct=0.25)

    bad_answer = (
        "📊 Daily Summary for date 2025-07-20\n\n"
        "- Total cost (services): 9999.00 (25.0% vs previous day)\n"
        "- Active users: 100 (11.1% vs previous day)\n"
    )

    res = verify_numeric_fidelity(bad_answer, evidence)
    assert res.passed is False
    assert any(i.type == "NUMBER_NOT_FOUND" for i in res.issues)


def test_verifier_fails_on_wrong_direction_claim() -> None:
    # Evidence says +25%, but answer claims it went down.
    evidence = _evidence_fixture(delta_pct=0.25)

    bad_answer = (
        "Total cost decreased vs yesterday.\nTotal cost (services): 1000 (25.0% vs previous day)\n"
    )

    res = verify_numeric_fidelity(bad_answer, evidence)
    assert res.passed is False
    assert any(i.type == "DIRECTION_MISMATCH" for i in res.issues)


def test_verifier_passes_on_good_answer_using_evidence_numbers() -> None:
    evidence = _evidence_fixture(delta_pct=-0.05)

    good_answer = (
        "📊 Daily Summary for date 2025-07-20\n\n"
        "Total cost fell vs yesterday.\n"
        "- Total cost (services): 1000.00 (-5.0% vs previous day)\n"
        "- Active users: 100 (11.1% vs previous day)\n\n"
        "Top services by cost:\n"
        "- Classifications: 700 (70.0%)\n"
        "- AmoCRM Calls: 200 (20.0%)\n"
        "- Tasks: 100 (10.0%)\n\n"
        "Top accounts by cost:\n"
        "- account 1: 55 (5.5%)\n"
        "- account 2: 50 (5.0%)\n"
    )

    res = verify_numeric_fidelity(good_answer, evidence)
    assert res.passed is True
    assert res.issues == []
