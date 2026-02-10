from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cost_agent_mvp.agent.analyst import Analyst


def _evidence_fixture() -> dict[str, pd.DataFrame]:
    return {
        "kpis": pd.DataFrame(
            [
                {
                    "date": "2025-07-20",
                    "total_cost_services": 100.0,
                    "active_users": 10,
                    "avg_cost_per_active_account_day": 10.0,
                }
            ]
        ),
        "top_accounts": pd.DataFrame(
            [
                {"account_id": 1, "total_cost_services": 40.0},
                {"account_id": 2, "total_cost_services": 35.0},
            ]
        ),
    }


def test_analyst_mock_mode_is_deterministic(tmp_path: Path) -> None:
    evidence = _evidence_fixture()

    analyst = Analyst()
    r1 = analyst.generate(evidence=evidence, mock_mode=True, mode="button", user_text="")
    r2 = analyst.generate(evidence=evidence, mock_mode=True, mode="button", user_text="")

    assert r1.model == "mock"
    assert r1.answer_text
    assert r1.answer_text == r2.answer_text
    assert "Tables:" in r1.answer_text
    assert "[kpis]" in r1.answer_text or "kpis" in r1.answer_text
    assert "2025-07-20" in r1.answer_text  # appears in CSV head snippet


def test_analyst_mock_mode_includes_question_when_provided() -> None:
    evidence = _evidence_fixture()
    analyst = Analyst()
    r = analyst.generate(
        evidence=evidence,
        mock_mode=True,
        mode="qa",
        user_text="Why did cost increase?",
    )
    assert "Question:" in r.answer_text
    assert "Why did cost increase?" in r.answer_text


def test_analyst_raises_on_empty_evidence() -> None:
    analyst = Analyst()
    with pytest.raises(ValueError):
        analyst.generate(evidence={}, mock_mode=True)
