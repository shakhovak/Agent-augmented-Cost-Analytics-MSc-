from __future__ import annotations

import pytest

from cost_agent_mvp.agent.planner import Planner, PlannerPlan, PlannerValidationError


def _semantic_layer_fixture() -> dict:
    return {
        "version": 1,
        "limits": {
            "max_rows_hard": 20000,
            "max_rows_default": 2000,
            "max_days_hard": 180,
            "max_days_default": 30,
            "default_top_n": 10,
            "max_group_by_cols": 2,
            "drilldown_requires_filters": ["account_id"],
            "drilldown_max_days": 7,
        },
        "measures": [
            {"name": "total_cost_tasks"},
            {"name": "total_cost_classifications"},
            {"name": "cost_amocrm_call"},
        ],
        "derived_metrics": [{"name": "total_cost_services"}],
        "allowed_operations": {
            "group_by": [
                "date",
                "account_id",
                "chat_type",
                "chat_id",
                "service_component",
            ],
            "filters": [{"name": "account_id_in", "args": ["values"]}],
            "charts": ["bar_top_n", "table"],
        },
    }


def test_semantic_validation_rejects_unsupported_metric(tmp_path, monkeypatch) -> None:
    planner = Planner(prompts_dir=str(tmp_path))
    cfg = _semantic_layer_fixture()

    def _bad_mock_plan(user_text: str, view):  # type: ignore[no-untyped-def]
        return PlannerPlan(
            status="OK",
            intent="question",
            metrics=["made_up_metric"],
            group_by=["date"],
        )

    monkeypatch.setattr(planner, "_mock_plan", _bad_mock_plan)

    with pytest.raises(PlannerValidationError) as e:
        planner.plan(user_text="Trend of made_up_metric", semantic_layer=cfg, mock_mode=True)

    msg = str(e.value)
    assert "Unsupported metric" in msg
    assert "made_up_metric" in msg


def test_semantic_validation_rejects_unsupported_dimension(tmp_path, monkeypatch) -> None:
    planner = Planner(prompts_dir=str(tmp_path))
    cfg = _semantic_layer_fixture()

    def _bad_mock_plan(user_text: str, view):  # type: ignore[no-untyped-def]
        return PlannerPlan(
            status="OK",
            intent="breakdown",
            metrics=["total_cost_services"],
            group_by=["country"],
        )

    monkeypatch.setattr(planner, "_mock_plan", _bad_mock_plan)

    with pytest.raises(PlannerValidationError) as e:
        planner.plan(user_text="Breakdown by country", semantic_layer=cfg, mock_mode=True)

    msg = str(e.value)
    assert "Unsupported group_by dimension" in msg
    assert "country" in msg
