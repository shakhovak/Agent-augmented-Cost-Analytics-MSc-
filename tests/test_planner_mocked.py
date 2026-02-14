from __future__ import annotations

import pytest

from cost_agent_mvp.agent.planner import Planner, PlannerPlan, PlannerValidationError


def _semantic_layer_fixture() -> dict:
    # Minimal semantic layer config for planner unit tests (kept self-contained).
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
        "derived_metrics": [
            {"name": "total_cost_services"},
            {"name": "active_users"},
        ],
        "allowed_operations": {
            "group_by": [
                "date",
                "account_id",
                "chat_type",
                "chat_id",
                "service_component",
            ],
            "filters": [
                {"name": "yesterday", "args": []},
                {"name": "single_day", "args": ["day"]},
                {"name": "last_n_days", "args": ["n_days"]},
                {"name": "account_id_in", "args": ["values"]},
            ],
            "charts": ["line_trend", "bar_top_n", "pie_composition", "table"],
        },
    }


def test_planner_mock_mode_returns_expected_plan(tmp_path) -> None:
    planner = Planner(prompts_dir=str(tmp_path))  # prompts not needed in mock_mode
    cfg = _semantic_layer_fixture()

    plan = planner.plan(
        user_text="Top 5 accounts by total service cost yesterday",
        semantic_layer=cfg,
        mock_mode=True,
    )

    assert plan.status == "OK"
    assert plan.intent == "top_n"
    assert plan.top_n == 5
    assert plan.group_by == ["account_id"]
    assert plan.metrics == ["total_cost_services"]
    assert plan.sort_by == "total_cost_services"
    assert plan.sort_dir == "desc"


def test_planner_mock_mode_invalid_plan_raises(tmp_path, monkeypatch) -> None:
    planner = Planner(prompts_dir=str(tmp_path))
    cfg = _semantic_layer_fixture()

    def _bad_mock_plan(user_text: str, view):  # type: ignore[no-untyped-def]
        # Metric is not allowed -> should be rejected by semantic validation
        return PlannerPlan(
            status="OK",
            intent="top_n",
            metrics=["revenue"],
            group_by=["account_id"],
            top_n=5,
            sort_by="revenue",
            sort_dir="desc",
        )

    monkeypatch.setattr(planner, "_mock_plan", _bad_mock_plan)

    with pytest.raises(PlannerValidationError) as e:
        planner.plan(user_text="Top accounts by revenue", semantic_layer=cfg, mock_mode=True)

    assert "Unsupported metric" in str(e.value)
