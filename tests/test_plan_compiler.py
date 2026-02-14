from __future__ import annotations

from datetime import date

import pytest

from cost_agent_mvp.agent.plan_compiler import (
    TimeWindowExceeded,
    compile_plan_to_queryspec,
)
from cost_agent_mvp.agent.planner import FilterSpec, PlannerPlan, TimeWindow
from cost_agent_mvp.core.constants import Aggregation, SortDirection
from cost_agent_mvp.core.errors import UnsupportedQuery
from cost_agent_mvp.core.types import TimeWindowType  # Required for type comparison


def _semantic_layer_min() -> dict:
    """
    Minimal semantic layer config for compiler tests.
    """
    return {
        "dimensions": [
            {"name": "date"},
            {"name": "account_id"},
            {"name": "chat_type"},
        ],
        "virtual_dimensions": [
            {"name": "service_component"},
        ],
        "measures": [
            {"name": "total_cost_tasks"},
            {"name": "total_cost_classifications"},
            {"name": "cost_amocrm_call"},
        ],
        "derived_metrics": [
            {"name": "total_cost_services"},
        ],
        "allowed_operations": {
            "time_windows": [
                "yesterday",
                "single_day",
                "last_n_days",
                "range",
            ],  # ✅ NO trailing spaces
            "group_by": ["date", "account_id", "chat_type"],
            "aggregations": ["sum", "mean", "median", "count", "count_distinct"],
            "filters": [
                {"name": "account_id_in"},
                {"name": "chat_id_in"},
                {"name": "chat_type_in"},
                {"name": "has_tasks"},
                {"name": "has_classifications"},
                {"name": "has_both"},
            ],
        },
        "limits": {
            "default_top_n": 10,
            "max_top_n": 50,
            "max_days_hard": 14,
            "max_filter_values_in": 1000,
            "max_rows_default": 5000,
            "max_rows_hard": 20000,
        },
    }


def test_compiler_happy_path_builds_queryspec() -> None:
    cfg = _semantic_layer_min()
    plan = PlannerPlan(
        status="OK",
        intent="top_n",
        time_window=TimeWindow(type="single_day", day="2025-07-20"),
        group_by=["account_id"],
        metrics=["sum(total_cost_tasks)"],
        filters=[FilterSpec(name="chat_type_in", args={"values": ["whatsapp"]})],
        top_n=5,
        sort_by="sum_total_cost_tasks",
        sort_dir="desc",
    )

    specs = compile_plan_to_queryspec(plan, cfg, anchor_day=date(2025, 7, 21))
    assert len(specs) == 1

    spec = specs[0]
    # ✅ Access attributes (not dict keys) - TimeWindow is an object
    # ✅ single_day compiles to RANGE type with matching start/end
    assert spec.time_window.type == TimeWindowType.RANGE
    assert spec.time_window.start == "2025-07-20"
    assert spec.time_window.end == "2025-07-20"

    assert spec.group_by == ["account_id"]
    assert len(spec.aggregations) == 1
    a0 = spec.aggregations[0]
    assert a0.field == "total_cost_tasks"
    assert a0.agg == Aggregation.SUM
    assert a0.as_name == "sum_total_cost_tasks"

    assert spec.top_n == 5
    assert spec.sort is not None
    assert spec.sort.by == "sum_total_cost_tasks"
    assert spec.sort.direction == SortDirection.DESC


def test_compiler_rejects_virtual_dimension() -> None:
    cfg = _semantic_layer_min()
    plan = PlannerPlan(
        status="OK",
        intent="breakdown",
        time_window=TimeWindow(type="single_day", day="2025-07-20"),
        group_by=["service_component"],  # virtual
        metrics=["sum(total_cost_tasks)"],
    )

    with pytest.raises(UnsupportedQuery):
        compile_plan_to_queryspec(plan, cfg, anchor_day=date(2025, 7, 21))


def test_compiler_rejects_derived_metric() -> None:
    cfg = _semantic_layer_min()
    plan = PlannerPlan(
        status="OK",
        intent="question",
        time_window=TimeWindow(type="single_day", day="2025-07-20"),
        group_by=["account_id"],
        metrics=["total_cost_services"],  # derived metric
    )

    with pytest.raises(UnsupportedQuery):
        compile_plan_to_queryspec(plan, cfg, anchor_day=date(2025, 7, 21))


def test_compiler_time_window_exceeded() -> None:
    cfg = _semantic_layer_min()
    plan = PlannerPlan(
        status="OK",
        intent="trend",
        time_window=TimeWindow(type="last_n_days", n_days=99),  # 99 days > max_days_hard=14
        metrics=["sum(total_cost_tasks)"],
        group_by=["date"],
    )

    # ✅ Now raises TimeWindowExceeded because:
    # 1. "last_n_days" (stripped) matches "last_n_days" in allowed_operations (no trailing space)
    # 2. Compiler proceeds to check span_days=99 > max_days_hard=14 → raises TimeWindowExceeded
    with pytest.raises(TimeWindowExceeded):
        compile_plan_to_queryspec(plan, cfg, anchor_day=date(2025, 7, 21))
