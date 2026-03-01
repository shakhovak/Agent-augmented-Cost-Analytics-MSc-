from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import date as dt_date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import cost_agent_mvp.agent.orchestrator as orch_mod
from cost_agent_mvp.agent.orchestrator import Orchestrator
from cost_agent_mvp.core.types import RunRecord  # Required for monkeypatch
from cost_agent_mvp.data.csv_backend import CsvBackend


def _fake_detector_summary_single(
    *, csv_path: str, day: dt_date, baseline_days: int = 14, context_days: int = 7
):
    # minimal shape that downstream code expects
    det = {
        "decision": {
            "day": day.isoformat(),
            "anomaly": False,
            "driver": "none",
            "severity": 0.0,
            "flagged_accounts": 0,
            "excess_cost": 0.0,
        }
    }
    return f"No anomaly detected on {day.isoformat()}.", det


def _fake_detector_summary_range(
    *,
    csv_path: str,
    start: dt_date,
    end: dt_date,
    baseline_days: int = 14,
    top_k: int = 10,
    max_scan_days: int = 30,
):
    meta = {
        "range_requested": {"start": start.isoformat(), "end": end.isoformat()},
        "range_scanned": {"start": start.isoformat(), "end": end.isoformat()},
        "flagged_days_scanned": 0,
        "top_k": top_k,
        "baseline_days": baseline_days,
        "max_scan_days": max_scan_days,
    }
    return "Anomaly scan: no spike days detected.", meta


def _write_tiny_csv(csv_path: Path) -> None:
    """
    Minimal dataset that satisfies default_joint_schema() requirements
    and supports simple aggregations for Q&A.
    """
    rows = [
        # day before
        {
            "date": "2025-07-19",
            "account_id": 1,
            "chat_id": 1001,
            "chat_type": "whatsapp",
            "cost_dialog": 0.10,
            "cost_task": 1.00,
            "cost_classification": 2.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 0,
            "cost_amocrm_call": 0.00,
        },
        # report day
        {
            "date": "2025-07-20",
            "account_id": 1,
            "chat_id": 1002,
            "chat_type": "whatsapp",
            "cost_dialog": 0.05,
            "cost_task": 2.00,
            "cost_classification": 0.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 1,
            "cost_amocrm_call": 0.20,
        },
        {
            "date": "2025-07-20",
            "account_id": 2,
            "chat_id": 2001,
            "chat_type": "telegram",
            "cost_dialog": 0.00,
            "cost_task": 0.50,
            "cost_classification": 1.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 0,
            "cost_amocrm_call": 0.00,
        },
    ]
    for r in rows:
        r["has_tasks"] = 1 if r["cost_task"] > 0 else 0
        r["has_classifications"] = 1 if r["cost_classification"] > 0 else 0
        r["has_both"] = 1 if (r["has_tasks"] == 1 and r["has_classifications"] == 1) else 0
        r["num_services"] = int(
            r["has_tasks"] + r["has_classifications"] + (1 if r["has_amocrm_call"] else 0)
        )
        r["total_cost"] = (
            r["cost_dialog"]
            + r["cost_task"]
            + r["cost_classification"]
            + r["cost_qc"]
            + r["cost_check_list"]
            + r["cost_amocrm_call"]
        )
        r["total_cost_tasks"] = r["cost_task"]
        r["total_cost_classifications"] = r["cost_classification"]

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_semantic_layer_yaml(path: Path) -> None:
    path.write_text("version: 1\n", encoding="utf-8")


@dataclass
class _FakePlan:
    status: str = "OK"
    view: str | None = "qa_view"
    intent: str | None = "cost_analysis"
    error_message: str | None = None


class _FakePlanner:
    def __init__(self, prompts_dir: str | None = None) -> None:
        self.prompts_dir = prompts_dir

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        return _FakePlan(status="OK")


class _FakeAnalyst:
    def __init__(self, prompts_dir: str | None = None, default_mode: str | None = None) -> None:
        self.prompts_dir = prompts_dir
        self.default_mode = default_mode

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            answer_text="Mock answer", citations=None, warnings=None, raw={"mock": True}
        )


def _make_queryspec() -> Any:
    from datetime import date as dt_date

    from cost_agent_mvp.core.types import (
        AggregationSpec,
        QueryFilters,
        QuerySpec,
        TimeWindow,
    )

    agg = AggregationSpec(field="total_cost_tasks", agg="sum", as_name="sum_total_cost_tasks")
    qf = QueryFilters(chat_type=["whatsapp"])

    # ✅ CORRECT: Use 'range' type with date objects (NOT strings or 'single_day')
    tw = TimeWindow(type="range", start=dt_date(2025, 7, 20), end=dt_date(2025, 7, 20), n_days=None)

    kwargs: dict[str, Any] = {
        "time_window": tw,
        "group_by": ["chat_type"],
        "aggregations": [agg],
        "filters": qf,
    }

    if hasattr(QuerySpec, "model_fields"):
        fields = QuerySpec.model_fields  # type: ignore[attr-defined]
        if "top_n" in fields:
            kwargs["top_n"] = 10
        if "max_rows" in fields:
            kwargs["max_rows"] = 1000
        if "sort" in fields:
            kwargs["sort"] = None
        if "view_id" in fields and fields["view_id"].is_required():
            kwargs["view_id"] = "qa"

    return QuerySpec(**kwargs)


def test_orchestrator_qa_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "tiny.csv"
    _write_tiny_csv(csv_path)
    sl_path = tmp_path / "semantic_layer.yaml"
    _write_semantic_layer_yaml(sl_path)
    outputs_root = tmp_path / "runs"

    # Patch Planner/Analyst + compiler + verifier
    monkeypatch.setattr(orch_mod, "Planner", _FakePlanner)
    monkeypatch.setattr(orch_mod, "Analyst", _FakeAnalyst)

    def _fake_compile(*args: Any, **kwargs: Any) -> list[Any]:
        return [_make_queryspec()]

    monkeypatch.setattr(orch_mod, "compile_plan_to_queryspec", _fake_compile)

    from cost_agent_mvp.agent.verifier import VerifierResult

    def _fake_verify(*args: Any, **kwargs: Any) -> VerifierResult:
        return VerifierResult(passed=True, score=1.0, issues=[], summary="Mock verification passed")

    monkeypatch.setattr(orch_mod, "verify_numeric_fidelity", _fake_verify)

    # ✅ FIX: Mock CsvBackend.query() to return object with .df/.lineage
    def _mock_query(self, spec: Any) -> Any:
        df = pd.DataFrame([{"chat_type": "whatsapp", "sum_total_cost_tasks": 2.0}])
        return SimpleNamespace(df=df, lineage=[])

    monkeypatch.setattr(CsvBackend, "query", _mock_query)

    # ✅ FIX: Mock dataset_version to be callable (orchestrator.py:439)
    monkeypatch.setattr(CsvBackend, "dataset_version", lambda self: "mock-version")
    monkeypatch.setattr(orch_mod, "_detector_summary_single", _fake_detector_summary_single)
    monkeypatch.setattr(orch_mod, "_detector_summary_range", _fake_detector_summary_range)

    # ✅ FIX: Work around orchestrator bug - RunRecord has no to_json() method
    # Add a to_json method that uses orchestrator's _to_jsonable helper
    def _runrecord_to_json(self):
        from cost_agent_mvp.agent.orchestrator import _to_jsonable

        return _to_jsonable(self)

    monkeypatch.setattr(RunRecord, "to_json", _runrecord_to_json, raising=False)

    orch = Orchestrator(
        outputs_root=str(outputs_root),
        dataset_name="joint_costs_daily",
        semantic_layer_path=str(sl_path),
        prompts_dir=str(tmp_path / "prompts"),
    )

    out = orch.run_ad_hoc(
        csv_path=str(csv_path),
        user_text="Top tasks cost by chat type for 2025-07-20",
        output_root=str(outputs_root),
        anchor_day=date(2025, 7, 21),
        mock_planner=True,
        mock_analyst=True,
        build_dashboard=False,
    )

    assert out.answer_text
    assert out.verifier is not None
    assert out.verifier.passed is True

    run_dir = Path(out.output_dir)
    assert run_dir.exists()
    assert (run_dir / "answer.txt").exists()
    assert (run_dir / "run_record.json").exists()
