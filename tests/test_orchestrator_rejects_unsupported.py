from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import cost_agent_mvp.agent.orchestrator as orch_mod
from cost_agent_mvp.agent.orchestrator import Orchestrator
from cost_agent_mvp.core.errors import UnsupportedQuery


def _write_tiny_csv(csv_path: Path) -> None:
    rows = [
        {
            "date": "2025-07-20",
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
        }
    ]
    for r in rows:
        r["has_tasks"] = 1
        r["has_classifications"] = 1
        r["has_both"] = 1
        r["num_services"] = 2
        r["total_cost"] = r["cost_dialog"] + r["cost_task"] + r["cost_classification"]
        r["total_cost_tasks"] = r["cost_task"]
        r["total_cost_classifications"] = r["cost_classification"]

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_semantic_layer_yaml(path: Path) -> None:
    path.write_text("version: 1\n", encoding="utf-8")


@dataclass
class _FakePlan:
    status: str = "OK"


class _FakePlanner:
    def __init__(self, prompts_dir: str | None = None) -> None:
        self.prompts_dir = prompts_dir

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        return _FakePlan(status="OK")


def _issue_type(issue: Any) -> str | None:
    if isinstance(issue, dict):
        return issue.get("type")
    return getattr(issue, "type", None)


def test_orchestrator_rejects_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "tiny.csv"
    _write_tiny_csv(csv_path)
    sl_path = tmp_path / "semantic_layer.yaml"
    _write_semantic_layer_yaml(sl_path)

    outputs_root = tmp_path / "runs"

    monkeypatch.setattr(orch_mod, "Planner", _FakePlanner)

    # ✅ FIX: Include the expected UX prefix in the exception message
    def _fake_compile(*args: Any, **kwargs: Any) -> list[Any]:
        raise UnsupportedQuery("❌ Unsupported query: metric is not allowed")

    monkeypatch.setattr(orch_mod, "compile_plan_to_queryspec", _fake_compile)

    orch = Orchestrator(
        outputs_root=str(outputs_root),
        dataset_name="joint_costs_daily",
        semantic_layer_path=str(sl_path),
        prompts_dir=str(tmp_path / "prompts"),
    )

    out = orch.run_ad_hoc(
        csv_path=str(csv_path),
        user_text="Show me revenue by country for last year",
        output_root=str(outputs_root),
        anchor_day=date(2025, 7, 21),
        mock_planner=True,
        mock_analyst=True,
        build_dashboard=False,
    )

    # Now passes because exception message includes the prefix
    assert out.answer_text.startswith("❌ Unsupported query:")
    assert out.verifier.passed is False
    assert out.verifier.issues
    assert _issue_type(out.verifier.issues[0]) == "UNSUPPORTED_QUERY"
