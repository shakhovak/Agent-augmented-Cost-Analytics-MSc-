from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from cost_agent_mvp.reports.daily_report import run_standard_daily_report


def _write_tiny_csv(csv_path: Path) -> None:
    """
    Minimal dataset that satisfies default_joint_schema() column requirements
    AND includes the service cost totals used by dashboard/KPIs.
    """
    rows = [
        # -------- 2025-07-19 (previous day) --------
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
        {
            "date": "2025-07-19",
            "account_id": 1,
            "chat_id": 1002,
            "chat_type": "whatsapp",
            "cost_dialog": 0.05,
            "cost_task": 0.50,
            "cost_classification": 0.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 1,
            "cost_amocrm_call": 0.20,
        },
        {
            "date": "2025-07-19",
            "account_id": 2,
            "chat_id": 2001,
            "chat_type": "telegram",
            "cost_dialog": 0.00,
            "cost_task": 0.00,
            "cost_classification": 1.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 0,
            "cost_amocrm_call": 0.00,
        },
        # -------- 2025-07-20 (report day) --------
        {
            "date": "2025-07-20",
            "account_id": 1,
            "chat_id": 1003,
            "chat_type": "whatsapp",
            "cost_dialog": 0.10,
            "cost_task": 2.00,
            "cost_classification": 1.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 1,
            "cost_amocrm_call": 0.50,
        },
        {
            "date": "2025-07-20",
            "account_id": 2,
            "chat_id": 2002,
            "chat_type": "telegram",
            "cost_dialog": 0.00,
            "cost_task": 0.00,
            "cost_classification": 3.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 0,
            "cost_amocrm_call": 0.00,
        },
        {
            "date": "2025-07-20",
            "account_id": 3,
            "chat_id": 3001,
            "chat_type": "whatsapp",
            "cost_dialog": 0.20,
            "cost_task": 1.00,
            "cost_classification": 0.00,
            "cost_qc": 0.00,
            "cost_check_list": 0.00,
            "has_amocrm_call": 1,
            "cost_amocrm_call": 1.50,
        },
    ]

    # Add required derived fields per schema + convenience totals used by KPIs
    for r in rows:
        r["has_tasks"] = 1 if r["cost_task"] > 0 else 0
        r["has_classifications"] = 1 if r["cost_classification"] > 0 else 0
        r["has_both"] = 1 if (r["has_tasks"] == 1 and r["has_classifications"] == 1) else 0

        # num_services is a simple count of used services (tasks/classifications/call)
        r["num_services"] = int(
            r["has_tasks"] + r["has_classifications"] + (1 if r["has_amocrm_call"] else 0)
        )

        # total_cost required by schema; include all components + call cost
        r["total_cost"] = (
            r["cost_dialog"]
            + r["cost_task"]
            + r["cost_classification"]
            + r["cost_qc"]
            + r["cost_check_list"]
            + r["cost_amocrm_call"]
        )

        # Columns used by dashboard/KPI logic (your newer pipeline)
        r["total_cost_tasks"] = r["cost_task"]
        r["total_cost_classifications"] = r["cost_classification"]
        # already have cost_amocrm_call

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def _write_report_templates_yaml(path: Path) -> None:
    """
    Minimal report_templates.yaml for the orchestrator.
    """
    yaml_text = """
templates:
  standard_daily_report:
    title: "Standard Daily Report"
    description: "Test template"
    constraints:
      top_n: 5
      max_days: 8
"""
    path.write_text(yaml_text.strip() + "\n", encoding="utf-8")


def test_daily_report_orchestrator_smoke(tmp_path: Path) -> None:
    # Arrange
    csv_path = tmp_path / "tiny.csv"
    _write_tiny_csv(csv_path)

    templates_path = tmp_path / "report_templates.yaml"
    _write_report_templates_yaml(templates_path)

    output_root = tmp_path / "runs"

    # Act
    artifacts = run_standard_daily_report(
        csv_path=str(csv_path),
        report_day=date.fromisoformat("2025-07-20"),
        output_root=str(output_root),
        report_templates_path=str(templates_path),
        template_id="standard_daily_report",
    )

    # Assert: output directory exists
    run_dir = Path(artifacts.output_dir)
    assert run_dir.exists() and run_dir.is_dir()

    # Assert: primary artifacts exist and are non-empty
    dashboard_png = Path(artifacts.dashboard_png)
    summary_txt = Path(artifacts.summary_txt)
    run_record_json = Path(artifacts.run_record_json)
    evidence_dir = Path(artifacts.evidence_dir)

    assert dashboard_png.exists() and dashboard_png.stat().st_size > 0
    assert summary_txt.exists() and summary_txt.stat().st_size > 0
    assert run_record_json.exists() and run_record_json.stat().st_size > 0
    assert evidence_dir.exists() and evidence_dir.is_dir()

    # Evidence CSVs exist and are non-empty
    evidence_csvs = sorted(evidence_dir.glob("*.csv"))
    assert len(evidence_csvs) > 0
    assert all(p.stat().st_size > 0 for p in evidence_csvs)

    # Assert: run_record.json has required keys and paths
    rr = json.loads(run_record_json.read_text(encoding="utf-8"))

    assert rr["run_id"] == artifacts.run_id
    assert "started_at_utc" in rr and rr["started_at_utc"]
    assert "finished_at_utc" in rr and rr["finished_at_utc"]
    assert rr["template_id"] == "standard_daily_report"

    assert "artifacts" in rr
    art = rr["artifacts"]
    assert Path(art["dashboard_png"]).exists()
    assert Path(art["summary_txt"]).exists()
    assert Path(art["run_record_json"]).exists()
    assert Path(art["evidence_dir"]).exists()

    # Notes: basic metadata
    assert "notes" in rr
    notes = rr["notes"]
    assert notes["report_day"] == "2025-07-20"
    assert Path(notes["csv_path"]).exists()
    assert "evidence_tables" in notes and isinstance(notes["evidence_tables"], list)
    assert len(notes["evidence_tables"]) > 0
