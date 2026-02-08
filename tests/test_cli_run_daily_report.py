from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_tiny_csv(csv_path: Path) -> None:
    # Must satisfy your schema (same fixture that worked for orchestrator smoke test)
    rows = [
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
            "date": "2025-07-20",
            "account_id": 1,
            "chat_id": 1002,
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
            "chat_id": 2001,
            "chat_type": "telegram",
            "cost_dialog": 0.00,
            "cost_task": 0.00,
            "cost_classification": 3.00,
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


def _write_templates_yaml(path: Path) -> None:
    path.write_text(
        """
version: 1
templates:
  standard_daily_report:
    title: "Standard Daily Report"
    description: "Test template"
    constraints:
      top_n: 5
      max_days: 8
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_cli_run_daily_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    _write_tiny_csv(csv_path)

    templates_path = tmp_path / "report_templates.yaml"
    _write_templates_yaml(templates_path)

    out_root = tmp_path / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "cost_agent_mvp.cli.run_daily_report",
        "--csv",
        str(csv_path),
        "--day",
        "2025-07-20",
        "--out",
        str(out_root),
        "--templates-path",
        str(templates_path),
        "--top-n",
        "3",
        "--trend-days",
        "2",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # parse run_id and output_dir from stdout
    stdout = proc.stdout
    assert "run_id:" in stdout
    assert "output_dir:" in stdout

    # Find output_dir line
    out_dir_line = [ln for ln in stdout.splitlines() if ln.startswith("output_dir:")][0]
    run_dir = Path(out_dir_line.split("output_dir:", 1)[1].strip())

    # Required artifacts exist
    assert (run_dir / "run_record.json").exists()
    assert (run_dir / "dashboard.png").exists()
    assert (run_dir / "summary.txt").exists()
    assert (run_dir / "evidence").exists()
    assert len(list((run_dir / "evidence").glob("*.csv"))) > 0

    # Non-empty
    assert (run_dir / "summary.txt").stat().st_size > 0
    assert (run_dir / "dashboard.png").stat().st_size > 0
    assert (run_dir / "run_record.json").stat().st_size > 0
