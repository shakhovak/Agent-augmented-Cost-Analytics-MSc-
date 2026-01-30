"""Orchestrates daily report generation: query -> evidence pack -> summary -> artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

import pandas as pd
import yaml

from src.analytics.evidence_pack import build_standard_daily_evidence, EvidencePack
from src.core.constants import SafetyLimits, TimeWindowType
from src.core.errors import ConfigError, ValidationError
from src.core.types import (
    RunArtifacts,
    RunRecord,
    TimeWindow,
)
from src.core.utils_dates import parse_date, normalize_time_window
from src.data.csv_backend import CsvBackend
from src.reports.summary_generator import generate_daily_summary
from src.viz.dashboard_builder import build_standard_daily_dashboard


def _load_yaml(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _export_evidence_pack(evidence: EvidencePack, out_dir: str) -> None:
    _ensure_dir(out_dir)
    for name, df in evidence.items():
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        df.to_csv(Path(out_dir) / f"{name}.csv", index=False)


def run_standard_daily_report(
    csv_path: str,
    report_day: date,
    output_root: str = "outputs/runs",
    dataset_name: str = "joint_costs_daily",
    report_templates_path: str = "configs/report_templates.yaml",
    template_id: str = "standard_daily_report",
    limits: Optional[SafetyLimits] = None,
) -> RunArtifacts:
    """
    Orchestrator for the "Standard Daily Report" button.

    - Loads template config
    - Loads CSV backend (cached DataFrame)
    - Builds evidence pack (deterministic)
    - Builds dashboard PNG
    - Builds deterministic summary text
    - Writes run_record.json and exports evidence tables
    """
    started_at = datetime.utcnow()
    run_id = uuid4().hex[:12]

    output_dir = _ensure_dir(str(Path(output_root) / run_id))
    evidence_dir = _ensure_dir(str(Path(output_dir) / "evidence"))

    # Load template parameters (top_n, trend_days, etc.)
    cfg = _load_yaml(report_templates_path)
    templates = cfg.get("templates") or {}
    tpl = templates.get(template_id)
    if not tpl:
        raise ConfigError(
            f"Template '{template_id}' not found in {report_templates_path}"
        )

    constraints = tpl.get("constraints") or {}
    top_n = int(constraints.get("top_n", 10))
    max_days = int(constraints.get("max_days", 8))

    # Instantiate backend (CSV-only Stage 1)
    backend = CsvBackend(
        csv_path=csv_path,
        dataset_name=dataset_name,
        limits=limits or SafetyLimits(),
    )

    # Build evidence pack from full DF (deterministic; uses date filtering internally)
    # Note: we do not perform "query planning" here yet; templates keep it simple.
    evidence = build_standard_daily_evidence(
        df_all=backend.df,
        report_day=report_day,
        trend_days=max(
            2, min(7, max_days - 1) + 1
        ),  # keep typical 7d, bounded by max_days
        top_n=top_n,
    )

    # Build dashboard
    dashboard_png = str(Path(output_dir) / "dashboard.png")
    build_standard_daily_dashboard(
        evidence=evidence,
        out_path=dashboard_png,
        title=f"{tpl.get('title', 'Standard Daily Report')} — {report_day.isoformat()}",
    )

    # Summary text
    summary_txt_path = str(Path(output_dir) / "summary.txt")
    summary_text = generate_daily_summary(evidence=evidence, top_n=min(5, top_n))
    Path(summary_txt_path).write_text(summary_text, encoding="utf-8")

    # Export evidence tables
    _export_evidence_pack(evidence, evidence_dir)

    # Build run record (minimal, but solid for research)
    artifacts = RunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        dashboard_png=dashboard_png,
        summary_txt=summary_txt_path,
        evidence_dir=evidence_dir,
        run_record_json=str(Path(output_dir) / "run_record.json"),
    )

    finished_at = datetime.utcnow()

    # Lineage is produced by backend.query(), but we didn’t call it here (we used backend.df).
    # For Stage 1, record dataset hash and high-level metadata anyway.
    record = RunRecord(
        run_id=run_id,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        mode="button",
        template_id=template_id,
        user_input=None,
        query_specs=[],
        lineage=[],
        artifacts=artifacts,
        verifier={"status": "not_enabled_stage_1"},
        notes={
            "dataset_name": dataset_name,
            "dataset_version": backend.dataset_version,
            "csv_path": os.path.abspath(csv_path),
            "report_day": report_day.isoformat(),
            "template": {
                "title": tpl.get("title"),
                "description": tpl.get("description"),
                "constraints": constraints,
            },
            "evidence_tables": list(evidence.keys()),
        },
    )

    Path(artifacts.run_record_json).write_text(
        json.dumps(_run_record_to_json(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return artifacts


def _run_record_to_json(rr: RunRecord) -> Dict:
    """
    Convert RunRecord (dataclasses) to JSON-serialisable dict.
    """

    def _serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat() + "Z"
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    d = asdict(rr)
    # dataclasses.asdict won't handle datetime/date; fix:
    d["started_at_utc"] = _serialize(rr.started_at_utc)
    d["finished_at_utc"] = (
        _serialize(rr.finished_at_utc) if rr.finished_at_utc else None
    )

    # artifacts
    d["artifacts"]["run_id"] = rr.artifacts.run_id
    # Ensure any stray datetime/date inside notes are serialised
    d["notes"] = json.loads(json.dumps(d["notes"], default=_serialize))
    return d
