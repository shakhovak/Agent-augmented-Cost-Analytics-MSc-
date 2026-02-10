"""Orchestrates daily report generation: query -> evidence pack -> summary -> artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from cost_agent_mvp.agent.analyst import Analyst
from cost_agent_mvp.agent.verifier import verify_numeric_fidelity
from cost_agent_mvp.analytics.evidence_pack import (
    EvidencePack,
    build_standard_daily_evidence,
)
from cost_agent_mvp.core.constants import SafetyLimits
from cost_agent_mvp.core.errors import ConfigError
from cost_agent_mvp.core.types import RunArtifacts, RunRecord
from cost_agent_mvp.data.csv_backend import CsvBackend
from cost_agent_mvp.reports.summary_generator import generate_daily_summary
from cost_agent_mvp.viz.dashboard_builder import build_standard_daily_dashboard


def _load_yaml(path: str) -> dict:
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


def _to_jsonable(obj):
    """Best-effort conversion to JSON-serialisable primitives (deterministic)."""
    from dataclasses import is_dataclass

    if obj is None or isinstance(obj, str | int | float | bool):
        return obj

    # numpy / pandas scalar types
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list | tuple | set):
        return [_to_jsonable(v) for v in obj]

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass

    # dataclasses
    if is_dataclass(obj):
        try:
            return _to_jsonable(asdict(obj))
        except Exception:
            pass

    # last resort: object __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass

    return str(obj)


def run_standard_daily_report(
    csv_path: str,
    report_day: date,
    output_root: str = "outputs/runs",
    dataset_name: str = "joint_costs_daily",
    report_templates_path: str = "configs/report_templates.yaml",
    template_id: str = "standard_daily_report",
    limits: SafetyLimits | None = None,
    top_n_override: int | None = None,
    trend_days_override: int | None = None,
    narrative_mode: str = "deterministic",
    mock_llm: bool = False,
    llm_temperature: float = 0.2,
    llm_max_output_tokens: int = 3000,
    prompts_dir: str | None = None,
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
        raise ConfigError(f"Template '{template_id}' not found in {report_templates_path}")

    constraints = tpl.get("constraints") or {}
    top_n_cfg = int(constraints.get("top_n", 10))
    max_days = int(constraints.get("max_days", 8))

    top_n = int(top_n_override) if top_n_override is not None else top_n_cfg

    trend_days_default = max(2, min(7, max_days - 1) + 1)
    trend_days = int(trend_days_override) if trend_days_override is not None else trend_days_default

    # Instantiate backend (CSV-only Stage 1)
    backend = CsvBackend(
        csv_path=csv_path,
        dataset_name=dataset_name,
        limits=limits or SafetyLimits(),
    )
    available_days = set(pd.to_datetime(backend.df["date"]).dt.date.unique())
    if report_day not in available_days:
        raise ValueError(
            f"report_day={report_day.isoformat()} is not present in dataset. "
            f"Available range: {min(available_days).isoformat()}..{max(available_days).isoformat()}"
        )

    # Build evidence pack from full DF (deterministic; uses date filtering internally)
    # Note: we do not perform "query planning" here yet; templates keep it simple.
    evidence = build_standard_daily_evidence(
        df_all=backend.df,
        report_day=report_day,
        trend_days=trend_days,
        top_accounts_n=top_n,
    )

    # Export evidence tables early so they exist even if later steps fail (e.g., LLM).
    _export_evidence_pack(evidence, evidence_dir)

    # Build dashboard
    dashboard_png = str(Path(output_dir) / "dashboard.png")
    build_standard_daily_dashboard(
        evidence=evidence,
        out_path=dashboard_png,
        title=f"{tpl.get('title', 'Standard Daily Report')} — {report_day.isoformat()}",
    )

    # Summary text (deterministic or LLM)
    summary_txt_path = str(Path(output_dir) / "summary.txt")

    verifier_payload: dict | None = None
    llm_meta: dict | None = None

    if narrative_mode not in ("deterministic", "llm"):
        raise ValueError("narrative_mode must be 'deterministic' or 'llm'")

    if narrative_mode == "deterministic":
        summary_text = generate_daily_summary(evidence=evidence, top_n=min(5, top_n))
        Path(summary_txt_path).write_text(summary_text, encoding="utf-8")
        verifier_payload = {"status": "not_enabled_deterministic"}
    else:
        analyst = Analyst(prompts_dir=prompts_dir)
        res = analyst.generate(
            evidence=evidence,
            mode="daily_report",
            user_text="",
            temperature=llm_temperature,
            max_output_tokens=llm_max_output_tokens,
            mock_mode=mock_llm,
        )
        summary_text = res.answer_text
        Path(summary_txt_path).write_text(summary_text, encoding="utf-8")

        # Deterministic verifier (numeric fidelity + direction checks)
        vres = verify_numeric_fidelity(answer_text=summary_text, evidence=evidence)
        verifier_payload = {
            "status": getattr(vres, "status", "unknown"),
            "passed": getattr(vres, "passed", False),
            "score": getattr(vres, "score", None),
            "issues": getattr(vres, "issues", []) or [],
        }
        verifier_json_path = str(Path(output_dir) / "verifier.json")
        verifier_payload = _to_jsonable(verifier_payload)
        Path(verifier_json_path).write_text(
            json.dumps(verifier_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        llm_meta = {
            "model": res.model,
            "prompt_chars": res.prompt_chars,
            "mock_llm": mock_llm,
            "temperature": llm_temperature,
            "max_output_tokens": llm_max_output_tokens,
        }

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
    evidence_tables = sorted(evidence.keys())
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
        verifier=verifier_payload or {"status": "not_enabled_stage_1"},
        notes={
            "dataset_name": dataset_name,
            "dataset_version": backend.dataset_version,
            "csv_path": os.path.abspath(csv_path),
            "report_day": report_day.isoformat(),
            "narrative_mode": narrative_mode,
            "llm_meta": llm_meta,
            "verifier_json": (
                str(Path(output_dir) / "verifier.json") if narrative_mode == "llm" else None
            ),
            "template": {
                "title": tpl.get("title"),
                "description": tpl.get("description"),
                "constraints": constraints,
            },
            "constraints": {
                **constraints,
                "top_n_effective": top_n,
                "trend_days_effective": trend_days,
            },
            "evidence_tables": evidence_tables,
        },
    )

    Path(artifacts.run_record_json).write_text(
        json.dumps(_run_record_to_json(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return artifacts


def _run_record_to_json(rr: RunRecord) -> dict:
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
    d["finished_at_utc"] = _serialize(rr.finished_at_utc) if rr.finished_at_utc else None

    # artifacts
    d["artifacts"]["run_id"] = rr.artifacts.run_id
    # Ensure any stray datetime/date inside notes are serialised
    d["notes"] = json.loads(json.dumps(d["notes"], default=_serialize))
    return d
