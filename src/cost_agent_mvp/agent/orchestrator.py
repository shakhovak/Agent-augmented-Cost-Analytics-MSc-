# src/agent/orchestrator.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.agent.analyst import Analyst
from src.agent.planner import Planner, PlannerPlan
from src.agent.verifier import (
    VerifierResult,
    verifier_result_to_json,
    verify_numeric_fidelity,
)
from src.analytics.evidence_pack import EvidencePack, build_standard_daily_evidence
from src.core.errors import ConfigError, ValidationError
from src.core.types import RunArtifacts, RunRecord
from src.core.utils_dates import parse_date
from src.data.csv_backend import CsvBackend
from src.reports.summary_generator import generate_daily_summary
from src.viz.dashboard_builder import build_standard_daily_dashboard

load_dotenv()


@dataclass(frozen=True)
class OrchestratorOutput:
    run_id: str
    mode: str  # "button" | "ad_hoc"
    output_dir: str
    evidence: EvidencePack
    dashboard_png: Optional[str]
    answer_text: str
    verifier: VerifierResult
    run_record_json: str


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _write_text(path: str, text: str) -> None:
    Path(path).write_text(text or "", encoding="utf-8")


def _export_evidence_pack(evidence: EvidencePack, out_dir: str) -> None:
    _ensure_dir(out_dir)
    for name, df in evidence.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        df.to_csv(Path(out_dir) / f"{name}.csv", index=False)


def _serialize_dt(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat() + "Z"
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _run_record_to_json(rr: RunRecord) -> Dict[str, Any]:
    d = asdict(rr)
    d["started_at_utc"] = _serialize_dt(rr.started_at_utc)
    d["finished_at_utc"] = (
        _serialize_dt(rr.finished_at_utc) if rr.finished_at_utc else None
    )
    # ensure notes are json-serializable
    d["notes"] = json.loads(json.dumps(d.get("notes", {}), default=_serialize_dt))
    return d


def _apply_filters_df(df: pd.DataFrame, plan: PlannerPlan) -> pd.DataFrame:
    """
    Apply a conservative subset of filters directly to the raw dataframe.
    This is used only for the ad-hoc MVP to scope evidence pack generation.

    NOTE: We intentionally do NOT implement arbitrary grouping here.
    Evidence pack generation remains deterministic and template-based.
    """
    out = df
    f = plan.filters

    if f.account_id:
        out = out[out["account_id"].isin(f.account_id)]

    if f.chat_type:
        out = out[out["chat_type"].isin(f.chat_type)]

    if f.chat_id:
        # Drilldown safety is already enforced in planner prompts, but keep defense-in-depth
        if not f.account_id:
            raise ValidationError(
                "chat_id filter requires account_id filter (drilldown safety)."
            )
        out = out[out["chat_id"].isin(f.chat_id)]

    if f.has_tasks is not None:
        out = out[out["has_tasks"].fillna(0).astype(int) == (1 if f.has_tasks else 0)]

    if f.has_classifications is not None:
        out = out[
            out["has_classifications"].fillna(0).astype(int)
            == (1 if f.has_classifications else 0)
        ]

    if f.has_both is not None:
        out = out[out["has_both"].fillna(0).astype(int) == (1 if f.has_both else 0)]

    return out


def _resolve_report_day_from_plan(plan: PlannerPlan) -> date:
    """
    For MVP, evidence pack builder is day-based. We map plan.time_window onto a report_day.
    - yesterday -> yesterday
    - last_n_days / range -> use end date as report_day
    """
    tw = plan.time_window
    today = datetime.utcnow().date()

    if tw.type == "yesterday":
        return today - timedelta(days=1)

    if tw.type == "last_n_days":
        # end = yesterday by default; interpret as ending yesterday for operational use
        end = today - timedelta(days=1)
        return end

    if tw.type == "range":
        if tw.end:
            return parse_date(tw.end)
        # if no end provided, fallback to yesterday
        return today - timedelta(days=1)

    # fallback
    return today - timedelta(days=1)


class Orchestrator:
    """
    Plain-Python orchestrator (no LangGraph).

    Supports:
      - button mode: standard daily report
      - ad-hoc mode: planner -> (template-based) evidence -> optional dashboard -> analyst -> verifier

    NOTE: In this MVP, ad-hoc execution is still template-based; the planner selects filters/time window,
    but evidence generation uses the standard daily evidence pack builder.
    """

    def __init__(
        self,
        *,
        prompts_dir: str = "src/agent/prompts",
        semantic_layer_path: str = "configs/semantic_layer.yaml",
        report_templates_path: str = "configs/report_templates.yaml",
        outputs_root: str = "outputs/runs",
        dataset_name: str = "joint_costs_daily",
    ) -> None:
        self.prompts_dir = prompts_dir
        self.semantic_layer_path = semantic_layer_path
        self.report_templates_path = report_templates_path
        self.outputs_root = outputs_root
        self.dataset_name = dataset_name

        self.planner = Planner(prompts_dir=prompts_dir)
        self.analyst = Analyst(prompts_dir=prompts_dir)

    def run_button_standard_daily(
        self,
        *,
        csv_path: str,
        report_day: date,
        use_llm_analyst: bool = True,
        build_dashboard: bool = True,
        top_n: int = 10,
        trend_days: int = 7,
    ) -> OrchestratorOutput:
        run_id = uuid4().hex[:12]
        started_at = datetime.utcnow()

        output_dir = _ensure_dir(str(Path(self.outputs_root) / run_id))
        evidence_dir = _ensure_dir(str(Path(output_dir) / "evidence"))

        backend = CsvBackend(csv_path=csv_path, dataset_name=self.dataset_name)
        evidence = build_standard_daily_evidence(
            df_all=backend.df,
            report_day=report_day,
            trend_days=trend_days,
            top_n=top_n,
        )

        dashboard_png = None
        if build_dashboard:
            dashboard_png = str(Path(output_dir) / "dashboard.png")
            build_standard_daily_dashboard(
                evidence=evidence,
                out_path=dashboard_png,
                title=f"Standard Daily Report — {report_day.isoformat()}",
            )

        # Narrative
        if use_llm_analyst:
            analyst_res = self.analyst.generate(
                evidence=evidence,
                mode="button",
                user_text="Standard daily report.",
            )
            answer_text = analyst_res.answer_text
        else:
            answer_text = generate_daily_summary(evidence=evidence)

        # Verify
        verifier_res = verify_numeric_fidelity(
            answer_text=answer_text, evidence=evidence
        )

        # Persist artifacts
        _export_evidence_pack(evidence, evidence_dir)
        summary_txt = str(Path(output_dir) / "summary.txt")
        _write_text(summary_txt, answer_text)

        finished_at = datetime.utcnow()

        artifacts = RunArtifacts(
            run_id=run_id,
            output_dir=output_dir,
            dashboard_png=dashboard_png,
            summary_txt=summary_txt,
            evidence_dir=evidence_dir,
            run_record_json=str(Path(output_dir) / "run_record.json"),
        )

        rr = RunRecord(
            run_id=run_id,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            mode="button",
            template_id="standard_daily_report",
            user_input=None,
            query_specs=[],
            lineage=[],
            artifacts=artifacts,
            verifier=verifier_result_to_json(verifier_res),
            notes={
                "dataset_name": self.dataset_name,
                "dataset_version": backend.dataset_version,
                "csv_path": os.path.abspath(csv_path),
                "report_day": report_day.isoformat(),
                "use_llm_analyst": use_llm_analyst,
                "evidence_tables": list(evidence.keys()),
            },
        )

        Path(artifacts.run_record_json).write_text(
            json.dumps(_run_record_to_json(rr), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return OrchestratorOutput(
            run_id=run_id,
            mode="button",
            output_dir=output_dir,
            evidence=evidence,
            dashboard_png=dashboard_png,
            answer_text=answer_text,
            verifier=verifier_res,
            run_record_json=artifacts.run_record_json,
        )

    def run_ad_hoc(
        self,
        *,
        csv_path: str,
        user_text: str,
        use_llm_analyst: bool = True,
        build_dashboard: bool = True,
    ) -> OrchestratorOutput:
        """
        Ad-hoc MVP:
          - Planner produces a plan (filters + time window + template suggestion).
          - We execute a template-based evidence pack (standard daily) scoped by plan filters.
          - Analyst generates answer and verifier checks fidelity.

        This keeps execution deterministic and safe while enabling "ask questions" UX.
        """
        run_id = uuid4().hex[:12]
        started_at = datetime.utcnow()

        output_dir = _ensure_dir(str(Path(self.outputs_root) / run_id))
        evidence_dir = _ensure_dir(str(Path(output_dir) / "evidence"))

        semantic_yaml = (
            Path(self.semantic_layer_path).read_text(encoding="utf-8")
            if Path(self.semantic_layer_path).exists()
            else ""
        )
        templates_yaml = (
            Path(self.report_templates_path).read_text(encoding="utf-8")
            if Path(self.report_templates_path).exists()
            else ""
        )

        plan = self.planner.plan(
            user_text=user_text,
            semantic_layer_yaml=semantic_yaml,
            report_templates_yaml=templates_yaml,
        )

        backend = CsvBackend(csv_path=csv_path, dataset_name=self.dataset_name)

        if plan.status != "OK":
            # Persist minimal run with UNSUPPORTED response
            answer_text = (
                f"Unsupported request.\n"
                f"Reason: {plan.reason or 'not provided'}\n"
                f"Suggested template: {plan.suggested_template or 'standard_daily_report'}"
            )
            evidence: EvidencePack = {}
            verifier_res = verify_numeric_fidelity(
                answer_text=answer_text,
                evidence={"kpi_today_vs_yesterday": pd.DataFrame()},
            )  # will fail; ok

            summary_txt = str(Path(output_dir) / "summary.txt")
            _write_text(summary_txt, answer_text)

            artifacts = RunArtifacts(
                run_id=run_id,
                output_dir=output_dir,
                dashboard_png=None,
                summary_txt=summary_txt,
                evidence_dir=evidence_dir,
                run_record_json=str(Path(output_dir) / "run_record.json"),
            )
            rr = RunRecord(
                run_id=run_id,
                started_at_utc=started_at,
                finished_at_utc=datetime.utcnow(),
                mode="ad_hoc",
                template_id=None,
                user_input=user_text,
                query_specs=[],
                lineage=[],
                artifacts=artifacts,
                verifier={
                    "status": "SKIPPED_UNSUPPORTED",
                    "issues": [],
                    "summary": "Planner returned UNSUPPORTED.",
                },
                notes={
                    "dataset_name": self.dataset_name,
                    "dataset_version": backend.dataset_version,
                    "csv_path": os.path.abspath(csv_path),
                    "plan": plan.model_dump(),
                },
            )
            Path(artifacts.run_record_json).write_text(
                json.dumps(_run_record_to_json(rr), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return OrchestratorOutput(
                run_id=run_id,
                mode="ad_hoc",
                output_dir=output_dir,
                evidence=evidence,
                dashboard_png=None,
                answer_text=answer_text,
                verifier=VerifierResult(
                    status="FAIL",
                    issues=[],
                    summary="UNSUPPORTED flow; verifier skipped.",
                ),
                run_record_json=artifacts.run_record_json,
            )

        report_day = _resolve_report_day_from_plan(plan)

        # Scope dataframe by plan filters (conservative MVP)
        scoped_df = _apply_filters_df(backend.df, plan)

        evidence = build_standard_daily_evidence(
            df_all=scoped_df,
            report_day=report_day,
            trend_days=7,
            top_n=10,
        )

        dashboard_png = None
        if build_dashboard:
            dashboard_png = str(Path(output_dir) / "dashboard.png")
            build_standard_daily_dashboard(
                evidence=evidence,
                out_path=dashboard_png,
                title=f"Ad-hoc Analysis — {report_day.isoformat()}",
            )

        if use_llm_analyst:
            analyst_res = self.analyst.generate(
                evidence=evidence,
                mode="ad_hoc",
                user_text=user_text,
            )
            answer_text = analyst_res.answer_text
        else:
            answer_text = generate_daily_summary(evidence=evidence)

        verifier_res = verify_numeric_fidelity(
            answer_text=answer_text, evidence=evidence
        )

        _export_evidence_pack(evidence, evidence_dir)
        summary_txt = str(Path(output_dir) / "answer.txt")
        _write_text(summary_txt, answer_text)

        finished_at = datetime.utcnow()

        artifacts = RunArtifacts(
            run_id=run_id,
            output_dir=output_dir,
            dashboard_png=dashboard_png,
            summary_txt=summary_txt,
            evidence_dir=evidence_dir,
            run_record_json=str(Path(output_dir) / "run_record.json"),
        )
        rr = RunRecord(
            run_id=run_id,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            mode="ad_hoc",
            template_id=plan.template_id,
            user_input=user_text,
            query_specs=[],
            lineage=[],
            artifacts=artifacts,
            verifier=verifier_result_to_json(verifier_res),
            notes={
                "dataset_name": self.dataset_name,
                "dataset_version": backend.dataset_version,
                "csv_path": os.path.abspath(csv_path),
                "plan": plan.model_dump(),
                "report_day": report_day.isoformat(),
                "use_llm_analyst": use_llm_analyst,
                "evidence_tables": list(evidence.keys()),
            },
        )
        Path(artifacts.run_record_json).write_text(
            json.dumps(_run_record_to_json(rr), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return OrchestratorOutput(
            run_id=run_id,
            mode="ad_hoc",
            output_dir=output_dir,
            evidence=evidence,
            dashboard_png=dashboard_png,
            answer_text=answer_text,
            verifier=verifier_res,
            run_record_json=artifacts.run_record_json,
        )
