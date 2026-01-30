# src/agent/orchestrator_langgraph.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from src.agent.analyst import Analyst
from src.agent.planner import Planner, PlannerPlan
from src.agent.verifier import (
    VerifierResult,
    verifier_result_to_json,
    verify_numeric_fidelity,
)
from src.analytics.evidence_pack import EvidencePack, build_standard_daily_evidence
from src.core.errors import ValidationError
from src.core.types import RunArtifacts, RunRecord
from src.data.csv_backend import CsvBackend
from src.reports.summary_generator import generate_daily_summary
from src.viz.dashboard_builder import build_standard_daily_dashboard

load_dotenv()


@dataclass(frozen=True)
class GraphOutput:
    run_id: str
    output_dir: str
    answer_text: str
    verifier: VerifierResult
    dashboard_png: str | None
    run_record_json: str


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


def _run_record_to_json(rr: RunRecord) -> dict[str, Any]:
    from dataclasses import asdict

    d = asdict(rr)
    d["started_at_utc"] = _serialize_dt(rr.started_at_utc)
    d["finished_at_utc"] = _serialize_dt(rr.finished_at_utc) if rr.finished_at_utc else None
    d["notes"] = json.loads(json.dumps(d.get("notes", {}), default=_serialize_dt))
    return d


def _resolve_report_day_from_plan(plan: PlannerPlan) -> date:
    from datetime import timedelta

    today = datetime.utcnow().date()

    tw = plan.time_window
    if tw.type == "yesterday":
        return today - timedelta(days=1)
    if tw.type == "last_n_days":
        return today - timedelta(days=1)
    if tw.type == "range":
        if tw.end:
            from src.core.utils_dates import parse_date

            return parse_date(tw.end)
        return today - timedelta(days=1)
    return today - timedelta(days=1)


def _apply_filters_df(df: pd.DataFrame, plan: PlannerPlan) -> pd.DataFrame:
    out = df
    f = plan.filters

    if f.account_id:
        out = out[out["account_id"].isin(f.account_id)]
    if f.chat_type:
        out = out[out["chat_type"].isin(f.chat_type)]
    if f.chat_id:
        if not f.account_id:
            raise ValidationError("chat_id filter requires account_id filter (drilldown safety).")
        out = out[out["chat_id"].isin(f.chat_id)]
    if f.has_tasks is not None:
        out = out[out["has_tasks"].fillna(0).astype(int) == (1 if f.has_tasks else 0)]
    if f.has_classifications is not None:
        out = out[
            out["has_classifications"].fillna(0).astype(int) == (1 if f.has_classifications else 0)
        ]
    if f.has_both is not None:
        out = out[out["has_both"].fillna(0).astype(int) == (1 if f.has_both else 0)]

    return out


class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator that composes existing modules into a DAG/state machine.

    Notes:
      - This graph keeps execution template-based (standard daily evidence pack), just like the plain orchestrator.
      - It adds structure for branching, logging, and later extension (custom query execution, retries, etc.).
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

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LangGraph is not installed. Install it (e.g., `pip install langgraph`) "
                "or use src/agent/orchestrator.py instead."
            ) from e

        # State is a plain dict; keep keys explicit and simple.
        # Keys we use:
        # run_id, started_at, mode, csv_path, user_text, report_day, backend, plan, evidence,
        # answer_text, verifier, dashboard_png, output_dir, evidence_dir, run_record_json
        sg = StateGraph(dict)

        def node_init(state: dict[str, Any]) -> dict[str, Any]:
            run_id = uuid4().hex[:12]
            output_dir = _ensure_dir(str(Path(self.outputs_root) / run_id))
            evidence_dir = _ensure_dir(str(Path(output_dir) / "evidence"))
            state.update(
                {
                    "run_id": run_id,
                    "started_at": datetime.utcnow(),
                    "output_dir": output_dir,
                    "evidence_dir": evidence_dir,
                }
            )
            return state

        def node_load_backend(state: dict[str, Any]) -> dict[str, Any]:
            backend = CsvBackend(csv_path=state["csv_path"], dataset_name=self.dataset_name)
            state["backend"] = backend
            state["dataset_version"] = backend.dataset_version
            return state

        def node_plan(state: dict[str, Any]) -> dict[str, Any]:
            if state.get("mode") != "ad_hoc":
                return state

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
                user_text=state.get("user_text", ""),
                semantic_layer_yaml=semantic_yaml,
                report_templates_yaml=templates_yaml,
            )
            state["plan"] = plan
            return state

        def node_build_evidence(state: dict[str, Any]) -> dict[str, Any]:
            backend: CsvBackend = state["backend"]
            mode = state.get("mode")

            if mode == "button":
                report_day: date = state["report_day"]
                df_scope = backend.df
            else:
                plan: PlannerPlan = state["plan"]
                if plan.status != "OK":
                    # no evidence
                    state["evidence"] = {}
                    return state
                report_day = _resolve_report_day_from_plan(plan)
                state["report_day"] = report_day
                df_scope = _apply_filters_df(backend.df, plan)

            evidence = build_standard_daily_evidence(
                df_all=df_scope,
                report_day=state["report_day"],
                trend_days=7,
                top_n=10,
            )
            state["evidence"] = evidence
            return state

        def node_dashboard(state: dict[str, Any]) -> dict[str, Any]:
            if not state.get("build_dashboard", True):
                state["dashboard_png"] = None
                return state

            evidence: EvidencePack = state.get("evidence", {})
            if not evidence:
                state["dashboard_png"] = None
                return state

            out = str(Path(state["output_dir"]) / "dashboard.png")
            build_standard_daily_dashboard(
                evidence=evidence,
                out_path=out,
                title=f"{'Standard Daily Report' if state.get('mode') == 'button' else 'Ad-hoc Analysis'} — {state.get('report_day')}",
            )
            state["dashboard_png"] = out
            return state

        def node_answer(state: dict[str, Any]) -> dict[str, Any]:
            mode = state.get("mode")
            use_llm = bool(state.get("use_llm_analyst", True))
            evidence: EvidencePack = state.get("evidence", {})

            # UNSUPPORTED plan
            if mode == "ad_hoc":
                plan: PlannerPlan = state.get("plan")
                if plan and plan.status != "OK":
                    answer_text = (
                        f"Unsupported request.\n"
                        f"Reason: {plan.reason or 'not provided'}\n"
                        f"Suggested template: {plan.suggested_template or 'standard_daily_report'}"
                    )
                    state["answer_text"] = answer_text
                    return state

            if use_llm and evidence:
                analyst_res = self.analyst.generate(
                    evidence=evidence,
                    mode="button" if mode == "button" else "ad_hoc",
                    user_text=(
                        state.get("user_text", "") if mode == "ad_hoc" else "Standard daily report."
                    ),
                )
                state["answer_text"] = analyst_res.answer_text
            else:
                # deterministic fallback
                if evidence:
                    state["answer_text"] = generate_daily_summary(evidence=evidence)
                else:
                    state["answer_text"] = "No evidence available for this request."

            return state

        def node_verify(state: dict[str, Any]) -> dict[str, Any]:
            evidence: EvidencePack = state.get("evidence", {})
            answer_text: str = state.get("answer_text", "")

            if not evidence:
                # verifier expects KPI by default; mark as FAIL but keep structured output
                state["verifier"] = VerifierResult(
                    status="FAIL", issues=[], summary="No evidence to verify."
                )
                return state

            state["verifier"] = verify_numeric_fidelity(answer_text=answer_text, evidence=evidence)
            return state

        def node_persist(state: dict[str, Any]) -> dict[str, Any]:
            output_dir = state["output_dir"]
            evidence_dir = state["evidence_dir"]
            evidence: EvidencePack = state.get("evidence", {})

            if evidence:
                _export_evidence_pack(evidence, evidence_dir)

            # Write answer text
            answer_path = str(
                Path(output_dir)
                / ("summary.txt" if state.get("mode") == "button" else "answer.txt")
            )
            _write_text(answer_path, state.get("answer_text", ""))

            # Run record
            run_record_json = str(Path(output_dir) / "run_record.json")
            state["run_record_json"] = run_record_json

            artifacts = RunArtifacts(
                run_id=state["run_id"],
                output_dir=output_dir,
                dashboard_png=state.get("dashboard_png"),
                summary_txt=answer_path,
                evidence_dir=evidence_dir,
                run_record_json=run_record_json,
            )

            rr = RunRecord(
                run_id=state["run_id"],
                started_at_utc=state["started_at"],
                finished_at_utc=datetime.utcnow(),
                mode=state.get("mode"),
                template_id=(
                    "standard_daily_report"
                    if state.get("mode") == "button"
                    else getattr(state.get("plan"), "template_id", None)
                ),
                user_input=(state.get("user_text") if state.get("mode") == "ad_hoc" else None),
                query_specs=[],
                lineage=[],
                artifacts=artifacts,
                verifier=(
                    verifier_result_to_json(state.get("verifier"))
                    if state.get("verifier") is not None
                    else {
                        "status": "SKIPPED",
                        "issues": [],
                        "summary": "Verifier not run.",
                    }
                ),
                notes={
                    "dataset_name": self.dataset_name,
                    "dataset_version": state.get("dataset_version"),
                    "csv_path": os.path.abspath(state["csv_path"]),
                    "report_day": (
                        state.get("report_day").isoformat() if state.get("report_day") else None
                    ),
                    "use_llm_analyst": bool(state.get("use_llm_analyst", True)),
                    "plan": (
                        state.get("plan").model_dump() if state.get("plan") is not None else None
                    ),
                    "evidence_tables": list(evidence.keys()) if evidence else [],
                },
            )

            Path(run_record_json).write_text(
                json.dumps(_run_record_to_json(rr), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return state

        # Register nodes
        sg.add_node("init", node_init)
        sg.add_node("load_backend", node_load_backend)
        sg.add_node("plan", node_plan)
        sg.add_node("build_evidence", node_build_evidence)
        sg.add_node("dashboard", node_dashboard)
        sg.add_node("answer", node_answer)
        sg.add_node("verify", node_verify)
        sg.add_node("persist", node_persist)

        # Graph edges
        sg.set_entry_point("init")
        sg.add_edge("init", "load_backend")
        sg.add_edge("load_backend", "plan")
        sg.add_edge("plan", "build_evidence")
        sg.add_edge("build_evidence", "dashboard")
        sg.add_edge("dashboard", "answer")
        sg.add_edge("answer", "verify")
        sg.add_edge("verify", "persist")
        sg.add_edge("persist", END)

        return sg.compile()

    def run_button_standard_daily(
        self,
        *,
        csv_path: str,
        report_day: date,
        use_llm_analyst: bool = True,
        build_dashboard: bool = True,
    ) -> GraphOutput:
        graph = self._build_graph()
        state = {
            "mode": "button",
            "csv_path": csv_path,
            "report_day": report_day,
            "use_llm_analyst": use_llm_analyst,
            "build_dashboard": build_dashboard,
        }
        out_state = graph.invoke(state)

        return GraphOutput(
            run_id=out_state["run_id"],
            output_dir=out_state["output_dir"],
            answer_text=out_state.get("answer_text", ""),
            verifier=out_state.get("verifier"),
            dashboard_png=out_state.get("dashboard_png"),
            run_record_json=out_state.get("run_record_json"),
        )

    def run_ad_hoc(
        self,
        *,
        csv_path: str,
        user_text: str,
        use_llm_analyst: bool = True,
        build_dashboard: bool = True,
    ) -> GraphOutput:
        graph = self._build_graph()
        state = {
            "mode": "ad_hoc",
            "csv_path": csv_path,
            "user_text": user_text,
            "use_llm_analyst": use_llm_analyst,
            "build_dashboard": build_dashboard,
        }
        out_state = graph.invoke(state)

        return GraphOutput(
            run_id=out_state["run_id"],
            output_dir=out_state["output_dir"],
            answer_text=out_state.get("answer_text", ""),
            verifier=out_state.get("verifier"),
            dashboard_png=out_state.get("dashboard_png"),
            run_record_json=out_state.get("run_record_json"),
        )
