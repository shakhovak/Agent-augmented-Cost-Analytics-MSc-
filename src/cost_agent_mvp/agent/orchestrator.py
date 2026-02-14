from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from cost_agent_mvp.agent.analyst import Analyst, AnalystResult
from cost_agent_mvp.agent.plan_compiler import compile_plan_to_queryspec
from cost_agent_mvp.agent.planner import Planner, PlannerPlan
from cost_agent_mvp.agent.verifier import (
    VerifierResult,
    verifier_result_to_json,
    verify_numeric_fidelity,
)
from cost_agent_mvp.core.types import (
    RunArtifacts,
    RunRecord,
)
from cost_agent_mvp.data.csv_backend import CsvBackend
from cost_agent_mvp.data.schema import default_joint_schema
from cost_agent_mvp.reports.daily_report import run_standard_daily_report

logger = logging.getLogger(__name__)


# -----------------------------
# Small helpers
# -----------------------------


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def python_version_str() -> str:
    return platform.python_version()


def best_effort_git_hash(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def resolve_output_root(output_root: str | None) -> str:
    return (output_root or os.getenv("COST_AGENT_OUTPUTS_DIR") or "outputs/runs").strip()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load YAML configs") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_df_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _read_text_if_exists(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _read_json_if_exists(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common objects to JSON-serializable forms.
    Avoids tight coupling to pydantic/dataclass internals.
    """
    from dataclasses import asdict, is_dataclass

    if obj is None or isinstance(obj, str | int | float | bool):
        return obj

    if isinstance(obj, date | datetime):
        return obj.isoformat()

    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}

    if hasattr(obj, "model_dump"):  # pydantic
        return _to_jsonable(obj.model_dump())

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list | tuple | set):
        return [_to_jsonable(v) for v in obj]

    return str(obj)


# -----------------------------
# Orchestrator I/O types
# -----------------------------


@dataclass(frozen=True)
class VerifierOutput:
    status: str
    passed: bool
    score: float
    issues: list[dict[str, Any]]
    payload: dict[str, Any]


@dataclass(frozen=True)
class OrchestratorOutput:
    run_id: str
    output_dir: str
    answer_text: str
    dashboard_png: str | None
    summary_txt: str | None
    verifier: VerifierOutput
    plan: PlannerPlan | None
    query_specs: list[Any]
    lineage: list[dict[str, Any]]


# -----------------------------
# Orchestrator
# -----------------------------


class Orchestrator:
    """
    Routing entrypoint:
      - button flow uses report templates (standard daily report)
      - ad-hoc flow uses planner -> compiler -> backend -> analyst -> verifier
    """

    def __init__(
        self,
        *,
        outputs_root: str | None = None,
        dataset_name: str = "joint_costs_daily",
        semantic_layer_path: str = "configs/semantic_layer.yaml",
        report_templates_path: str = "configs/report_templates.yaml",
        prompts_dir: str | None = None,
    ) -> None:
        self.outputs_root = outputs_root
        self.dataset_name = dataset_name
        self.semantic_layer_path = semantic_layer_path
        self.report_templates_path = report_templates_path
        self.prompts_dir = prompts_dir

    # -----------------------------
    # Button flow (daily report)
    # -----------------------------

    def run_button_standard_daily(
        self,
        *,
        csv_path: str,
        report_day: date,
        template_id: str = "standard_daily_report",
        trend_days: int = 8,
        top_n: int = 10,
        use_llm_analyst: bool = False,
        mock_llm: bool = False,
        build_dashboard: bool = True,
        output_root: str | None = None,
    ) -> OrchestratorOutput:
        mode = "llm" if use_llm_analyst else "deterministic"

        artifacts = run_standard_daily_report(
            csv_path=csv_path,
            report_day=report_day,
            output_root=output_root or self.outputs_root,
            report_templates_path=self.report_templates_path,
            template_id=template_id,
            trend_days=trend_days,
            top_n=top_n,
            mode=mode,
            mock_llm=mock_llm,
            prompts_dir=self.prompts_dir,
        )

        run_record = _read_json_if_exists(artifacts.run_record_json)
        verifier_payload = run_record.get("verifier") or {"status": "not_enabled"}
        status = str(
            verifier_payload.get("status") or ("PASS" if verifier_payload.get("passed") else "FAIL")
        )
        passed = bool(verifier_payload.get("passed", status.upper() == "PASS"))
        score = float(verifier_payload.get("score", 1.0 if passed else 0.0))
        issues = list(verifier_payload.get("issues") or [])

        verifier = VerifierOutput(
            status=status,
            passed=passed,
            score=score,
            issues=issues,
            payload=verifier_payload,
        )

        return OrchestratorOutput(
            run_id=artifacts.run_id,
            output_dir=str(artifacts.output_dir),
            answer_text=_read_text_if_exists(artifacts.summary_txt),
            dashboard_png=(str(artifacts.dashboard_png) if artifacts.dashboard_png else None),
            summary_txt=str(artifacts.summary_txt) if artifacts.summary_txt else None,
            verifier=verifier,
            plan=None,
            query_specs=list(run_record.get("query_specs") or []),
            lineage=list(run_record.get("lineage") or []),
        )

    # -----------------------------
    # Ad-hoc Q&A flow (M4-04)
    # -----------------------------

    def run_ad_hoc(
        self,
        *,
        csv_path: str,
        user_text: str,
        anchor_day: date | None = None,
        output_root: str | None = None,
        mock_planner: bool = False,
        mock_analyst: bool = False,
        mode: str = "llm",
        build_dashboard: bool = False,
    ) -> OrchestratorOutput:
        """
        Q&A pipeline:
          planner -> compiler -> backend.query -> mini-evidence -> analyst -> verifier
        """
        if not user_text.strip():
            raise ValueError("user_text is empty")

        semantic_layer = _load_yaml(self.semantic_layer_path)
        backend = CsvBackend(csv_path=csv_path, schema=default_joint_schema())

        anchor = anchor_day or date.today()

        # Run folder
        root = resolve_output_root(output_root or self.outputs_root)
        run_id = uuid4().hex[:12]
        run_dir = Path(root) / run_id
        _ensure_dir(run_dir)
        evidence_dir = run_dir / "evidence"
        _ensure_dir(evidence_dir)

        started_at = now_utc_iso()

        # 1) Plan
        planner = Planner(prompts_dir=self.prompts_dir)
        plan = planner.plan(
            user_text=user_text,
            semantic_layer=semantic_layer,
            mock_mode=mock_planner,
        )

        # If planner explicitly says "UNSUPPORTED", return cleanly
        if getattr(plan, "status", "OK") != "OK":
            msg = getattr(plan, "error_message", None) or "Unsupported request."
            verifier = VerifierOutput(
                status="UNSUPPORTED",
                passed=False,
                score=0.0,
                issues=[{"type": "UNSUPPORTED_QUERY", "detail": msg}],
                payload={
                    "status": "UNSUPPORTED",
                    "passed": False,
                    "issues": [{"type": "UNSUPPORTED_QUERY", "detail": msg}],
                },
            )
            return OrchestratorOutput(
                run_id=run_id,
                output_dir=str(run_dir),
                answer_text=msg,
                dashboard_png=None,
                summary_txt=None,
                verifier=verifier,
                plan=plan,
                query_specs=[],
                lineage=[],
            )

        # 2) Compile
        try:
            specs = compile_plan_to_queryspec(plan, semantic_layer, anchor_day=anchor)
        except Exception as exc:
            msg = str(exc) or "Unsupported request."
            verifier = VerifierOutput(
                status="UNSUPPORTED",
                passed=False,
                score=0.0,
                issues=[{"type": "UNSUPPORTED_QUERY", "detail": msg}],
                payload={
                    "status": "UNSUPPORTED",
                    "passed": False,
                    "score": 0.0,
                    "issues": [{"type": "UNSUPPORTED_QUERY", "detail": msg}],
                },
            )
            return OrchestratorOutput(
                run_id=run_id,
                output_dir=str(run_dir),
                answer_text=msg,
                dashboard_png=None,
                summary_txt=None,
                verifier=verifier,
                plan=plan,
                query_specs=[],
                lineage=[],
            )

        # 3) Execute
        evidence: dict[str, pd.DataFrame] = {}
        lineage: list[dict[str, Any]] = []

        base_name = (getattr(plan, "view", None) or plan.intent or "query").strip() or "query"
        for i, spec in enumerate(specs, start=1):
            res = backend.query(spec)
            table_name = base_name if i == 1 else f"{base_name}_{i}"
            evidence[table_name] = res.df
            lineage.extend(res.lineage or [])
            _write_df_csv(res.df, evidence_dir / f"{table_name}.csv")

        # 4) Analyst
        analyst = Analyst(prompts_dir=self.prompts_dir, default_mode="qa")
        analyst_res: AnalystResult = analyst.generate(
            evidence=evidence,
            mode="qa",
            user_text=user_text,
            mock_mode=mock_analyst or (mode != "llm"),
        )
        answer_text = analyst_res.answer_text.strip()

        # 5) Verifier
        verifier_res: VerifierResult = verify_numeric_fidelity(
            answer_text,
            evidence,
            require_kpi_table=False,  # Q&A may not include kpis
        )
        verifier_payload = verifier_result_to_json(verifier_res)

        verifier = VerifierOutput(
            status=str(verifier_payload.get("status", "PASS" if verifier_res.passed else "FAIL")),
            passed=verifier_res.passed,
            score=float(verifier_payload.get("score", 1.0 if verifier_res.passed else 0.0)),
            issues=list(verifier_payload.get("issues") or []),
            payload=verifier_payload,
        )

        # 6) Write artifacts
        summary_path = run_dir / "answer.txt"
        summary_path.write_text(answer_text, encoding="utf-8")

        (run_dir / "verifier.json").write_text(
            json.dumps(verifier_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "plan.json").write_text(
            json.dumps(_to_jsonable(plan), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "query_specs.json").write_text(
            json.dumps([_to_jsonable(s) for s in specs], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 7) Run record
        artifacts = RunArtifacts(
            run_id=run_id,
            output_dir=str(run_dir),
            dashboard_png=None,
            summary_txt=str(summary_path),
            evidence_dir=str(evidence_dir),
            run_record_json=str(run_dir / "run_record.json"),
        )

        finished_at = now_utc_iso()

        record = RunRecord(
            run_id=run_id,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            mode="qa",
            template_id=None,
            user_input=user_text,
            query_specs=[_to_jsonable(s) for s in specs],
            lineage=lineage,
            artifacts=artifacts,
            verifier=verifier_payload,
            notes={
                "dataset_name": self.dataset_name,
                "dataset_version": backend.dataset_version,
                "csv_path": os.path.abspath(csv_path),
                "anchor_day": anchor.isoformat(),
                "semantic_layer_path": self.semantic_layer_path,
                "plan": _to_jsonable(plan),
                "git_hash": best_effort_git_hash(),
                "python": python_version_str(),
            },
        )

        Path(artifacts.run_record_json).write_text(
            json.dumps(_to_jsonable(record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        dashboard_png = None
        if build_dashboard:
            # Not part of v0; keep a hook.
            dashboard_png = None

        return OrchestratorOutput(
            run_id=run_id,
            output_dir=str(run_dir),
            answer_text=answer_text,
            dashboard_png=dashboard_png,
            summary_txt=str(summary_path),
            verifier=verifier,
            plan=plan,
            query_specs=[_to_jsonable(s) for s in specs],
            lineage=lineage,
        )
