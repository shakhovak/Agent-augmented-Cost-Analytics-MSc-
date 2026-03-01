from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from cost_agent_mvp.agent.analyst import Analyst, AnalystResult
from cost_agent_mvp.agent.detector import (
    detect_anomaly_for_end_day,
    format_detector_brief,
)
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
def query_context_to_df(
    *,
    anchor_day: date | None,
    win_start: date | None,
    win_end: date | None,
    tw_type: str | None,
) -> pd.DataFrame:
    def parts(d: date | None, prefix: str) -> dict[str, Any]:
        if d is None:
            return {
                f"{prefix}_date": None,
                f"{prefix}_year": None,
                f"{prefix}_month": None,
                f"{prefix}_day": None,
            }
        return {
            f"{prefix}_date": d.isoformat(),
            f"{prefix}_year": d.year,
            f"{prefix}_month": d.month,
            f"{prefix}_day": d.day,
        }

    row: dict[str, Any] = {
        "anchor_day": anchor_day.isoformat() if anchor_day else None,
        "time_window_type": tw_type,
        "span_days": ((win_end - win_start).days + 1 if (win_start and win_end) else None),
    }
    row.update(parts(win_start, "start"))
    row.update(parts(win_end, "end"))

    # for single-day queries, also provide "day_*" fields explicitly
    if win_start and win_end and win_start == win_end:
        row.update(parts(win_end, "day"))
    else:
        row.update({"day_date": None, "day_year": None, "day_month": None, "day_day": None})

    return pd.DataFrame([row])


def detector_payload_to_df(det: dict[str, Any]) -> pd.DataFrame:
    """
    Convert detector payload (single-day or range meta) into a tiny, numeric-friendly table
    so verifier can accept numbers mentioned in the narrative.
    """
    if not isinstance(det, dict) or not det:
        return pd.DataFrame([{"skipped": True, "reason": "empty_detector_payload"}])

    if det.get("skipped"):
        return pd.DataFrame([{"skipped": True, "reason": det.get("reason", "")}])

    # common single-day shape: det["decision"] exists
    decision = det.get("decision") if isinstance(det.get("decision"), dict) else None
    if decision:
        row = {
            "day": decision.get("day"),
            "anomaly": bool(decision.get("anomaly", False)),
            "driver": decision.get("driver"),
            "severity": decision.get("severity"),
            "flagged_accounts": decision.get("flagged_accounts"),
            "excess_cost": decision.get("excess_cost"),
        }
        # optionally include z-scores if present (keep stable column names)
        for k in (
            "services_total_cost_z",
            "log_tasks_per_chat_z",
            "log_dialog_cost_per_chat_z",
        ):
            if k in decision:
                row[k] = decision.get(k)
        return pd.DataFrame([row])

    # range meta: try best-effort list extraction
    # (adapt keys to your actual meta shape if needed)
    days = det.get("top_days") or det.get("flagged_days") or det.get("days") or []
    if isinstance(days, list) and days and isinstance(days[0], dict):
        # list of dicts already
        df = pd.DataFrame(days)
        # keep only a compact, stable subset if present
        keep = [
            c
            for c in [
                "day",
                "driver",
                "severity",
                "flagged_accounts",
                "excess_cost",
                "z",
            ]
            if c in df.columns
        ]
        return df[keep] if keep else df

    # fallback: single row
    return pd.DataFrame([{"skipped": False, "note": "unrecognized_detector_shape"}])


def _parse_day(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.fromisoformat(s).date()


def _extract_compiled_window_from_specs(
    specs: list[Any],
) -> tuple[date | None, date | None]:
    """
    Best-effort extraction of (start, end) from compiled QuerySpec.
    We use _to_jsonable(spec) to avoid relying on internal types.
    """
    if not specs:
        return None, None

    s0 = _to_jsonable(specs[0])
    tw = (s0.get("time_window") or {}) if isinstance(s0, dict) else {}

    # common shapes: {"type":"single_day","day":"YYYY-MM-DD"} or {"type":"range","start":...,"end":...}
    day = _parse_day(tw.get("day"))
    start = _parse_day(tw.get("start"))
    end = _parse_day(tw.get("end"))

    if day and not end:
        end = day
    if end and not start:
        start = end

    return start, end


def _detector_summary_single(
    *,
    csv_path: str,
    day: date,
    baseline_days: int = 14,
    context_days: int = 7,
) -> tuple[str, dict[str, Any]]:
    det = detect_anomaly_for_end_day(
        csv_path=csv_path,
        end_day=day,
        baseline_days=baseline_days,
        context_days=context_days,
    )
    return format_detector_brief(det), det


def _detector_summary_range(
    *,
    csv_path: str,
    start: date,
    end: date,
    baseline_days: int = 14,
    top_k: int = 10,
    max_scan_days: int = 30,
) -> tuple[str, dict[str, Any]]:
    """
    Loop day-by-day, but only keep top-K flagged days (token efficient).
    Caps scanning to max_scan_days (MVP safety for long ranges).
    """
    total_days = (end - start).days + 1
    scan_days = min(total_days, max_scan_days)

    # If range is long, scan only the *tail* ending at 'end' (most relevant operationally).
    scan_start = end - timedelta(days=scan_days - 1)

    flagged: list[tuple[float, str]] = []
    d = scan_start
    while d <= end:
        det = detect_anomaly_for_end_day(
            csv_path=csv_path,
            end_day=d,
            baseline_days=baseline_days,
            context_days=7,
        )
        dec = det.get("decision") or {}
        if bool(dec.get("anomaly")):
            sev = float(dec.get("severity") or 0.0)
            why = str(dec.get("why") or "").strip()
            if why:
                flagged.append((sev, why))
        d += timedelta(days=1)

    flagged.sort(key=lambda x: x[0], reverse=True)
    top = flagged[:top_k]

    if not flagged:
        msg = (
            f"Anomaly scan (services-only, baseline=prev {baseline_days}d): "
            f"no spike days detected in scanned range {scan_start.isoformat()}..{end.isoformat()}."
        )
    else:
        lines = "\n".join([f"- {why}" for _, why in top])
        msg = (
            f"Anomaly scan (services-only, baseline=prev {baseline_days}d): "
            f"{len(flagged)} flagged day(s) in scanned range {scan_start.isoformat()}..{end.isoformat()}.\n"
            f"Top {len(top)}:\n{lines}"
        )

    meta = {
        "range_requested": {"start": start.isoformat(), "end": end.isoformat()},
        "range_scanned": {"start": scan_start.isoformat(), "end": end.isoformat()},
        "flagged_days_scanned": len(flagged),
        "top_k": top_k,
        "baseline_days": baseline_days,
        "max_scan_days": max_scan_days,
    }
    return msg, meta


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
            msg = (
                getattr(plan, "reason", None)
                or getattr(plan, "error_message", None)
                or "Unsupported request."
            )
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
            # 2.5) Detector (deterministic) — token-efficient summary for Analyst
        detector_summary = ""
        detector_payload: dict[str, Any] = {}

        # Use the *planner* time_window type to decide “single vs range”
        tw = getattr(plan, "time_window", None)
        tw_type = getattr(tw, "type", None) if tw is not None else None

        # We rely on compiled specs to get concrete dates (anchor resolved).
        win_start, win_end = _extract_compiled_window_from_specs(specs)

        limits = (semantic_layer.get("limits") or {}) if isinstance(semantic_layer, dict) else {}
        max_scan_days = int(limits.get("max_days_default", 30))

        # MVP rule: yesterday -> skip detector (as you described)
        if tw_type == "yesterday":
            detector_summary = ""
            detector_payload = {
                "skipped": True,
                "reason": "time_window=yesterday (MVP: no anomaly scan)",
            }
        else:
            # single day if start==end, otherwise range scan
            if win_start and win_end and win_start == win_end:
                detector_summary, det = _detector_summary_single(
                    csv_path=csv_path,
                    day=win_end,
                    baseline_days=14,
                    context_days=7,
                )
                detector_payload = det
            elif win_start and win_end:
                detector_summary, meta = _detector_summary_range(
                    csv_path=csv_path,
                    start=win_start,
                    end=win_end,
                    baseline_days=14,
                    top_k=10,
                    max_scan_days=max_scan_days,
                )
                detector_payload = meta
            else:
                detector_summary = ""
                detector_payload = {
                    "skipped": True,
                    "reason": "Could not infer window bounds from compiled specs.",
                }

        # Persist detector payload for debugging / evaluation
        (run_dir / "detector.json").write_text(
            json.dumps(_to_jsonable(detector_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 3) Execute
        evidence: dict[str, pd.DataFrame] = {}
        lineage: list[dict[str, Any]] = []
        det_df = detector_payload_to_df(detector_payload)
        evidence["detector"] = det_df
        _write_df_csv(det_df, evidence_dir / "detector.csv")
        base_name = (getattr(plan, "view", None) or plan.intent or "query").strip() or "query"
        for i, spec in enumerate(specs, start=1):
            try:
                res = backend.query(spec)
            except Exception as exc:
                (run_dir / "backend_error.txt").write_text(str(exc), encoding="utf-8")
                raise
            table_name = base_name if i == 1 else f"{base_name}_{i}"
            evidence[table_name] = res.df
            lineage.extend(res.lineage or [])
            _write_df_csv(res.df, evidence_dir / f"{table_name}.csv")

        # --- Add query context table so verifier can match date numbers (2025, 6, 20, etc.) ---
        ctx_df = query_context_to_df(
            anchor_day=anchor_day,
            win_start=win_start,
            win_end=win_end,
            tw_type=str(tw_type) if tw_type is not None else None,
        )

        evidence["query_context"] = ctx_df
        _write_df_csv(ctx_df, evidence_dir / "query_context.csv")

        # 4) Analyst
        analyst = Analyst(prompts_dir=self.prompts_dir, default_mode="qa")
        analyst_res: AnalystResult = analyst.generate(
            evidence=evidence,
            mode="qa",
            user_text=user_text,
            detector_summary=detector_summary,  # NEW
            detector_max_chars=700,  # keep tight for tokens
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
