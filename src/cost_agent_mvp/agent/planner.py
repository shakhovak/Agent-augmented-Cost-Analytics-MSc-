from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from cost_agent_mvp.agent.prompt_loader import PromptLoader

# -----------------------------
# Errors
# -----------------------------


class PlannerError(RuntimeError):
    """Base planner error."""


class PlannerValidationError(PlannerError):
    """Raised when a plan is invalid or violates the semantic layer."""


# -----------------------------
# PlannerPlan schema (structured intent)
# -----------------------------

IntentType = Literal["kpi_summary", "trend", "breakdown", "top_n", "question", "unsupported"]
TimeWindowType = Literal["yesterday", "single_day", "last_n_days", "range"]
SortDir = Literal["asc", "desc"]
PlanStatus = Literal["OK", "UNSUPPORTED"]


class TimeWindow(BaseModel):
    type: TimeWindowType = "yesterday"
    # last_n_days
    n_days: int | None = None
    # single_day
    day: str | None = None  # YYYY-MM-DD
    # range
    start: str | None = None  # YYYY-MM-DD
    end: str | None = None  # YYYY-MM-DD


class FilterSpec(BaseModel):
    """
    Filter representation aligned to semantic_layer.allowed_operations.filters entries.
    Example:
      {"name": "account_id_in", "args": {"values": [123, 456]}}
      {"name": "last_n_days", "args": {"n_days": 14}}
    """

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ChartRequest(BaseModel):
    type: str  # validated against semantic layer charts
    title: str | None = None
    x: str | None = None
    y: str | list[str] | None = None


class PlannerPlan(BaseModel):
    status: PlanStatus = "OK"
    intent: IntentType = "question"
    view: str | None = None
    time_window: TimeWindow = Field(default_factory=TimeWindow)

    # What to compute (executor compiles this into QuerySpec(s))
    metrics: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)

    # Ranking / limiting
    top_n: int | None = None
    sort_by: str | None = None
    sort_dir: SortDir | None = None

    # Optional desired visuals (executor may ignore / downgrade)
    chart_requests: list[ChartRequest] = Field(default_factory=list)

    # Diagnostics
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None

    # src/cost_agent_mvp/agent/planner.py lines ~280-350

    @staticmethod
    def _mock_plan(user_text: str, view: SemanticLayerView) -> PlannerPlan:
        t = (user_text or "").strip().lower()
        if not t:
            return PlannerPlan(
                status="UNSUPPORTED", intent="unsupported", error_message="Empty question."
            )

        # ✅ VALID: Use physical measures (NOT derived metrics)
        if "total cost" in t and ("yesterday" in t or "2025-12-30" in t):
            return PlannerPlan(
                status="OK",
                intent="question",
                time_window=TimeWindow(type="single_day", day="2025-12-30"),
                metrics=["sum(total_cost)"],  # ← Physical measure
                group_by=[],
            )

        if "top" in t and "account" in t:
            return PlannerPlan(
                status="OK",
                intent="top_n",
                time_window=TimeWindow(type="yesterday"),
                metrics=["sum(total_cost)"],  # ← Physical measure
                group_by=["account_id"],
                top_n=5,
                sort_by="sum_total_cost",
                sort_dir="desc",
            )

        if "breakdown" in t or "by chat type" in t:
            return PlannerPlan(
                status="OK",
                intent="breakdown",
                time_window=TimeWindow(type="yesterday"),
                metrics=["sum(total_cost)"],  # ← Physical measure
                group_by=["chat_type"],
            )

        if "trend" in t or "last 7 days" in t:
            return PlannerPlan(
                status="OK",
                intent="trend",
                time_window=TimeWindow(type="last_n_days", n_days=7),
                metrics=["sum(total_cost)"],  # ← Physical measure
                group_by=["date"],
            )

        # UNSUPPORTED queries
        if "revenue" in t or "country" in t:
            return PlannerPlan(
                status="UNSUPPORTED",
                intent="unsupported",
                error_message="Metric 'revenue' is not available in this dataset",
            )

        if "99 days" in t:
            return PlannerPlan(
                status="UNSUPPORTED",
                intent="unsupported",
                error_message="Requested time window exceeds maximum allowed days (14)",
            )

        return PlannerPlan(
            status="UNSUPPORTED",
            intent="unsupported",
            error_message="Query not understood in mock mode",
        )


# -----------------------------
# Semantic layer view (for validation)
# -----------------------------


@dataclass(frozen=True)
class SemanticLayerView:
    allowed_group_by: set[str]
    allowed_metrics: set[str]
    allowed_filter_names: set[str]
    allowed_chart_types: set[str]
    limits: dict[str, Any]


def load_semantic_layer(semantic_layer: dict[str, Any] | str) -> dict[str, Any]:
    """
    Accept either a loaded dict or YAML text/path.
    """
    if isinstance(semantic_layer, dict):
        return semantic_layer

    s = semantic_layer.strip()
    # If it looks like a path and exists, read it
    if ("\n" not in s) and os.path.exists(s):
        from pathlib import Path

        s = Path(s).read_text(encoding="utf-8")

    cfg = yaml.safe_load(s)
    if not isinstance(cfg, dict):
        raise PlannerValidationError("semantic_layer must be a YAML mapping/dict.")
    return cfg


def semantic_view(cfg: dict[str, Any]) -> SemanticLayerView:
    measures = {m["name"] for m in cfg.get("measures", []) if isinstance(m, dict) and "name" in m}
    derived = {
        m["name"] for m in cfg.get("derived_metrics", []) if isinstance(m, dict) and "name" in m
    }

    ops = cfg.get("allowed_operations", {}) or {}
    group_by = set(ops.get("group_by", []) or [])
    charts = set(ops.get("charts", []) or [])

    filters = set()
    for f in ops.get("filters", []) or []:
        if isinstance(f, dict) and "name" in f:
            filters.add(str(f["name"]))
        elif isinstance(f, str):
            filters.add(f)

    limits = cfg.get("limits", {}) or {}

    return SemanticLayerView(
        allowed_group_by=set(map(str, group_by)),
        allowed_metrics={*map(str, measures), *map(str, derived)},
        allowed_filter_names=set(map(str, filters)),
        allowed_chart_types=set(map(str, charts)),
        limits=limits,
    )


# -----------------------------
# LLM client (Yandex OpenAI-compatible responses API)
# -----------------------------


def _get_yandex_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required for LLM planner mode.") from e

    api_key = os.getenv("YANDEX_CLOUD_API_KEY", "").strip()
    folder = os.getenv("YANDEX_CLOUD_FOLDER", "").strip()
    base_url = os.getenv("YANDEX_CLOUD_BASE_URL", "https://ai.api.cloud.yandex.net/v1").strip()

    if not api_key:
        raise RuntimeError("YANDEX_CLOUD_API_KEY is not set")
    if not folder:
        raise RuntimeError("YANDEX_CLOUD_FOLDER is not set")

    return OpenAI(api_key=api_key, base_url=base_url, project=folder)


def _yandex_model_id() -> str:
    folder = os.getenv("YANDEX_CLOUD_FOLDER", "").strip()
    model = os.getenv("YANDEX_CLOUD_MODEL", "").strip()
    if not folder or not model:
        raise RuntimeError("YANDEX_CLOUD_FOLDER / YANDEX_CLOUD_MODEL must be set")
    return f"gpt://{folder}/{model}"


def _extract_response_text(resp: Any) -> str:
    out_parts: list[str] = []
    output = getattr(resp, "output", None) or []
    for item in output:
        if getattr(item, "type", None) == "message":
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "type", None) in ("output_text", "text"):
                    t = getattr(c, "text", None)
                    if t:
                        out_parts.append(str(t))
    if not out_parts:
        maybe = getattr(resp, "text", None)
        if isinstance(maybe, str) and maybe.strip():
            return maybe
    return "\n".join(out_parts).strip()


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise PlannerError("Empty model output (expected JSON).")
    if text.startswith("{") and text.endswith("}"):
        return text
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise PlannerError("Model output did not contain a JSON object.")
    return m.group(0)


# -----------------------------
# Planner
# -----------------------------


class Planner:
    def __init__(
        self,
        *,
        prompts_dir: str = "src/cost_agent_mvp/agent/prompts",
    ) -> None:
        self.loader = PromptLoader(prompts_dir=prompts_dir)

    @staticmethod
    def _validate_plan(plan: PlannerPlan, view: SemanticLayerView) -> PlannerPlan:
        bad_metrics = [m for m in plan.metrics if m not in view.allowed_metrics]
        if bad_metrics:
            raise PlannerValidationError(
                f"Unsupported metric(s): {bad_metrics}. Allowed: {sorted(view.allowed_metrics)}"
            )

        bad_dims = [d for d in plan.group_by if d not in view.allowed_group_by]
        if bad_dims:
            raise PlannerValidationError(
                f"Unsupported group_by dimension(s): {bad_dims}. Allowed: {sorted(view.allowed_group_by)}"
            )

        bad_filters = [f.name for f in plan.filters if f.name not in view.allowed_filter_names]
        if bad_filters:
            raise PlannerValidationError(
                f"Unsupported filter(s): {bad_filters}. Allowed: {sorted(view.allowed_filter_names)}"
            )

        bad_charts = [c.type for c in plan.chart_requests if c.type not in view.allowed_chart_types]
        if bad_charts:
            raise PlannerValidationError(
                f"Unsupported chart type(s): {bad_charts}. Allowed: {sorted(view.allowed_chart_types)}"
            )

        max_days_hard = int(view.limits.get("max_days_hard", 180))
        max_days_default = int(view.limits.get("max_days_default", 30))

        if plan.time_window.type == "last_n_days":
            n = plan.time_window.n_days or 0
            if n <= 0:
                raise PlannerValidationError("last_n_days requires n_days > 0.")
            if n > max_days_hard:
                raise PlannerValidationError(
                    f"Requested n_days={n} exceeds max_days_hard={max_days_hard}."
                )
            if n > max_days_default:
                plan.warnings.append(
                    f"Requested n_days={n} exceeds default limit {max_days_default}; "
                    f"executor may clamp to {max_days_default}."
                )

        drill_req = view.limits.get("drilldown_requires_filters", ["account_id"])
        drill_req = [str(x) for x in drill_req] if isinstance(drill_req, list) else ["account_id"]
        uses_chat_id = ("chat_id" in plan.group_by) or any(
            f.name == "chat_id_in" for f in plan.filters
        )
        if uses_chat_id:
            has_required = (
                any(f.name == "account_id_in" for f in plan.filters)
                if "account_id" in drill_req
                else True
            )
            if not has_required:
                raise PlannerValidationError(
                    "Drilldown by chat_id requires an account_id filter (account_id_in)."
                )

        if plan.top_n is not None:
            max_rows_hard = int(view.limits.get("max_rows_hard", 20000))
            if plan.top_n <= 0:
                raise PlannerValidationError("top_n must be > 0.")
            if plan.top_n > max_rows_hard:
                raise PlannerValidationError(
                    f"Requested top_n={plan.top_n} exceeds max_rows_hard={max_rows_hard}."
                )

        max_group_by_cols = int(view.limits.get("max_group_by_cols", 2))
        if len(plan.group_by) > max_group_by_cols:
            raise PlannerValidationError(
                f"Too many group_by dimensions ({len(plan.group_by)}). Max allowed: {max_group_by_cols}."
            )

        return plan

    @staticmethod
    def _mock_plan(user_text: str, view: SemanticLayerView) -> PlannerPlan:
        t = (user_text or "").strip().lower()
        if not t:
            return PlannerPlan(status="UNSUPPORTED", intent="unsupported", reason="Empty question.")

        def _extract_int(default: int) -> int:
            m = re.search(r"\btop\s+(\d+)\b", t)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return default
            return default

        if "trend" in t or "over last" in t or "last 7" in t or "last 14" in t:
            n = 7 if "14" not in t else 14
            metric = "total_cost_services" if "cost" in t else "active_users"
            if metric not in view.allowed_metrics:
                metric = next(iter(view.allowed_metrics))
            return PlannerPlan(
                status="OK",
                intent="trend",
                time_window=TimeWindow(type="last_n_days", n_days=n),
                metrics=[metric],
                group_by=["date"],
                filters=[FilterSpec(name="last_n_days", args={"n_days": n})],
                sort_by="date",
                sort_dir="asc",
                chart_requests=[],
            )

        if "top" in t and "account" in t:
            n = _extract_int(default=int(view.limits.get("default_top_n", 10)))
            metric = "total_cost_services"
            if metric not in view.allowed_metrics:
                metric = next(iter(view.allowed_metrics))
            return PlannerPlan(
                status="OK",
                intent="top_n",
                time_window=TimeWindow(type="yesterday"),
                metrics=[metric],
                group_by=["account_id"],
                top_n=n,
                sort_by=metric,
                sort_dir="desc",
                chart_requests=[],
            )

        if "breakdown" in t or "by service" in t or "service mix" in t:
            metric = "total_cost_services"
            if metric not in view.allowed_metrics:
                metric = next(iter(view.allowed_metrics))
            gb = (
                "service_component" if "service_component" in view.allowed_group_by else "chat_type"
            )
            return PlannerPlan(
                status="OK",
                intent="breakdown",
                time_window=TimeWindow(type="yesterday"),
                metrics=[metric],
                group_by=[gb],
                chart_requests=[],
            )

        return PlannerPlan(
            status="UNSUPPORTED",
            intent="unsupported",
            reason="Unsupported question in mock mode.",
        )

    def plan(
        self,
        *,
        user_text: str,
        semantic_layer: dict[str, Any] | str,
        mock_mode: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> PlannerPlan:
        cfg = load_semantic_layer(semantic_layer)
        view = semantic_view(cfg)

        if mock_mode:
            plan = self._mock_plan(user_text, view)
            if plan.status == "OK":
                return self._validate_plan(plan, view)
            return plan

        bundle = self.loader.load_and_render(
            "planner",
            system_vars={},
            user_vars={
                "USER_TEXT": user_text,
                "SEMANTIC_LAYER_YAML": yaml.safe_dump(cfg, sort_keys=False),
            },
        )

        client = _get_yandex_openai_client()
        resp = client.responses.create(
            model=_yandex_model_id(),
            input=[
                {"role": "system", "content": bundle.system},
                {"role": "user", "content": bundle.user},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        text = _extract_response_text(resp)
        json_text = _extract_first_json_object(text)

        try:
            data = json.loads(json_text)
        except Exception as e:
            raise PlannerError(f"Failed to parse planner JSON output: {e}\nRaw:\n{text}") from e

        try:
            plan = PlannerPlan.model_validate(data)
        except ValidationError as e:
            raise PlannerValidationError(
                f"Planner output did not match PlannerPlan schema: {e}"
            ) from e

        if plan.status == "OK":
            plan = self._validate_plan(plan, view)
        return plan
