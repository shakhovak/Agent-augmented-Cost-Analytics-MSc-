from __future__ import annotations

import json
import logging
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from cost_agent_mvp.agent.prompt_loader import PromptLoader

# Same environment-loading approach as your attached script (load_dotenv) :contentReference[oaicite:1]{index=1}
load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate handlers (same style as your attached script) :contentReference[oaicite:2]{index=2}
for h in logger.handlers[:]:
    logger.removeHandler(h)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.propagate = False


# -----------------------------
# Structured output schema
# -----------------------------

TimeWindowType = Literal["yesterday", "last_n_days", "range"]


class TimeWindowModel(BaseModel):
    type: TimeWindowType
    n_days: int | None = None
    start: str | None = None  # YYYY-MM-DD
    end: str | None = None  # YYYY-MM-DD


class FiltersModel(BaseModel):
    account_id: list[int] | None = None
    chat_type: list[str] | None = None
    chat_id: list[str] | None = None
    has_tasks: bool | None = None
    has_classifications: bool | None = None
    has_both: bool | None = None


class ChartModel(BaseModel):
    id: str
    type: str
    table: str
    x: str | None = None
    y: Any | None = None  # str or list[str]
    label: str | None = None
    value: str | None = None
    title: str | None = None


class OutputsModel(BaseModel):
    evidence_tables: list[str] = Field(default_factory=list)
    charts: list[ChartModel] = Field(default_factory=list)


class IntentModel(BaseModel):
    question: str
    assumptions: list[str] = Field(default_factory=list)


class ConstraintsModel(BaseModel):
    top_n: int = 10
    max_rows: int = 2000
    max_days: int = 30


class PlannerPlan(BaseModel):
    status: Literal["OK", "UNSUPPORTED"]
    mode: Literal["button", "ad_hoc"] = "ad_hoc"
    template_id: str | None = None

    time_window: TimeWindowModel = Field(default_factory=lambda: TimeWindowModel(type="yesterday"))
    filters: FiltersModel = Field(default_factory=FiltersModel)
    intent: IntentModel

    outputs: OutputsModel = Field(default_factory=OutputsModel)
    constraints: ConstraintsModel = Field(default_factory=ConstraintsModel)

    reason: str | None = None
    suggested_template: str | None = None
    suggested_plan: dict[str, Any] | None = None


# -----------------------------
# LLM client adapter
# -----------------------------


def _get_openai_client():
    """
    Minimal OpenAI client factory.
    If you already have a shared client (like util_minimal.client in your other project),
    you can swap this to import it.

    We keep it local to avoid hard dependency at import time.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI package not available. Install openai or plug in your existing client."
        ) from e

    return OpenAI()


# -----------------------------
# Planner
# -----------------------------


class Planner:
    def __init__(
        self,
        prompts_dir: str = "src/agent/prompts",
        model_env: str = "OPENAI_MODEL",
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.model_env = model_env
        self.default_model = default_model

    def build_messages(
        self,
        user_text: str,
        semantic_layer_yaml: str,
        report_templates_yaml: str = "",
    ) -> list[dict[str, str]]:
        bundle = self.loader.load_and_render(
            "planner",
            user_vars={
                "USER_TEXT": user_text,
                "SEMANTIC_LAYER_YAML": semantic_layer_yaml,
                "REPORT_TEMPLATES_YAML": report_templates_yaml or "",
            },
            system_vars={},
        )
        return [
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": bundle.user},
        ]

    def plan(
        self,
        user_text: str,
        semantic_layer_yaml: str,
        report_templates_yaml: str = "",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> PlannerPlan:
        """
        Uses structured outputs to return a PlannerPlan.

        This follows the same structured parsing pattern you use elsewhere :contentReference[oaicite:3]{index=3},
        but adapted to this repo and schema.
        """
        messages = self.build_messages(user_text, semantic_layer_yaml, report_templates_yaml)
        chosen_model = model or _get_env(self.model_env, self.default_model)

        client = _get_openai_client()

        try:
            # Preferred: structured parsing (similar to your parse() usage) :contentReference[oaicite:4]{index=4}
            resp = client.beta.chat.completions.parse(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=PlannerPlan,
            )
            plan_obj: PlannerPlan = resp.choices[0].message.parsed

            # Basic sanity fallback: ensure charts/tables exist for OK responses
            if plan_obj.status == "OK" and not plan_obj.outputs.evidence_tables:
                logger.info("Planner returned OK but no evidence_tables; adding minimal default.")
                plan_obj.outputs.evidence_tables = [
                    "kpi_today_vs_yesterday",
                    "top_accounts_by_total_cost",
                ]

            return plan_obj

        except Exception as e:
            logger.error(f"Planner structured parse failed: {e}")

            # Fallback: try normal completion, then JSON parse
            try:
                resp2 = client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp2.choices[0].message.content or ""
                data = json.loads(text)
                return PlannerPlan.model_validate(data)
            except Exception as e2:
                logger.error(f"Planner fallback JSON parse failed: {e2}")

                # Final safe fallback: UNSUPPORTED
                return PlannerPlan(
                    status="UNSUPPORTED",
                    mode="ad_hoc",
                    template_id=None,
                    time_window=TimeWindowModel(type="yesterday"),
                    filters=FiltersModel(),
                    intent=IntentModel(
                        question=user_text,
                        assumptions=[
                            "Planner failed to produce a valid plan; returning UNSUPPORTED."
                        ],
                    ),
                    outputs=OutputsModel(evidence_tables=[], charts=[]),
                    constraints=ConstraintsModel(),
                    reason="Planner failed to parse model output into a valid plan.",
                    suggested_template="standard_daily_report",
                    suggested_plan=None,
                )


def _get_env(key: str, default: str) -> str:
    import os

    v = os.getenv(key)
    return v.strip() if v and v.strip() else default
