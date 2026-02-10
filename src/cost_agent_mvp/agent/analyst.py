from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from cost_agent_mvp.agent.prompt_loader import PromptLoader

load_dotenv()

logger = logging.getLogger(__name__)

EvidencePack = dict[str, pd.DataFrame]

_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Z0-9_]+)\s*}}")


@dataclass(frozen=True)
class AnalystResult:
    """
    Structured result for analyst generation.
    Keep this stable: it becomes part of your evaluation artifacts.
    """

    answer_text: str
    model: str
    prompt_chars: int
    citations: list[str] | None = None
    warnings: list[str] | None = None
    raw: dict[str, Any] | None = None


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default


def _compact_df(df: pd.DataFrame, max_rows: int = 8, max_cols: int = 20) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    cols = list(df.columns)[:max_cols]
    return df.loc[:, cols].head(max_rows).copy()


def evidence_pack_to_text(
    evidence: EvidencePack,
    *,
    max_rows_per_table: int = 8,
    max_cols_per_table: int = 20,
    max_tables: int = 12,
    max_chars: int = 25_000,
) -> str:
    """
    Compact, deterministic representation of Evidence Pack for LLM input.

    - Stable ordering: tables sorted by name.
    - Compact tables: head only, limited columns.
    - Hard size cap: max_chars.
    """
    if not evidence:
        return "Evidence Pack is empty."

    parts: list[str] = []
    used = 0

    for i, name in enumerate(sorted(evidence.keys())):
        if i >= max_tables:
            parts.append(f"... truncated: more than {max_tables} tables")
            break

        df = evidence[name]
        df2 = _compact_df(df, max_rows=max_rows_per_table, max_cols=max_cols_per_table)

        block: list[str] = [f"[{name}]"]
        if df2.empty:
            block.append("empty\n")
        else:
            block.append("columns: " + ", ".join(map(str, df2.columns.tolist())))
            block.append("head:")
            block.append(df2.to_csv(index=False).strip())
            block.append("")  # blank line between tables

        block_text = "\n".join(block)
        if used + len(block_text) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                parts.append(block_text[:remaining])
            parts.append("\n... truncated to max_chars")
            break

        parts.append(block_text)
        used += len(block_text)

    return "\n".join(parts).strip()


def _get_yandex_openai_client():
    """
    Yandex Cloud OpenAI-compatible client (Responses API).
    Uses env:
      YANDEX_CLOUD_API_KEY
      YANDEX_CLOUD_FOLDER
      YANDEX_CLOUD_BASE_URL (optional; default: https://ai.api.cloud.yandex.net/v1)
    """
    try:
        import openai  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not available. Install openai.") from e

    api_key = _env("YANDEX_CLOUD_API_KEY")
    folder = _env("YANDEX_CLOUD_FOLDER")
    base_url = _env("YANDEX_CLOUD_BASE_URL", "https://ai.api.cloud.yandex.net/v1")

    if not api_key:
        raise RuntimeError("YANDEX_CLOUD_API_KEY is not set")
    if not folder:
        raise RuntimeError("YANDEX_CLOUD_FOLDER is not set")

    # Some openai client versions accept `project=...`. If not, fall back gracefully.
    try:
        return openai.OpenAI(api_key=api_key, base_url=base_url, project=folder)
    except TypeError:
        return openai.OpenAI(api_key=api_key, base_url=base_url)


def _yandex_model_id() -> str:
    folder = _env("YANDEX_CLOUD_FOLDER")
    model = _env("YANDEX_CLOUD_MODEL")
    if not folder:
        raise RuntimeError("YANDEX_CLOUD_FOLDER is not set")
    if not model:
        raise RuntimeError("YANDEX_CLOUD_MODEL is not set")
    return f"gpt://{folder}/{model}"


def _extract_text_from_responses_api(resp: Any) -> str:
    """
    Parse OpenAI Responses API shape:
      resp.output = [ ... ResponseOutputMessage(... content=[ResponseOutputText(text=...), ...]) ...]
    """
    # Best effort: support both attribute and dict-like access
    output = getattr(resp, "output", None)
    if not output and isinstance(resp, dict):
        output = resp.get("output")

    texts: list[str] = []
    if output:
        for item in output:
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            role = getattr(item, "role", None) or (
                item.get("role") if isinstance(item, dict) else None
            )
            if item_type != "message":
                continue
            if role and role != "assistant":
                continue
            content = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            if not content:
                continue
            for c in content:
                c_type = getattr(c, "type", None) or (
                    c.get("type") if isinstance(c, dict) else None
                )
                if c_type != "output_text":
                    continue
                t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "")
                if t:
                    texts.append(str(t))

    # Some client versions may expose resp.output_text or resp.text
    if not texts:
        for attr in ("output_text", "text"):
            t = getattr(resp, attr, None)
            if t:
                texts.append(str(t))

    return "\n".join(texts).strip()


def _mock_answer(*, mode: str, question: str, evidence: EvidencePack) -> str:
    """
    Deterministic offline response used in CI/tests.
    No numbers computed; only references evidence table names and first visible cells.
    """
    table_names = sorted(evidence.keys())
    lines = [
        "MOCK Analyst (offline mode)",
        f"Mode: {mode}",
    ]
    if question:
        lines.append(f"Question: {question}")

    lines.append("Tables: " + ", ".join(table_names) if table_names else "Tables: <none>")

    # Include a tiny, deterministic snippet from the first non-empty table for stability.
    for name in table_names:
        df = evidence[name]
        if isinstance(df, pd.DataFrame) and not df.empty:
            df2 = _compact_df(df, max_rows=2, max_cols=5)
            lines.append(f"[{name}] head:")
            lines.append(df2.to_csv(index=False).strip())
            break

    lines.append("Note: This is a deterministic mock response (no external calls).")
    return "\n".join(lines).strip()


class Analyst:
    """
    LLM narrative generation grounded in Evidence Pack.

    Supports:
      - mock_mode=True: deterministic output for CI/tests (no network)
      - mock_mode=False: Yandex Cloud OpenAI-compatible Responses API call
    """

    def __init__(
        self,
        *,
        prompts_dir: str | None = None,
        default_mode: str = "button",
    ) -> None:
        # Prefer PromptLoader's own default if prompts_dir is None.
        self.loader = (
            PromptLoader(prompts_dir=prompts_dir) if prompts_dir is not None else PromptLoader()
        )
        self.default_mode = default_mode

    def build_prompt(
        self,
        *,
        mode: str,
        user_text: str,
        evidence: EvidencePack,
        charts_metadata: str = "",
        max_rows_per_table: int = 8,
        max_cols_per_table: int = 20,
        max_tables: int = 12,
        max_chars: int = 25_000,
    ) -> tuple[str, str]:
        if not evidence:
            raise ValueError("Evidence Pack is empty (no tables).")

        evidence_text = evidence_pack_to_text(
            evidence,
            max_rows_per_table=max_rows_per_table,
            max_cols_per_table=max_cols_per_table,
            max_tables=max_tables,
            max_chars=max_chars,
        )

        bundle = self.loader.load_and_render(
            "analyst",
            system_vars=None,
            user_vars={
                "MODE": mode,
                "USER_TEXT": user_text or "",
                "EVIDENCE_PACK_SUMMARY": evidence_text,
                "CHARTS_METADATA": charts_metadata or "",
            },
        )
        return bundle.system, bundle.user

    def generate(
        self,
        *,
        evidence: EvidencePack,
        mode: str | None = None,
        user_text: str = "",
        charts_metadata: str = "",
        temperature: float = 0.2,
        max_output_tokens: int = 900,
        max_rows_per_table: int = 8,
        max_cols_per_table: int = 20,
        max_tables: int = 12,
        max_chars: int = 25_000,
        mock_mode: bool = False,
    ) -> AnalystResult:
        mode = mode or self.default_mode

        if not evidence:
            raise ValueError("Evidence Pack is empty (no tables).")

        # Mock mode (CI-safe)
        if mock_mode:
            text = _mock_answer(mode=mode, question=user_text, evidence=evidence)
            return AnalystResult(
                answer_text=text,
                model="mock",
                prompt_chars=0,
                warnings=["mock_mode=True"],
                raw=None,
            )

        system_text, user_prompt = self.build_prompt(
            mode=mode,
            user_text=user_text,
            evidence=evidence,
            charts_metadata=charts_metadata,
            max_rows_per_table=max_rows_per_table,
            max_cols_per_table=max_cols_per_table,
            max_tables=max_tables,
            max_chars=max_chars,
        )

        # Compose input for Responses API as messages.
        prompt_chars = len(system_text) + len(user_prompt)
        client = _get_yandex_openai_client()
        model_id = _yandex_model_id()

        resp = client.responses.create(
            model=model_id,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        answer = _extract_text_from_responses_api(resp)

        if not answer:
            raise RuntimeError("LLM returned empty response text.")

        # Keep raw minimal (avoid huge objects); store ids and usage if present.
        raw: dict[str, Any] = {}
        for k in ("id", "model", "status"):
            v = getattr(resp, k, None)
            if v is not None:
                raw[k] = v
        usage = getattr(resp, "usage", None)
        if usage is not None:
            raw["usage"] = getattr(usage, "total_tokens", None) or getattr(usage, "__dict__", None)

        return AnalystResult(
            answer_text=answer,
            model=model_id,
            prompt_chars=prompt_chars,
            raw=raw or None,
        )
