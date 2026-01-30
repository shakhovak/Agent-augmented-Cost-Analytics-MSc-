# src/agent/analyst.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# Evidence pack type: dict[str, pandas.DataFrame]
import pandas as pd
from dotenv import load_dotenv

from src.agent.prompt_loader import PromptLoader

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.propagate = False


EvidencePack = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class AnalystResult:
    answer_text: str
    model: str
    prompt_chars: int


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI package not available. Install openai.") from e
    return OpenAI()


def _env(key: str, default: str) -> str:
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
    max_tables: int = 12,
) -> str:
    """
    Compact, deterministic representation of evidence pack for LLM input.
    Keeps tables small to avoid context blowups.

    Format:
      [table_name]
      columns: ...
      head:
      <csv-like rows>
    """
    if not evidence:
        return "Evidence Pack is empty."

    parts = []
    for i, (name, df) in enumerate(evidence.items()):
        if i >= max_tables:
            parts.append(f"... truncated: more than {max_tables} tables")
            break

        df2 = _compact_df(df, max_rows=max_rows_per_table)
        parts.append(f"[{name}]")
        if df2.empty:
            parts.append("empty\n")
            continue

        parts.append("columns: " + ", ".join(map(str, df2.columns.tolist())))
        # Use a stable CSV-ish dump
        parts.append("head:")
        parts.append(df2.to_csv(index=False).strip())
        parts.append("")  # blank line between tables

    return "\n".join(parts).strip()


class Analyst:
    """
    LLM-based narrative generation over the Evidence Pack.

    Inputs: evidence pack (tables) + optional user question
    Output: plain text narrative grounded in provided evidence tables.
    """

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
        *,
        mode: str,
        user_text: str,
        evidence: EvidencePack,
        charts_metadata: str = "",
        max_rows_per_table: int = 8,
        max_tables: int = 12,
    ):
        bundle = self.loader.load_and_render(
            "analyst",
            user_vars={
                "MODE": mode,
                "USER_TEXT": user_text or "",
                "EVIDENCE_PACK_SUMMARY": evidence_pack_to_text(
                    evidence,
                    max_rows_per_table=max_rows_per_table,
                    max_tables=max_tables,
                ),
                "CHARTS_METADATA": charts_metadata or "",
            },
        )
        return [
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": bundle.user},
        ]

    def generate(
        self,
        *,
        evidence: EvidencePack,
        mode: str = "button",
        user_text: str = "",
        charts_metadata: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 900,
        max_rows_per_table: int = 8,
        max_tables: int = 12,
    ) -> AnalystResult:
        chosen_model = model or _env(self.model_env, self.default_model)
        messages = self.build_messages(
            mode=mode,
            user_text=user_text,
            evidence=evidence,
            charts_metadata=charts_metadata,
            max_rows_per_table=max_rows_per_table,
            max_tables=max_tables,
        )

        prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
        client = _get_openai_client()

        resp = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = resp.choices[0].message.content or ""
        return AnalystResult(
            answer_text=text.strip(), model=chosen_model, prompt_chars=prompt_chars
        )
