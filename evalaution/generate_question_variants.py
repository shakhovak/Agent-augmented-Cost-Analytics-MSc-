#!/usr/bin/env python3
"""
Generate question phrasings (variants) for each fact-instance record (JSONL).

- Reads system prompt from a text file (e.g., prompt_for_questions.txt).
- Calls Yandex Cloud OpenAI-compatible Responses API (same as analyst.py).
- Validates constraints: no relative dates, explicit 2025 dates, required tokens, etc.
- Writes output JSONL with question_variants + llm metadata per record.

Env (same as analyst.py):
  YANDEX_CLOUD_API_KEY
  YANDEX_CLOUD_FOLDER
  YANDEX_CLOUD_MODEL
  YANDEX_CLOUD_BASE_URL (optional, default used in analyst.py)

Usage:
  python generate_question_variants.py \
    --facts outputs/facts/facts.jsonl \
    --out outputs/facts/facts_with_questions.jsonl \
    --prompt prompt_for_questions.txt \
    --n 5 \
    --temperature 0.2
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dateutil import (
    parser as dtparser,
)  # pip package python-dateutil (usually installed)
from tqdm import tqdm

try:
    from cost_agent_mvp.agent.analyst import (  # type: ignore
        _extract_text_from_responses_api,
        _get_yandex_openai_client,
        _yandex_model_id,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import cost_agent_mvp.agent.analyst helpers. "
        "Run this from the repo root with the package available."
    ) from e


RELATIVE_DATE_WORDS = [
    "today",
    "yesterday",
    "tomorrow",
    "last week",
    "this week",
    "last month",
    "this month",
    "recent",
    "currently",
    "now",
]

# Detect obvious answer leakage:
DECIMAL_RE = re.compile(r"\b\d+[.,]\d+\b")
BIG_INT_RE = re.compile(r"\b\d{3,}\b")  # 3+ digits (blocks 1539, 3088, etc.)
# --- Date patterns (we allow multiple explicit formats, but still 2025 only) ---

# Common explicit date patterns to extract candidates
DATE_CANDIDATES = [
    re.compile(r"\b2025-\d{2}-\d{2}\b"),  # ISO
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]2025\b"),  # 20.01.2025 / 20/01/2025 / 20-01-2025
    re.compile(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
        r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
        r"\s+\d{1,2},?\s+2025\b",
        re.I,
    ),  # January 20, 2025 / Jan 20 2025
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
        r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
        r"\s+2025\b",
        re.I,
    ),  # 20th of January 2025 (without "of" still matches)
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+"
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
        r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
        r"\s+2025\b",
        flags=re.IGNORECASE,
    ),
]

ISO_DATE_RE = re.compile(r"\b2025-\d{2}-\d{2}\b")
DOT_DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.2025\b")
MONTH_NAME_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+2025\b",
    flags=re.IGNORECASE,
)


# Extract numeric tokens AFTER removing dates/account ids we explicitly allow
NUM_TOKEN_RE = re.compile(r"\b\d+\b")


def _has_relative_date(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in RELATIVE_DATE_WORDS)


def _extract_date_strings(text: str) -> list[str]:
    found = []
    for cre in DATE_CANDIDATES:
        for m in cre.finditer(text):
            found.append(m.group(0))
    # de-dupe preserving order
    out, seen = [], set()
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


ISO_DATE_RE = re.compile(r"\b2025-\d{2}-\d{2}\b")


def _parse_to_date(s: str) -> date | None:
    s = s.strip()

    # If it's ISO, parse deterministically
    if ISO_DATE_RE.fullmatch(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    # Otherwise parse flexibly (good for 20/01/2025, 20 Jan 2025, January 20, 2025)
    try:
        dt = dtparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.date()
    except Exception:
        return None


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _strip_targets_for_generator(fact: dict[str, Any]) -> dict[str, Any]:
    """
    For safety, do NOT send numeric targets to the question generator.
    It only needs type/params + allowed_numbers + required_tokens + metadata.
    """
    keep_keys = {
        "id",
        "type",
        "params",
        "metadata",
        "anchor_day",
        "allowed_numbers",
        "required_tokens",
    }
    return {k: v for k, v in fact.items() if k in keep_keys}


def _extract_dates_2025(text: str) -> list[str]:
    dates = []
    dates += ISO_DATE_RE.findall(text)
    dates += DOT_DATE_RE.findall(text)
    dates += MONTH_NAME_DATE_RE.findall(text)  # returns month name only; handle separately
    # MONTH_NAME_DATE_RE is grouped; reconstruct full matches instead:
    full = [m.group(0) for m in MONTH_NAME_DATE_RE.finditer(text)]
    dates = [d for d in dates if isinstance(d, str)]
    dates += full
    # de-dup preserving order
    seen = set()
    out = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _validate_question(
    question: str,
    *,
    expected_day: str | None = None,
    expected_start: str | None = None,
    expected_end: str | None = None,
    required_tokens: list[str] | None = None,
) -> tuple[bool, list[str]]:
    issues = []
    q = question.strip()
    if not q:
        return False, ["empty question"]

    if _has_relative_date(q):
        issues.append("contains relative date phrasing")

    # REQUIRED tokens (account_id/service label)
    if required_tokens:
        for tok in required_tokens:
            if tok and tok not in q:
                issues.append(f"missing required token: {tok}")

    # Block leaking answers:
    if DECIMAL_RE.search(q):
        issues.append("contains decimal number (likely answer leakage)")

    # Block 3+ digit integers except year 2025 and required account_id(s)
    big_ints = set(BIG_INT_RE.findall(q))
    allowed_big = {"2025"}
    if required_tokens:
        for tok in required_tokens:
            if tok.isdigit() and len(tok) >= 3:
                allowed_big.add(tok)
    leaked_big = sorted(x for x in big_ints if x not in allowed_big)
    if leaked_big:
        issues.append(f"contains disallowed 3+ digit numbers: {leaked_big}")

    # Date presence + match the FACT params
    date_strs = _extract_date_strings(q)
    parsed_dates = [d for s in date_strs if (d := _parse_to_date(s)) is not None]

    if not parsed_dates:
        issues.append("no parseable explicit date found")

    def iso_to_date(s: str) -> date:
        return datetime.strptime(s, "%Y-%m-%d").date()

    if expected_day:
        want = iso_to_date(expected_day)
        if want not in parsed_dates:
            issues.append(f"date does not match expected day {expected_day}")
    if expected_start and expected_end:
        want_s = iso_to_date(expected_start)
        want_e = iso_to_date(expected_end)
        if want_s not in parsed_dates or want_e not in parsed_dates:
            issues.append(f"window dates do not match expected {expected_start}..{expected_end}")

    return (len(issues) == 0), issues


def _call_question_llm(
    *,
    system_prompt: str,
    fact_for_llm: dict[str, Any],
    n_variants: int,
    temperature: float,
    max_output_tokens: int,
) -> tuple[list[str], dict[str, Any]]:
    client = (
        _get_yandex_openai_client()
    )  # uses env vars like in analyst.py :contentReference[oaicite:1]{index=1}
    model_id = _yandex_model_id()  # gpt://{folder}/{model} :contentReference[oaicite:2]{index=2}

    user_payload = {
        "N": n_variants,
        "FACT_SPEC": fact_for_llm,
    }
    user_text = json.dumps(user_payload, ensure_ascii=False)

    resp = client.responses.create(
        model=model_id,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )  # same call shape as analyst.py :contentReference[oaicite:3]{index=3}

    text = _extract_text_from_responses_api(
        resp
    )  # stable extraction :contentReference[oaicite:4]{index=4}
    if not text:
        raise RuntimeError("LLM returned empty response text")

    data = json.loads(text)
    variants = data.get("variants")
    if not isinstance(variants, list):
        raise ValueError("LLM output JSON must contain 'variants' list")

    questions: list[str] = []
    for item in variants:
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError("Each variant must be an object with key 'question'")
        questions.append(str(item["question"]))

    meta: dict[str, Any] = {}
    for k in ("id", "model", "status"):
        v = getattr(resp, k, None)
        if v is not None:
            meta[k] = v
    usage = getattr(resp, "usage", None)
    if usage is not None:
        meta["usage_total_tokens"] = getattr(usage, "total_tokens", None) or getattr(
            usage, "__dict__", None
        )

    meta["temperature"] = temperature
    meta["max_output_tokens"] = max_output_tokens
    return questions, meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate question variants for fact instances (JSONL)."
    )
    ap.add_argument("--facts", required=True, help="Input facts JSONL (from your sampler script)")
    ap.add_argument("--out", required=True, help="Output JSONL with question variants added")
    ap.add_argument(
        "--prompt", required=True, help="System prompt text file (prompt_for_questions.txt)"
    )
    ap.add_argument("--n", type=int, default=3, help="Number of variants per fact")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records")
    ap.add_argument(
        "--skip-existing", action="store_true", help="Skip if record already has question_variants"
    )
    ap.add_argument(
        "--max-retries", type=int, default=2, help="Retries per record if invalid output"
    )
    args = ap.parse_args()

    base_system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    if not base_system_prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt}")

    records = _read_jsonl(args.facts)
    if args.limit is not None:
        records = records[: args.limit]

    enriched: list[dict[str, Any]] = []

    for rec in tqdm(records, desc="Generating question variants", unit="fact"):
        # Skip if requested and already has questions
        if args.skip_existing and rec.get("question_variants"):
            enriched.append(rec)
            continue

        required_tokens = rec.get("required_tokens") or None
        params = rec.get("params") or {}
        expected_day = params.get("day")
        expected_start = params.get("start")
        expected_end = params.get("end")

        # Safety: LLM should not see targets
        fact_for_llm = _strip_targets_for_generator(rec)

        last_err: str | None = None
        best_payload: dict[str, Any] | None = None

        # Try multiple attempts if needed
        for attempt in range(args.max_retries + 1):
            # Do NOT mutate the base prompt; just add a short suffix on retries
            system_prompt = base_system_prompt
            if attempt > 0:
                system_prompt = (
                    base_system_prompt
                    + "\n\nIMPORTANT: Your previous output violated constraints. "
                    + "Return ONLY valid JSON and obey all constraints strictly. "
                    + "Make sure dates match FACT_SPEC.params exactly and avoid any answer numbers."
                )

            try:
                questions, meta = _call_question_llm(
                    system_prompt=system_prompt,
                    fact_for_llm=fact_for_llm,
                    n_variants=args.n,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )

                # Validate questions, but ALWAYS keep them for debugging
                validation: list[dict[str, Any]] = []
                ok_questions: list[str] = []

                for q in questions:
                    ok, issues = _validate_question(
                        q,
                        expected_day=expected_day,
                        expected_start=expected_start,
                        expected_end=expected_end,
                        required_tokens=required_tokens,
                    )
                    validation.append({"question": q, "ok": ok, "issues": issues})
                    if ok:
                        ok_questions.append(q)

                payload = {
                    "question_variants_raw": questions,
                    "question_validation": validation,
                    "question_variants_ok": ok_questions,
                    "question_llm_meta": meta,
                    "question_generation_attempt": attempt,
                }

                # If everything ok: accept immediately
                if len(ok_questions) == len(questions) and len(questions) == args.n:
                    best_payload = payload
                    last_err = None
                    break

                # If partially ok: keep as best so far, then retry to try improving
                best_payload = payload
                last_err = f"Validation failed (attempt {attempt}): {validation}"
                continue

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                # Keep retrying; no payload from this attempt
                continue

        # Build output record
        rec2 = dict(rec)

        if best_payload is not None:
            # Always store raw + validation
            rec2.update(best_payload)

            # Backward-compatible field for downstream steps:
            # prefer OK questions; if none OK, still keep raw so you never end up empty.
            ok_qs = rec2.get("question_variants_ok") or []
            raw_qs = rec2.get("question_variants_raw") or []
            rec2["question_variants"] = ok_qs if ok_qs else raw_qs

            if last_err is not None:
                rec2["question_llm_error"] = last_err
            else:
                rec2.pop("question_llm_error", None)
        else:
            # Complete failure: record the error, keep empty questions
            rec2["question_variants_raw"] = []
            rec2["question_variants_ok"] = []
            rec2["question_variants"] = []
            rec2["question_validation"] = []
            rec2["question_llm_error"] = last_err or "Unknown error"
            rec2["question_llm_meta"] = {}

        enriched.append(rec2)

    _write_jsonl(args.out, enriched)
    print(f"Wrote {len(enriched)} records to {args.out}")


if __name__ == "__main__":
    main()
