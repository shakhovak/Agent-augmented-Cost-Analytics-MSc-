#!/usr/bin/env python
"""
Evaluate question variants end-to-end through Orchestrator:
LLM(Planner) -> deterministic tools -> LLM(Analyst)

Enhancements vs original:
- Supports negative cases (type == "negative_case"): score by planner UNSUPPORTED (+ optional reason_contains)
- Adds planner correctness metric for positive cases:
  - planner_status == OK
  - parsed time window matches FACT_SPEC.params (day or start/end)
  - required_tokens (if any) appear in planner JSON
- Writes per-run artifacts + summary CSV/JSONL

Expected fact formats:
POSITIVE fact (existing):
  {
    "id": 1,
    "type": "total_cost_single_day",
    "params": {"day": "2025-01-20"},
    "targets": [{"key":"sum_total_cost","value":..., "tol_abs":0.01}],
    "required_tokens": [... optional ...],
    "question_variants": [...],
    "anchor_day": "2025-12-31",
    "csv_fingerprint": {"csv_path":"..."}
  }

NEGATIVE fact:
  {
    "id": 9001,
    "type": "negative_case",
    "category": "relative_date",
    "expected": {"planner_status":"UNSUPPORTED", "reason_contains":["explicit","2025"]},
    "question_variants": [...],
    "anchor_day": "2025-12-31",
    "csv_fingerprint": {"csv_path":"..."}
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI as OriginalOpenAI
from tqdm import tqdm

from cost_agent_mvp.agent.analyst import _extract_text_from_responses_api
from cost_agent_mvp.agent.orchestrator import Orchestrator

# -------------------- Logging --------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True)
logger = logging.getLogger("eval")


# -------------------- LLM trace capture (Planner + Analyst) --------------------


@dataclass
class LLMCallRecord:
    kind: str  # "planner" | "analyst" | "other"
    system_preview: str
    user_preview: str
    response_text: str


CURRENT_TRACE: list[LLMCallRecord] | None = None


def _classify_call(input_msgs: list[dict[str, Any]]) -> str:
    sys_msg = ""
    for m in input_msgs:
        if m.get("role") == "system" and not sys_msg:
            sys_msg = str(m.get("content", ""))
    s = sys_msg.lower()
    if "you are the planner" in s:
        return "planner"
    if "you are the analyst" in s:
        return "analyst"
    return "other"


class LoggedOpenAI(OriginalOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original_create = self.responses.create

        def logged_create(*args, **kwargs):
            global CURRENT_TRACE

            input_msgs = kwargs.get("input") or []
            kind = _classify_call(input_msgs)

            sys_preview = ""
            user_preview = ""
            for msg in input_msgs:
                role = msg.get("role", "")
                content = str(msg.get("content", ""))
                if role == "system" and not sys_preview:
                    sys_preview = content[:800]
                if role == "user" and not user_preview:
                    user_preview = content[:800]

            result = original_create(*args, **kwargs)
            text = _extract_text_from_responses_api(result) or ""

            if CURRENT_TRACE is not None:
                CURRENT_TRACE.append(
                    LLMCallRecord(
                        kind=kind,
                        system_preview=sys_preview,
                        user_preview=user_preview,
                        response_text=text,
                    )
                )
            return result

        self.responses.create = logged_create


def patch_openai_globally() -> None:
    import openai  # noqa

    openai.OpenAI = LoggedOpenAI


# -------------------- IO helpers --------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------- Number extraction (final answer) --------------------
# Extract numbers like "3,088.27" or "3088.2691" but avoid date tokens.

DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # 2025-01-20
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b"),  # 20/01/2025, 20.01.2025
]

NUM_RE = re.compile(
    r"""
(?<!\w)
[-+]?
(?:
  (?:\d{1,3}(?:[,\s]\d{3})+|\d+)
)
(?:[.,]\d+)?
(?!\w)
""",
    re.VERBOSE,
)


def _mask_dates(text: str) -> str:
    masked = text
    for pat in DATE_PATTERNS:
        masked = pat.sub(" ", masked)
    return masked


def _to_float(num_str: str) -> float | None:
    s = num_str.strip()
    s = s.replace("\u202f", " ").replace("\xa0", " ").replace(" ", "")

    if "," in s and "." in s:
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2, 3):
            try:
                return float(parts[0] + "." + parts[1])
            except ValueError:
                return None
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    try:
        return float(s)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    masked = _mask_dates(text)
    vals: list[float] = []
    for m in NUM_RE.finditer(masked):
        v = _to_float(m.group(0))
        if v is not None:
            vals.append(v)
    return vals


# -------------------- Numeric scoring --------------------


@dataclass
class TargetCheck:
    key: str
    expected: float
    tol_abs: float
    matched: float | None
    abs_error: float | None
    passed: bool


def score_targets(extracted: list[float], targets: list[dict[str, Any]]) -> list[TargetCheck]:
    checks: list[TargetCheck] = []
    for t in targets:
        key = str(t["key"])
        expected = float(t["value"])
        tol = float(t.get("tol_abs", 0.0))

        best_val: float | None = None
        best_err: float | None = None
        for x in extracted:
            err = abs(x - expected)
            if best_err is None or err < best_err:
                best_err = err
                best_val = x

        passed = (best_err is not None) and (best_err <= tol)
        checks.append(
            TargetCheck(
                key=key,
                expected=expected,
                tol_abs=tol,
                matched=best_val,
                abs_error=best_err,
                passed=passed,
            )
        )
    return checks


# -------------------- Planner metric --------------------


@dataclass
class PlannerEval:
    parsed: bool
    status: str | None
    intent: str | None
    reason: str | None
    time_window_type: str | None
    day: str | None
    start: str | None
    end: str | None
    status_ok: bool | None
    time_window_ok: bool | None
    required_tokens_ok: bool | None
    reason_contains_ok: bool | None
    passed: bool


def _parse_planner_json(planner_text: str) -> dict[str, Any] | None:
    try:
        return json.loads(planner_text)
    except Exception:
        return None


def _planner_from_trace(
    trace: list[LLMCallRecord],
) -> tuple[str | None, dict[str, Any] | None]:
    planner_texts = [c.response_text for c in trace if c.kind == "planner" and c.response_text]
    if not planner_texts:
        return None, None
    raw = planner_texts[-1]
    return raw, _parse_planner_json(raw)


def _expected_planner_for_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """
    Returns an expectation object used for planner metric.
    For positive cases: expect status OK and correct time window + tokens.
    For negative cases: expect status UNSUPPORTED (+ optional reason_contains).
    """
    if fact.get("type") == "negative_case":
        exp = fact.get("expected") or {}
        return {
            "expected_status": exp.get("planner_status", "UNSUPPORTED"),
            "reason_contains": exp.get("reason_contains") or [],
            "expected_day": None,
            "expected_start": None,
            "expected_end": None,
            "required_tokens": [],  # optional; could also add for negatives
        }

    params = fact.get("params") or {}
    return {
        "expected_status": "OK",
        "reason_contains": [],
        "expected_day": params.get("day"),
        "expected_start": params.get("start"),
        "expected_end": params.get("end"),
        "required_tokens": [str(x) for x in (fact.get("required_tokens") or [])],
    }


def _check_planner(fact: dict[str, Any], planner_obj: dict[str, Any] | None) -> PlannerEval:
    exp = _expected_planner_for_fact(fact)

    if not planner_obj:
        return PlannerEval(
            parsed=False,
            status=None,
            intent=None,
            reason=None,
            time_window_type=None,
            day=None,
            start=None,
            end=None,
            status_ok=False,
            time_window_ok=None,
            required_tokens_ok=None,
            reason_contains_ok=None,
            passed=False,
        )

    status = planner_obj.get("status")
    intent = planner_obj.get("intent")
    reason = planner_obj.get("reason")

    tw = planner_obj.get("time_window") or {}
    tw_type = tw.get("type")
    day = tw.get("day")
    start = tw.get("start")
    end = tw.get("end")

    status_ok = status == exp["expected_status"]

    # Time window check:
    # - if expected_day present: compare to tw.day
    # - else if expected_start/end present: compare to tw.start/tw.end
    # - else: not applicable (negatives, etc.)
    expected_day = exp.get("expected_day")
    expected_start = exp.get("expected_start")
    expected_end = exp.get("expected_end")

    if expected_day:
        time_window_ok = day == expected_day
    elif expected_start and expected_end:
        time_window_ok = (start == expected_start) and (end == expected_end)
    else:
        time_window_ok = None

    # Required tokens check (if any)
    required_tokens = exp.get("required_tokens") or []
    if required_tokens:
        dumped = json.dumps(planner_obj, ensure_ascii=False)
        required_tokens_ok = all(tok in dumped for tok in required_tokens)
    else:
        required_tokens_ok = None

    # Reason substring check (for negatives, optional)
    reason_contains = exp.get("reason_contains") or []
    if reason_contains:
        r = (reason or "").lower()
        reason_contains_ok = all(s.lower() in r for s in reason_contains)
    else:
        reason_contains_ok = None

    # Overall planner pass:
    passed = bool(status_ok)

    # For positives: require time window match if applicable
    if fact.get("type") != "negative_case" and time_window_ok is not None:
        passed = passed and bool(time_window_ok)

    # If tokens expected, require them
    if required_tokens_ok is not None:
        passed = passed and bool(required_tokens_ok)

    # If reason substrings expected, require them
    # if reason_contains_ok is not None:
    #     passed = passed and bool(reason_contains_ok)

    return PlannerEval(
        parsed=True,
        status=status,
        intent=intent,
        reason=reason,
        time_window_type=tw_type,
        day=day,
        start=start,
        end=end,
        status_ok=status_ok,
        time_window_ok=time_window_ok,
        required_tokens_ok=required_tokens_ok,
        reason_contains_ok=reason_contains_ok,
        passed=passed,
    )


# -------------------- Helpers --------------------


def _fact_dir_name(fact_id: Any) -> str:
    if isinstance(fact_id, int):
        return f"fact_{fact_id:04d}"
    if fact_id is None:
        return "fact_unknown"
    s = str(fact_id)
    return f"fact_{s}"


# -------------------- Main runner --------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate question variants via Orchestrator.run_ad_hoc"
    )
    ap.add_argument("--facts", required=True, help="JSONL with facts + question_variants")
    ap.add_argument(
        "--csv-path",
        default=None,
        help="Override CSV path (else use csv_fingerprint.csv_path if present)",
    )
    ap.add_argument("--semantic-layer-path", required=True, help="Path to semantic layer YAML")
    ap.add_argument(
        "--prompts-dir",
        required=True,
        help="Prompts directory (planner/analyst prompts)",
    )
    ap.add_argument("--outputs-root", default="outputs/eval_questions", help="Root output dir")
    ap.add_argument("--suite-id", default="mvp_v0", help="Suite id for folder grouping")
    ap.add_argument("--limit-facts", type=int, default=None)
    ap.add_argument(
        "--limit-variants",
        type=int,
        default=None,
        help="Limit variants per fact (debug)",
    )
    ap.add_argument("--mock-planner", action="store_true")
    ap.add_argument("--mock-analyst", action="store_true")
    ap.add_argument("--mode", default="llm", choices=["llm", "qa"])
    args = ap.parse_args()

    patch_openai_globally()

    facts = read_jsonl(Path(args.facts))
    if args.limit_facts is not None:
        facts = facts[: args.limit_facts]

    suite_root = Path(args.outputs_root) / args.suite_id
    suite_root.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(
        outputs_root=str(suite_root),
        dataset_name="joint_costs_daily",
        semantic_layer_path=args.semantic_layer_path,
        prompts_dir=args.prompts_dir,
    )

    rows: list[dict[str, Any]] = []

    for fact in tqdm(facts, desc="Facts", unit="fact"):
        fact_id = fact.get("id")
        fact_type = fact.get("type")
        is_negative = fact_type == "negative_case"
        category = fact.get("category") if is_negative else None
        display_fact_type = f"negative/{category}" if is_negative and category else str(fact_type)

        anchor_day = fact.get("anchor_day")
        anchor_dt = (
            datetime.strptime(anchor_day, "%Y-%m-%d").date() if anchor_day else date(2025, 12, 31)
        )

        csv_path = args.csv_path or (fact.get("csv_fingerprint", {}) or {}).get("csv_path")
        if not csv_path:
            raise ValueError(
                "csv_path not provided and not found in fact['csv_fingerprint']['csv_path']."
            )

        questions = fact.get("question_variants") or fact.get("question_variants_ok") or []
        if args.limit_variants is not None:
            questions = questions[: args.limit_variants]

        targets = fact.get("targets") or []
        if is_negative:
            targets = []  # negatives are scored by planner behavior, not numeric targets

        if not questions:
            rows.append(
                {
                    "fact_id": fact_id,
                    "fact_type": display_fact_type,
                    "is_negative": is_negative,
                    "variant_idx": None,
                    "question": None,
                    "status": "NO_QUESTIONS",
                }
            )
            continue

        for j, q in enumerate(questions, start=1):
            run_dir = suite_root / _fact_dir_name(fact_id) / f"v{j:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            global CURRENT_TRACE
            CURRENT_TRACE = []

            try:
                out = orch.run_ad_hoc(
                    csv_path=str(csv_path),
                    user_text=str(q),
                    anchor_day=anchor_dt,
                    output_root=str(run_dir),
                    mock_planner=bool(args.mock_planner),
                    mock_analyst=bool(args.mock_analyst),
                    mode=str(args.mode),
                    build_dashboard=False,
                )
                answer_text = getattr(out, "answer_text", "") or ""
            except Exception as e:
                trace_dump = [asdict(x) for x in (CURRENT_TRACE or [])]
                write_json(run_dir / "llm_trace.json", trace_dump)

                rows.append(
                    {
                        "fact_id": fact_id,
                        "fact_type": display_fact_type,
                        "is_negative": is_negative,
                        "variant_idx": j,
                        "question": q,
                        "status": "ERROR",
                        "error": f"{type(e).__name__}: {e}",
                        "run_dir": str(run_dir),
                    }
                )
                CURRENT_TRACE = None
                continue

            trace = CURRENT_TRACE or []

            # Save trace + responses
            write_json(run_dir / "llm_trace.json", [asdict(x) for x in trace])

            planner_raw, planner_obj = _planner_from_trace(trace)
            analyst_texts = [
                c.response_text for c in trace if c.kind == "analyst" and c.response_text
            ]

            if planner_raw:
                (run_dir / "planner_response.json").write_text(planner_raw, encoding="utf-8")
            if analyst_texts:
                (run_dir / "analyst_response.txt").write_text(analyst_texts[-1], encoding="utf-8")
            (run_dir / "final_answer.txt").write_text(answer_text, encoding="utf-8")

            # Planner metric (always computed)
            planner_eval = _check_planner(fact, planner_obj)
            write_json(run_dir / "planner_eval.json", asdict(planner_eval))

            # Numeric fidelity (positives only)
            extracted: list[float] = []
            checks: list[TargetCheck] = []
            passed_all: bool | None = None
            passed_targets: int | None = None
            n_targets: int | None = None

            if not is_negative:
                extracted = extract_numbers(answer_text)
                checks = score_targets(extracted, targets)
                passed_all = all(c.passed for c in checks) if checks else False
                passed_targets = sum(1 for c in checks if c.passed)
                n_targets = len(targets)

                write_json(run_dir / "extracted_numbers.json", {"numbers": extracted})
                write_json(
                    run_dir / "target_checks.json",
                    {"checks": [asdict(c) for c in checks]},
                )

            # Decide run pass/fail:
            # - positives: end-to-end numeric pass
            # - negatives: planner refusal pass
            run_passed = planner_eval.passed if is_negative else bool(passed_all)

            rows.append(
                {
                    "fact_id": fact_id,
                    "fact_type": display_fact_type,
                    "is_negative": is_negative,
                    "variant_idx": j,
                    "question": q,
                    "status": "OK",
                    # Primary outcome (E2E for positives; refusal correctness for negatives)
                    "passed": run_passed,
                    # Numeric fields (positives only)
                    "n_numbers_extracted": (len(extracted) if not is_negative else None),
                    "targets": n_targets,
                    "passed_targets": passed_targets,
                    # Planner metric fields (new)
                    "planner_passed": planner_eval.passed,
                    "planner_status": planner_eval.status,
                    "planner_intent": planner_eval.intent,
                    "planner_time_window_ok": planner_eval.time_window_ok,
                    "planner_required_tokens_ok": planner_eval.required_tokens_ok,
                    "planner_reason_contains_ok": planner_eval.reason_contains_ok,
                    "planner_day": planner_eval.day,
                    "planner_start": planner_eval.start,
                    "planner_end": planner_eval.end,
                    "run_dir": str(run_dir),
                }
            )

            CURRENT_TRACE = None

    # Summary outputs
    df = pd.DataFrame(rows)
    df.to_csv(suite_root / "summary.csv", index=False)
    (suite_root / "summary.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    logger.info(f"Wrote summary to: {suite_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
