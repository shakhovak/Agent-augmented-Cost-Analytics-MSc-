# src/agent/verifier.py
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

EvidencePack = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class VerifierIssue:
    type: str  # NUMBER_NOT_FOUND | DIRECTION_MISMATCH | UNSUPPORTED_CLAIM | MISSING_EVIDENCE
    detail: str
    severity: str  # low | medium | high
    suggested_fix: str


@dataclass(frozen=True)
class VerifierResult:
    status: str  # PASS | FAIL
    issues: list[VerifierIssue]
    summary: str


def _extract_numbers(text: str) -> list[float]:
    """
    Extract numbers from text in a conservative way.
    Supports:
      - 123
      - 123.45
      - 1,234.56
      - 1 234.56 (spaces)
    We intentionally ignore percentages like "12%" and treat them as numbers too (12).
    """
    if not text:
        return []
    # Grab sequences like 1,234.56 or 1234 or 1 234,56 (we normalize)
    candidates = re.findall(r"(?<!\w)[+-]?\d[\d\s,]*\.?\d*(?:%?)", text)
    out: list[float] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        is_pct = c.endswith("%")
        c = c[:-1] if is_pct else c
        # normalize thousand separators
        c_norm = c.replace(" ", "").replace(",", "")
        try:
            val = float(c_norm)
            # store pct as 0-100 number; we match approximately anyway
            out.append(val)
        except Exception:
            continue
    return out


def _round_variants(x: float) -> set[str]:
    """
    Create a small set of string variants to match typical formatting.
    """
    variants = set()
    for d in (0, 1, 2, 3):
        variants.add(f"{x:.{d}f}")
    # integer-ish
    variants.add(str(int(round(x))))
    return variants


def _evidence_number_bank(evidence: EvidencePack, max_cells: int = 20000) -> set[str]:
    """
    Build a bank of numeric values present in the evidence pack as string variants.
    We cap work to keep verification fast.
    """
    bank: set[str] = set()
    seen = 0

    for _, df in evidence.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # keep only numeric columns
        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            continue

        for col in num_df.columns:
            vals = num_df[col].dropna().values
            for v in vals:
                try:
                    x = float(v)
                except Exception:
                    continue
                for s in _round_variants(x):
                    bank.add(s)
                seen += 1
                if seen >= max_cells:
                    return bank

    return bank


def _get_kpi_row(evidence: EvidencePack) -> dict[str, float] | None:
    kpi = evidence.get("kpi_today_vs_yesterday")
    if kpi is None or not isinstance(kpi, pd.DataFrame) or kpi.empty:
        return None
    row = kpi.iloc[0].to_dict()
    # keep numeric-ish values
    out = {}
    for k, v in row.items():
        try:
            if isinstance(v, int | float) and not pd.isna(v):
                out[k] = float(v)
        except Exception:
            pass
    return out


def _direction_word(text: str) -> str | None:
    """
    Very conservative detection of direction words.
    """
    t = (text or "").lower()
    if any(w in t for w in ["increased", "up", "rose", "higher", "grew"]):
        return "up"
    if any(w in t for w in ["decreased", "down", "fell", "lower", "dropped"]):
        return "down"
    return None


def verify_numeric_fidelity(
    answer_text: str,
    evidence: EvidencePack,
    *,
    tolerance_abs: float = 0.01,
    require_kpi_table: bool = True,
) -> VerifierResult:
    """
    Deterministic numeric fidelity checks:
    1) All numbers mentioned should exist in evidence (approx via string variants).
       - We match via rounded string variants to handle formatting.
    2) Direction consistency for "total cost" if explicitly described as up/down.

    This is intentionally conservative: it flags likely hallucinated numbers,
    but may allow some numbers that coincidentally match. It’s a baseline verifier.
    """
    issues: list[VerifierIssue] = []

    if require_kpi_table and (
        "kpi_today_vs_yesterday" not in evidence
        or evidence["kpi_today_vs_yesterday"] is None
        or evidence["kpi_today_vs_yesterday"].empty
    ):
        issues.append(
            VerifierIssue(
                type="MISSING_EVIDENCE",
                detail="Evidence pack missing kpi_today_vs_yesterday, cannot verify key claims.",
                severity="high",
                suggested_fix="Ensure kpi_today_vs_yesterday is generated and passed to the verifier.",
            )
        )
        return VerifierResult(status="FAIL", issues=issues, summary="Missing KPI evidence.")

    bank = _evidence_number_bank(evidence)
    mentioned = _extract_numbers(answer_text)

    # Check number presence
    not_found = []
    for x in mentioned:
        # Ignore obviously non-informative small integers commonly used in prose (e.g., "7 days", "3–5 bullets")
        if abs(x) in {0, 1, 2, 3, 4, 5, 7, 10, 14, 30}:
            continue

        # match by rounded variants
        ok = any(s in bank for s in _round_variants(x))
        if not ok:
            not_found.append(x)

    if not_found:
        # cap the list for readability
        preview = ", ".join(str(x) for x in not_found[:8])
        issues.append(
            VerifierIssue(
                type="NUMBER_NOT_FOUND",
                detail=f"Numbers mentioned not found in evidence (approx match): {preview}"
                + (" ..." if len(not_found) > 8 else ""),
                severity="high" if len(not_found) >= 3 else "medium",
                suggested_fix="Remove or rephrase numeric claims; use only values present in the evidence tables.",
            )
        )

    # Direction check for total cost if analyst explicitly states up/down
    kpi_row = _get_kpi_row(evidence)
    if kpi_row:
        delta_pct = kpi_row.get("total_cost_delta_pct")
        direction_claim = _direction_word(answer_text)
        if delta_pct is not None and direction_claim is not None:
            if direction_claim == "up" and delta_pct < 0:
                issues.append(
                    VerifierIssue(
                        type="DIRECTION_MISMATCH",
                        detail=f"Answer implies total cost increased, but KPI delta is negative ({delta_pct:.3f}).",
                        severity="high",
                        suggested_fix="Update the narrative direction to match KPI deltas.",
                    )
                )
            if direction_claim == "down" and delta_pct > 0:
                issues.append(
                    VerifierIssue(
                        type="DIRECTION_MISMATCH",
                        detail=f"Answer implies total cost decreased, but KPI delta is positive ({delta_pct:.3f}).",
                        severity="high",
                        suggested_fix="Update the narrative direction to match KPI deltas.",
                    )
                )

    status = "PASS" if not issues else "FAIL"
    summary = "All checks passed." if status == "PASS" else f"{len(issues)} issue(s) found."
    return VerifierResult(status=status, issues=issues, summary=summary)


def verifier_result_to_json(res: VerifierResult) -> dict:
    """
    Convenience for saving verifier outputs in run logs.
    """
    return {
        "status": res.status,
        "issues": [
            {
                "type": i.type,
                "detail": i.detail,
                "severity": i.severity,
                "suggested_fix": i.suggested_fix,
            }
            for i in res.issues
        ],
        "summary": res.summary,
    }
