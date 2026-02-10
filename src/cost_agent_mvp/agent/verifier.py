from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

EvidencePack = dict[str, pd.DataFrame]

# -------- Extraction / normalization helpers --------

# Remove ISO date patterns before numeric extraction to avoid false positives
_HYPHEN_CHARS = r"[-\u2010\u2011\u2012\u2013\u2212]"
_DATE_RE = re.compile(rf"\b\d{{4}}{_HYPHEN_CHARS}\d{{2}}{_HYPHEN_CHARS}\d{{2}}\b")
# Conservative numeric extraction: 1,234.56 | 1234.56 | 1 234.56 | 12% etc.
_NUM_RE = re.compile(r"(?<!\w)[+-]?\d[\d\s,]*\.?\d*(?:%?)")

# Direction detection (conservative, English only for v0)
_UP_WORDS = ("increased", "increase", "up", "rose", "higher", "grew", "grow")
_DOWN_WORDS = ("decreased", "decrease", "down", "fell", "lower", "dropped", "drop")


@dataclass(frozen=True)
class VerifierIssue:
    type: str  # NUMBER_NOT_FOUND | DIRECTION_MISMATCH | UNSUPPORTED_CLAIM | MISSING_EVIDENCE
    detail: str
    severity: str  # low | medium | high
    suggested_fix: str


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    issues: list[VerifierIssue]
    summary: str
    score: float | None = None  # optional; pass=1.0 fail=0.0 by default

    @property
    def status(self) -> str:
        # Backwards compatible alias
        return "PASS" if self.passed else "FAIL"


def extract_numbers(text: str) -> list[float]:
    """
    Extract numbers from text in a conservative way.
    Supports:
      - 123
      - 123.45
      - 1,234.56
      - 1 234.56 (spaces)
      - 12% (stored as 12.0)
    We strip ISO dates first to avoid matching YYYY-MM-DD parts.
    """
    if not text:
        return []

    # strip dates to avoid matching 2025-10-01 as numbers
    scrubbed = _DATE_RE.sub(" ", text)

    candidates = _NUM_RE.findall(scrubbed)
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
            out.append(float(c_norm))
        except Exception:
            continue
    return out


def _round_variants(x: float) -> set[str]:
    """
    Create a small set of string variants to match typical formatting.
    - 0..3 decimal variants
    - integer-ish variant
    """
    variants = set()
    for d in (0, 1, 2, 3):
        variants.add(f"{x:.{d}f}")
    variants.add(str(int(round(x))))
    return variants


def evidence_numeric_bank(evidence: EvidencePack, *, max_cells: int = 20_000) -> set[str]:
    """
    Build a bank of numeric values present in the evidence pack as string variants.

    Important v0 behavior:
    - Adds rounded variants for each numeric cell.
    - Also adds *100 variants for small ratios (abs(x) <= 1) to allow percent-form answers
      when evidence stores proportions (e.g., delta_pct=0.059 but answer says 5.9%).
    """
    bank: set[str] = set()
    seen = 0

    for _, df in evidence.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

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

                # percent scaling for ratios / shares / delta_pct style values
                if 0.0001 <= abs(x) <= 1.0:
                    for s in _round_variants(x * 100.0):
                        bank.add(s)

                seen += 1
                if seen >= max_cells:
                    return bank

    return bank


def _get_kpi_row(evidence: EvidencePack) -> dict[str, float] | None:
    """
    Prefer new table name 'kpis'; fall back to legacy 'kpi_today_vs_yesterday'.
    """
    for table_name in ("kpis", "kpi_today_vs_yesterday"):
        kpi = evidence.get(table_name)
        if kpi is None or not isinstance(kpi, pd.DataFrame) or kpi.empty:
            continue
        row = kpi.iloc[0].to_dict()
        out: dict[str, float] = {}
        for k, v in row.items():
            try:
                if isinstance(v, int | float) and not pd.isna(v):
                    out[k] = float(v)
            except Exception:
                continue
        return out
    return None


def _direction_word(text: str) -> str | None:
    """
    Very conservative detection of direction words.
    If both up and down words appear, return None (ambiguous).
    """
    t = (text or "").lower()
    up = any(w in t for w in _UP_WORDS)
    down = any(w in t for w in _DOWN_WORDS)
    if up and down:
        return None
    if up:
        return "up"
    if down:
        return "down"
    return None


def verify_numeric_fidelity(
    answer_text: str,
    evidence: EvidencePack,
    *,
    require_kpi_table: bool = True,
) -> VerifierResult:
    """
    Deterministic fidelity checks:

    1) Numeric fidelity:
       All non-trivial numbers mentioned in the answer should exist in evidence
       (approx via rounded string variants; also supports % scaling).

    2) Direction fidelity (v0):
       If answer implies total cost went up/down, verify it matches KPI delta sign.
       We use:
         - total_cost_services_delta_pct (preferred, new KPI schema), or
         - total_cost_delta_pct (legacy)
    """
    issues: list[VerifierIssue] = []

    if require_kpi_table:
        kpi_row = _get_kpi_row(evidence)
        if not kpi_row:
            issues.append(
                VerifierIssue(
                    type="MISSING_EVIDENCE",
                    detail="Evidence pack missing KPI table ('kpis' or 'kpi_today_vs_yesterday'); cannot verify key claims.",
                    severity="high",
                    suggested_fix="Ensure KPI table is generated and passed to the verifier.",
                )
            )
            return VerifierResult(
                passed=False,
                issues=issues,
                summary="Missing KPI evidence.",
                score=0.0,
            )

    bank = evidence_numeric_bank(evidence)
    mentioned = extract_numbers(answer_text)

    # Numbers to ignore (common prose counts / fixed caps / window sizes)
    IGNORE = {0, 1, 2, 3, 4, 5, 7, 10, 14, 30}

    not_found: list[float] = []
    for x in mentioned:
        if abs(x) in IGNORE:
            continue

        ok = any(s in bank for s in _round_variants(x))
        if not ok:
            not_found.append(x)

    if not_found:
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

    # Direction check for total cost if analyst explicitly states up/down.
    kpi_row = _get_kpi_row(evidence) or {}
    direction_claim = _direction_word(answer_text)

    # Prefer new KPI column
    delta_pct = None
    for key in ("total_cost_services_delta_pct", "total_cost_delta_pct"):
        if key in kpi_row:
            delta_pct = kpi_row.get(key)
            break

    if delta_pct is not None and direction_claim is not None:
        if direction_claim == "up" and float(delta_pct) < 0:
            issues.append(
                VerifierIssue(
                    type="DIRECTION_MISMATCH",
                    detail=f"Answer implies total cost increased, but KPI delta is negative ({float(delta_pct):.3f}).",
                    severity="high",
                    suggested_fix="Update the narrative direction to match KPI deltas.",
                )
            )
        if direction_claim == "down" and float(delta_pct) > 0:
            issues.append(
                VerifierIssue(
                    type="DIRECTION_MISMATCH",
                    detail=f"Answer implies total cost decreased, but KPI delta is positive ({float(delta_pct):.3f}).",
                    severity="high",
                    suggested_fix="Update the narrative direction to match KPI deltas.",
                )
            )

    passed = not issues
    score = 1.0 if passed else 0.0
    summary = "All checks passed." if passed else f"{len(issues)} issue(s) found."
    return VerifierResult(passed=passed, issues=issues, summary=summary, score=score)


def verifier_result_to_json(res: VerifierResult) -> dict:
    """
    Convenience for saving verifier outputs in run logs.
    """
    return {
        "passed": res.passed,
        "status": res.status,
        "score": res.score,
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
