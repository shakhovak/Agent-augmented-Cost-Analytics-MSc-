from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import load_csv

FIXTURE_CSV = Path("tests/fixtures/kpi_fixture.csv")
GOLDEN_JSON = Path("tests/golden/kpis_expected.json")

REPORT_DAY = pd.to_datetime("2025-01-02").date()


def _to_builtin(x: Any) -> Any:
    # Convert numpy/pandas scalars to plain Python for stable JSON.
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass

    # JSON-friendly dates
    if isinstance(x, date | datetime):
        return x.isoformat()

    return x


def _normalize_kpi_row(d: dict[str, Any], decimals: int = 6) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted(d.keys()):
        v = _to_builtin(d[k])
        if isinstance(v, float):
            out[k] = round(v, decimals)
        else:
            out[k] = v
    return out


def _compute_kpis_snapshot() -> dict[str, Any]:
    df = load_csv(str(FIXTURE_CSV))
    evidence = build_standard_daily_evidence(
        df_all=df, report_day=REPORT_DAY, trend_days=2, top_n=3, cap=50
    )
    row = evidence["kpis"].iloc[0].to_dict()
    return _normalize_kpi_row(row, decimals=6)


def test_kpis_golden_snapshot():
    snapshot = _compute_kpis_snapshot()

    # Update golden file intentionally when KPI semantics change
    if os.getenv("UPDATE_GOLDEN") == "1":
        GOLDEN_JSON.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_JSON.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Make the test pass in update mode
        assert True
        return

    expected = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    assert snapshot == expected, (
        "KPI golden snapshot mismatch.\n"
        "If this change is intentional, re-run with UPDATE_GOLDEN=1 and commit the updated JSON.\n"
        f"Computed: {snapshot}\nExpected: {expected}\n"
    )
