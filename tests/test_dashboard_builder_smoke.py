from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.viz.dashboard_builder import build_standard_daily_dashboard


def _make_min_df(report_day: date) -> pd.DataFrame:
    # 2 days so churn + deltas are defined
    y = report_day - timedelta(days=1)
    rows = []
    rng = np.random.default_rng(0)

    # 5 accounts, each with 3 rows per day
    for d in [y, report_day]:
        for acc in range(1, 6):
            for i in range(3):
                tasks = float(rng.integers(0, 5))
                cls = float(rng.integers(0, 10))
                amocrm = float(rng.integers(0, 3))
                # Introduce some zeros (should be filtered out if all 3 are zero)
                if i == 0:
                    tasks, cls, amocrm = 0.0, 0.0, 0.0

                rows.append(
                    {
                        "account_id": acc,
                        "chat_id": acc * 1000 + i,
                        "chat_type": "telegram",
                        "date": d,
                        "total_cost_tasks": tasks,
                        "total_cost_classifications": cls,
                        "cost_amocrm_call": amocrm,
                        # dialog exists but must be ignored by services logic
                        "cost_dialog": (999.0 if (tasks == cls == amocrm == 0.0) else 0.0),
                    }
                )
    df = pd.DataFrame(rows)
    return df


def test_dashboard_builder_smoke(tmp_path) -> None:
    report_day = date(2025, 2, 1)
    df = _make_min_df(report_day)

    evidence = build_standard_daily_evidence(
        df, report_day, trend_days=7, top_accounts_n=5, hist_bins=10
    )
    out = tmp_path / "dashboard.png"
    p = build_standard_daily_dashboard(evidence, out)

    assert p.exists()
    assert p.stat().st_size > 0
