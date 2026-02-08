from __future__ import annotations

from datetime import date
from pathlib import Path

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import default_joint_schema, load_csv
from cost_agent_mvp.viz.dashboard_builder import build_standard_daily_dashboard


def main() -> None:
    csv_path = Path("data/joint_info_2025_with_anomalies.csv")
    report_day = date.fromisoformat("2025-07-20")

    df = load_csv(csv_path, schema=default_joint_schema())
    evidence = build_standard_daily_evidence(
        df_all=df,
        report_day=report_day,
    )

    out_png = Path("outputs/dashboard_sample.png")
    result = build_standard_daily_dashboard(
        evidence=evidence, out_path=out_png, title=f"KPI Dashboard for {report_day.isoformat()}"
    )

    print("Wrote:", getattr(result, "png_path", str(result)))


if __name__ == "__main__":
    main()
