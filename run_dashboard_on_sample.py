from __future__ import annotations

from datetime import date
from pathlib import Path

from cost_agent_mvp.analytics.evidence_pack import build_standard_daily_evidence
from cost_agent_mvp.data.schema import default_joint_schema, load_csv
from cost_agent_mvp.reports.summary_generator import generate_daily_summary
from cost_agent_mvp.viz.dashboard_builder import build_standard_daily_dashboard


def main() -> None:
    csv_path = Path("data/joint_info_2025_with_anomalies.csv")
    report_day = date.fromisoformat("2025-07-20")

    df = load_csv(csv_path, schema=default_joint_schema())
    evidence = build_standard_daily_evidence(
        df_all=df,
        report_day=report_day,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dashboard PNG
    out_png = out_dir / "dashboard_sample.png"
    result = build_standard_daily_dashboard(
        evidence=evidence,
        out_path=out_png,
        title=f"KPI Dashboard for {report_day.isoformat()}",
    )
    print("Wrote:", getattr(result, "png_path", str(result)))

    # 2) Deterministic text summary
    summary_text = generate_daily_summary(evidence=evidence, top_n=3)
    out_txt = out_dir / "summary_sample.txt"
    out_txt.write_text(summary_text, encoding="utf-8")
    print("Wrote:", out_txt)


if __name__ == "__main__":
    main()
