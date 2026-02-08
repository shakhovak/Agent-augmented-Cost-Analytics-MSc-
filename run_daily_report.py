from datetime import date

from cost_agent_mvp.reports.daily_report import run_standard_daily_report

artifacts = run_standard_daily_report(
    csv_path="data/joint_info_2025_with_anomalies.csv",
    report_day=date.fromisoformat("2025-07-20"),
)
print("Run folder:", artifacts.output_dir)
print("Dashboard:", artifacts.dashboard_png)
print("Summary:", artifacts.summary_txt)
print("Record:", artifacts.run_record_json)
