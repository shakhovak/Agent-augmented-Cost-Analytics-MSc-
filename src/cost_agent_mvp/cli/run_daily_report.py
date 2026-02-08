from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from cost_agent_mvp.reports.daily_report import run_standard_daily_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the standard daily report (deterministic baseline)."
    )
    p.add_argument("--csv", required=True, help="Path to CSV dataset")
    p.add_argument("--day", required=True, help="Report day in YYYY-MM-DD format")

    p.add_argument(
        "--out",
        default="outputs/runs",
        help="Output root directory (default: outputs/runs)",
    )
    p.add_argument(
        "--templates-path",
        default="configs/report_templates.yaml",
        help="Path to report_templates.yaml",
    )
    p.add_argument(
        "--template",
        default="standard_daily_report",
        help="Template id (default: standard_daily_report)",
    )
    p.add_argument("--dataset-name", default="joint_costs_daily", help="Dataset name label")

    # Optional overrides (milestone wants these)
    p.add_argument("--top-n", type=int, default=None, help="Override template constraints.top_n")
    p.add_argument("--trend-days", type=int, default=None, help="Override trend window in days")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        csv_path = str(Path(args.csv))
        report_day = date.fromisoformat(args.day)

        artifacts = run_standard_daily_report(
            csv_path=csv_path,
            report_day=report_day,
            output_root=args.out,
            dataset_name=args.dataset_name,
            report_templates_path=args.templates_path,
            template_id=args.template,
            # optional overrides (requires small change in orchestrator; see below)
            top_n_override=args.top_n,
            trend_days_override=args.trend_days,
        )

        # Print “run evidence”
        print(f"run_id: {artifacts.run_id}")
        print(f"output_dir: {artifacts.output_dir}")
        print(f"dashboard_png: {artifacts.dashboard_png}")
        print(f"summary_txt: {artifacts.summary_txt}")
        print(f"run_record_json: {artifacts.run_record_json}")
        print(f"evidence_dir: {artifacts.evidence_dir}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
