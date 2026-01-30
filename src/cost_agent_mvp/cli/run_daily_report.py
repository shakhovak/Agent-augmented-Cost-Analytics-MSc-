"""CLI entrypoint: produce outputs to /outputs/."""

# src/cli/run_daily_report.py
from __future__ import annotations

import argparse
import sys
from datetime import date as date_type

from src.core.utils_dates import compute_yesterday, parse_date
from src.reports.daily_report import run_standard_daily_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_daily_report",
        description="Generate the Standard Daily Cost Monitoring Report from a CSV snapshot.",
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to the joint CSV file (snapshot).",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Report day in YYYY-MM-DD. If omitted, uses yesterday (UTC).",
    )
    p.add_argument(
        "--out",
        default="outputs/runs",
        help="Output root directory for run artifacts. Default: outputs/runs",
    )
    p.add_argument(
        "--template",
        default="standard_daily_report",
        help="Template id from configs/report_templates.yaml. Default: standard_daily_report",
    )
    p.add_argument(
        "--templates",
        default="configs/report_templates.yaml",
        help="Path to report_templates.yaml. Default: configs/report_templates.yaml",
    )
    p.add_argument(
        "--dataset-name",
        default="joint_costs_daily",
        help="Dataset name for run metadata. Default: joint_costs_daily",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.date:
        report_day: date_type = parse_date(args.date)
    else:
        report_day, _ = compute_yesterday()  # returns (yesterday, yesterday)

    artifacts = run_standard_daily_report(
        csv_path=args.csv,
        report_day=report_day,
        output_root=args.out,
        dataset_name=args.dataset_name,
        report_templates_path=args.templates,
        template_id=args.template,
    )

    print("Report generated successfully.")
    print(f"Run ID: {artifacts.run_id}")
    print(f"Output dir: {artifacts.output_dir}")
    if artifacts.dashboard_png:
        print(f"Dashboard PNG: {artifacts.dashboard_png}")
    if artifacts.summary_txt:
        print(f"Summary: {artifacts.summary_txt}")
    if artifacts.evidence_dir:
        print(f"Evidence tables: {artifacts.evidence_dir}")
    if artifacts.run_record_json:
        print(f"Run record: {artifacts.run_record_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
