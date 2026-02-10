from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from cost_agent_mvp.reports.daily_report import run_standard_daily_report


def _parse_day(s: str) -> date:
    return date.fromisoformat(s.strip())


def _default_templates_path() -> str:
    """
    Resolve default templates path relative to this file, so it works regardless
    of current working directory (CLI/tests/Telegram).
    Repo layout assumed:
      <repo_root>/configs/report_templates.yaml
      <repo_root>/src/cost_agent_mvp/cli/run_daily_report.py   (this file)
    """
    here = Path(__file__).resolve()
    # .../src/cost_agent_mvp/cli/run_daily_report.py -> repo_root is parents[4]
    repo_root = here.parents[4]
    return str(repo_root / "Agent-augmented-Cost-Analytics-MSc-/configs" / "report_templates.yaml")


def main(argv: list[str] | None = None) -> int:
    # Load .env so YANDEX_* etc are available when running locally.
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run standard daily report (deterministic or LLM)."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV dataset.")
    parser.add_argument("--day", required=True, help="Report day in YYYY-MM-DD format.")
    parser.add_argument(
        "--out",
        default="outputs/runs",
        help="Output root directory (default: outputs/runs).",
    )
    parser.add_argument(
        "--trend-days", type=int, default=None, help="Override trend window length."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override top-N accounts/services shown.",
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Narrative mode (default: deterministic).",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Run LLM mode without network calls (deterministic mock response).",
    )
    parser.add_argument(
        "--prompts-dir",
        default=None,
        help="Optional prompts directory override (defaults to package prompts).",
    )

    # ✅ Required by tests/test_cli_run_daily_report.py
    parser.add_argument(
        "--templates-path",
        default=None,
        help="Path to report_templates.yaml (useful for tests / overrides).",
    )
    parser.add_argument(
        "--template-id",
        default="standard_daily_report",
        help="Template id in report_templates.yaml (default: standard_daily_report).",
    )

    args = parser.parse_args(argv)

    templates_path = args.templates_path or _default_templates_path()

    artifacts = run_standard_daily_report(
        csv_path=args.csv,
        report_day=_parse_day(args.day),
        output_root=args.out,
        report_templates_path=templates_path,
        template_id=args.template_id,
        top_n_override=args.top_n,
        trend_days_override=args.trend_days,
        narrative_mode=args.mode,
        mock_llm=args.mock_llm,
        prompts_dir=args.prompts_dir,
    )

    # Print key info (used by tests / demo logs)
    print(f"run_id: {artifacts.run_id}")
    print(f"output_dir: {artifacts.output_dir}")
    print(f"dashboard_png: {artifacts.dashboard_png}")
    print(f"summary_txt: {artifacts.summary_txt}")
    print(f"evidence_dir: {artifacts.evidence_dir}")
    print(f"run_record_json: {artifacts.run_record_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
