"""Generate text summary (Stage 1: deterministic or template-based, no LLM yet)."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


EvidencePack = Dict[str, pd.DataFrame]


def _fmt_money(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def generate_daily_summary(evidence: EvidencePack, top_n: int = 5) -> str:
    """
    Deterministic summary for the Standard Daily Report.
    Uses only the evidence pack tables (no LLM).

    Expected tables:
      - kpi_today_vs_yesterday (1 row)
      - service_cost_breakdown
      - top_accounts_by_total_cost
      - top_accounts_by_delta
      - distribution_stats (optional)
      - exceptions_queue (optional)
    """
    kpi = evidence.get("kpi_today_vs_yesterday", pd.DataFrame())
    svc = evidence.get("service_cost_breakdown", pd.DataFrame())
    top_cost = evidence.get("top_accounts_by_total_cost", pd.DataFrame())
    top_delta = evidence.get("top_accounts_by_delta", pd.DataFrame())
    dist_stats = evidence.get("distribution_stats", pd.DataFrame())
    exc = evidence.get("exceptions_queue", pd.DataFrame())

    lines: List[str] = []
    lines.append("Daily Cost Monitoring Summary")
    lines.append("=" * 30)

    if not kpi.empty:
        r = kpi.iloc[0].to_dict()
        lines.append(f"Date: {r.get('date', '')}")
        lines.append("")
        lines.append(
            f"Total cost: {_fmt_money(r.get('total_cost', 0.0))} "
            f"({_fmt_pct(r.get('total_cost_delta_pct', 0.0))} vs previous day)"
        )
        lines.append(
            f"Active users: {int(r.get('active_users', 0))} "
            f"({_fmt_pct(r.get('active_users_delta_pct', 0.0))} vs previous day)"
        )
        lines.append(
            f"Dials analyzed: {int(r.get('dials_analyzed', 0))} "
            f"({_fmt_pct(r.get('dials_analyzed_delta_pct', 0.0))} vs previous day)"
        )
        lines.append(
            f"Avg cost per account (diluted): {_fmt_money(r.get('avg_cost_per_account', 0.0))} "
            f"({_fmt_pct(r.get('avg_cost_per_account_delta_pct', 0.0))} vs previous day)"
        )
        lines.append(
            f"Avg cost per account (non-diluted): {_fmt_money(r.get('avg_cost_per_account_non_diluted', 0.0))} "
            f"({_fmt_pct(r.get('avg_cost_per_account_non_diluted_delta_pct', 0.0))} vs previous day)"
        )
    else:
        lines.append("KPI table is empty (no data for the selected day/window).")

    lines.append("")
    lines.append("Cost composition (top components):")
    if not svc.empty and {"component", "cost", "share_of_total"}.issubset(svc.columns):
        svc2 = svc.sort_values("cost", ascending=False).head(top_n)
        for _, row in svc2.iterrows():
            lines.append(
                f"- {row['component']}: {_fmt_money(row['cost'])} ({_fmt_pct(row['share_of_total'])})"
            )
    else:
        lines.append("- Not available")

    lines.append("")
    lines.append(f"Top {top_n} accounts by total cost:")
    if not top_cost.empty and {"account_id", "total_cost", "share_of_total"}.issubset(
        top_cost.columns
    ):
        tc = top_cost.head(top_n)
        for _, row in tc.iterrows():
            lines.append(
                f"- account {int(row['account_id'])}: {_fmt_money(row['total_cost'])} "
                f"({ _fmt_pct(row['share_of_total']) } of total)"
            )
    else:
        lines.append("- Not available")

    lines.append("")
    lines.append(f"Top {top_n} accounts by day-over-day increase:")
    if not top_delta.empty and {"account_id", "delta_abs", "delta_pct"}.issubset(
        top_delta.columns
    ):
        td = top_delta.head(top_n)
        for _, row in td.iterrows():
            lines.append(
                f"- account {int(row['account_id'])}: +{_fmt_money(row['delta_abs'])} "
                f"({_fmt_pct(row['delta_pct'])} vs previous day)"
            )
    else:
        lines.append("- Not available")

    if not dist_stats.empty:
        lines.append("")
        lines.append("Per-account total cost distribution (yesterday):")
        s = dist_stats.iloc[0].to_dict()
        lines.append(
            f"- mean: {_fmt_money(s.get('mean', 0.0))}, median: {_fmt_money(s.get('median', 0.0))}, "
            f"p90: {_fmt_money(s.get('p90', 0.0))}, max: {_fmt_money(s.get('max', 0.0))}"
        )

    lines.append("")
    exc_count = int(len(exc)) if isinstance(exc, pd.DataFrame) else 0
    lines.append(f"Exceptions flagged: {exc_count}")
    if exc_count > 0 and {"rule_id", "severity"}.issubset(exc.columns):
        preview = exc.head(min(5, exc_count))
        for _, row in preview.iterrows():
            lines.append(f"- {row.get('severity', '')}: {row.get('rule_id', '')}")

    lines.append("")
    lines.append("Recommended checks (generic):")
    lines.append(
        "- Review the top accounts and confirm whether the increase aligns with expected usage patterns."
    )
    lines.append(
        "- If cost increased without usage growth, check routing/model changes or unusually expensive components."
    )
    lines.append(
        "- If the service mix shifted (tasks vs classifications), validate upstream workflow volume and configuration."
    )

    return "\n".join(lines)
