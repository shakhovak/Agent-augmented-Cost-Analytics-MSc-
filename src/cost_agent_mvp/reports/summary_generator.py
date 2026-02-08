"""Deterministic daily summary grounded only in Evidence Pack tables (no LLM)."""

from __future__ import annotations

import pandas as pd

EvidencePack = dict[str, pd.DataFrame]


def _fmt_money(x: object) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def _fmt_int(x: object) -> str:
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return str(x)


def _fmt_pct_from_fraction(x: object) -> str:
    """Evidence deltas are fractions (e.g., 0.031 == 3.1%)."""
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def _trend_note(trend: pd.DataFrame) -> str:
    """
    Deterministic 7-day trend note based only on trend_daily table.
    Uses first and last points in the provided window.
    """
    if trend is None or trend.empty or len(trend) < 2:
        return "7-day trend: Not available"

    # Prefer diluted cost metric if present, otherwise fall back
    col = None
    for c in ["avg_cost_per_active_account_day", "avg_cost_non_diluted"]:
        if c in trend.columns:
            col = c
            break

    if col is None:
        return "7-day trend: Not available"

    start = float(trend[col].iloc[0])
    end = float(trend[col].iloc[-1])

    direction = "flat"
    if end > start:
        direction = "up"
    elif end < start:
        direction = "down"

    return f"7-day trend ({col}): {direction} ({_fmt_money(start)} → {_fmt_money(end)})"


def generate_daily_summary(evidence: EvidencePack, top_n: int = 3) -> str:
    """
    Deterministic summary grounded only in Evidence Pack v0 output (E2).

    Required content:
      - today's total cost + delta vs previous period
      - top 3 services by cost
      - top 3 accounts by cost
      - short 7-day trend note (from trend_daily)
    """
    # Evidence Pack v0 keys (per build_standard_daily_evidence)
    kpi = evidence.get("kpis", pd.DataFrame())
    service_breakdown = evidence.get("service_breakdown", pd.DataFrame())
    top_accounts = evidence.get("top_accounts", pd.DataFrame())
    trend = evidence.get("trend_daily", pd.DataFrame())

    lines: list[str] = []
    lines.append("Daily Cost Monitoring Summary")
    lines.append("=" * 30)

    if kpi is None or kpi.empty:
        lines.append("KPI table is empty (no data for the selected day/window).")
        return "\n".join(lines)

    r = kpi.iloc[0].to_dict()
    lines.append(f"Date: {r.get('date', '')}")
    lines.append("")

    # --- Total cost + delta
    total_cost = r.get("total_cost_services", 0.0)
    total_cost_delta_pct = r.get("total_cost_services_delta_pct", 0.0)
    lines.append(
        f"Total cost (services): {_fmt_money(total_cost)} "
        f"({_fmt_pct_from_fraction(total_cost_delta_pct)} vs previous day)"
    )

    # Optional (nice to have, still deterministic): active + churn + dials
    if "active_users" in r:
        lines.append(
            f"Active users: {_fmt_int(r.get('active_users', 0))} "
            f"({_fmt_pct_from_fraction(r.get('active_users_delta_pct', 0.0))} vs previous day)"
        )
    if "new_users" in r and "churned_users" in r:
        lines.append(
            f"User churn: new={_fmt_int(r.get('new_users', 0))}, "
            f"churned={_fmt_int(r.get('churned_users', 0))}, "
            f"net={_fmt_int(r.get('net_user_change', 0))}"
        )
    if "dials_analyzed" in r:
        lines.append(
            f"Dials analyzed: {_fmt_int(r.get('dials_analyzed', 0))} "
            f"({_fmt_pct_from_fraction(r.get('dials_analyzed_delta_pct', 0.0))} vs previous day)"
        )

    # --- Top services by cost
    lines.append("")
    lines.append(f"Top {top_n} services by cost:")
    if (
        service_breakdown is not None
        and not service_breakdown.empty
        and {"label", "cost"}.issubset(service_breakdown.columns)
    ):
        sb = service_breakdown.sort_values("cost", ascending=False).head(top_n)
        for _, row in sb.iterrows():
            share = row["share_of_total"] if "share_of_total" in sb.columns else None
            if share is None:
                lines.append(f"- {row['label']}: {_fmt_money(row['cost'])}")
            else:
                lines.append(
                    f"- {row['label']}: {_fmt_money(row['cost'])} ({_fmt_pct_from_fraction(share)})"
                )
    else:
        lines.append("- Not available")

    # --- Top accounts by cost
    lines.append("")
    lines.append(f"Top {top_n} accounts by cost:")
    if (
        top_accounts is not None
        and not top_accounts.empty
        and {"account_id", "total_cost_services"}.issubset(top_accounts.columns)
    ):
        ta = top_accounts.sort_values("total_cost_services", ascending=False).head(top_n)
        for _, row in ta.iterrows():
            share = row["share_of_total_cost"] if "share_of_total_cost" in ta.columns else None
            if share is None:
                lines.append(
                    f"- account {int(row['account_id'])}: {_fmt_money(row['total_cost_services'])}"
                )
            else:
                lines.append(
                    f"- account {int(row['account_id'])}: {_fmt_money(row['total_cost_services'])} "
                    f"({_fmt_pct_from_fraction(share)} of total)"
                )
    else:
        lines.append("- Not available")

    # --- 7-day trend note (from trend_daily)
    lines.append("")
    lines.append(_trend_note(trend))

    return "\n".join(lines)
