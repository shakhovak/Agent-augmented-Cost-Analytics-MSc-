"""Create PNG dashboard (can reuse existing plotting code)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


EvidencePack = Dict[str, pd.DataFrame]


@dataclass(frozen=True)
class DashboardBuildResult:
    png_path: str


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


def build_standard_daily_dashboard(
    evidence: EvidencePack,
    out_path: str,
    title: str = "Standard Daily Report",
) -> DashboardBuildResult:
    """
    Build a single PNG dashboard from the standard daily evidence pack.

    Expects evidence tables:
      - kpi_today_vs_yesterday
      - service_cost_breakdown
      - top_accounts_by_total_cost
      - trend_7d
      - distribution_avg_cost_per_account
      - exceptions_queue (optional; may be empty)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    kpi = evidence.get("kpi_today_vs_yesterday", pd.DataFrame())
    svc = evidence.get("service_cost_breakdown", pd.DataFrame())
    top = evidence.get("top_accounts_by_total_cost", pd.DataFrame())
    trend = evidence.get("trend_7d", pd.DataFrame())
    hist = evidence.get("distribution_avg_cost_per_account", pd.DataFrame())
    exc = evidence.get("exceptions_queue", pd.DataFrame())

    # Create a clean multi-panel layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # Grid: 2 rows x 3 cols
    ax_kpi = fig.add_subplot(2, 3, 1)
    ax_pie = fig.add_subplot(2, 3, 2)
    ax_top = fig.add_subplot(2, 3, 3)
    ax_trend_cost = fig.add_subplot(2, 3, 4)
    ax_trend_usage = fig.add_subplot(2, 3, 5)
    ax_hist = fig.add_subplot(2, 3, 6)

    # --- KPI cards (as text block) ---
    ax_kpi.axis("off")
    if not kpi.empty:
        row = kpi.iloc[0].to_dict()

        lines = [
            f"Date: {row.get('date', '')}",
            "",
            f"Total cost: {_fmt_money(row.get('total_cost', 0.0))} "
            f"({_fmt_pct(row.get('total_cost_delta_pct', 0.0))} vs prev)",
            f"Active users: {int(row.get('active_users', 0))} "
            f"({_fmt_pct(row.get('active_users_delta_pct', 0.0))} vs prev)",
            f"Dials analyzed: {int(row.get('dials_analyzed', 0))} "
            f"({_fmt_pct(row.get('dials_analyzed_delta_pct', 0.0))} vs prev)",
            "",
            f"Avg cost / account (diluted): {_fmt_money(row.get('avg_cost_per_account', 0.0))} "
            f"({_fmt_pct(row.get('avg_cost_per_account_delta_pct', 0.0))} vs prev)",
            f"Avg cost / account (non-diluted): {_fmt_money(row.get('avg_cost_per_account_non_diluted', 0.0))} "
            f"({_fmt_pct(row.get('avg_cost_per_account_non_diluted_delta_pct', 0.0))} vs prev)",
        ]
        ax_kpi.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=11)
        ax_kpi.set_title("KPI Overview", fontsize=12)
    else:
        ax_kpi.set_title("KPI Overview (no data)", fontsize=12)

    # --- Composition pie ---
    if not svc.empty and "component" in svc.columns and "cost" in svc.columns:
        # Keep only positive values
        svc2 = svc[svc["cost"].fillna(0) > 0].copy()
        if not svc2.empty:
            ax_pie.pie(
                svc2["cost"].values,
                labels=svc2["component"].astype(str).values,
                autopct="%1.1f%%",
            )
            ax_pie.set_title("Cost Composition (yesterday)", fontsize=12)
        else:
            ax_pie.axis("off")
            ax_pie.set_title("Cost Composition (no positive cost)", fontsize=12)
    else:
        ax_pie.axis("off")
        ax_pie.set_title("Cost Composition (no data)", fontsize=12)

    # --- Top accounts bar ---
    if not top.empty and "account_id" in top.columns and "total_cost" in top.columns:
        top2 = top.copy()
        # Use account_id as string labels
        ax_top.bar(top2["account_id"].astype(str).values, top2["total_cost"].values)
        ax_top.set_title("Top Accounts by Total Cost", fontsize=12)
        ax_top.set_xlabel("account_id")
        ax_top.set_ylabel("total_cost")
        ax_top.tick_params(axis="x", labelrotation=45)
    else:
        ax_top.axis("off")
        ax_top.set_title("Top Accounts (no data)", fontsize=12)

    # --- Trend: total cost ---
    if not trend.empty and "date" in trend.columns and "total_cost" in trend.columns:
        t = trend.sort_values("date")
        ax_trend_cost.plot(t["date"], t["total_cost"])
        ax_trend_cost.set_title("Total Cost Trend (7 days)", fontsize=12)
        ax_trend_cost.set_xlabel("date")
        ax_trend_cost.set_ylabel("total_cost")
        ax_trend_cost.tick_params(axis="x", labelrotation=45)
    else:
        ax_trend_cost.axis("off")
        ax_trend_cost.set_title("Trend (no data)", fontsize=12)

    # --- Trend: usage proxies ---
    if not trend.empty and "date" in trend.columns:
        t = trend.sort_values("date")
        have_any = False
        if "active_users" in t.columns:
            ax_trend_usage.plot(t["date"], t["active_users"], label="active_users")
            have_any = True
        if "dials_analyzed" in t.columns:
            ax_trend_usage.plot(t["date"], t["dials_analyzed"], label="dials_analyzed")
            have_any = True

        if have_any:
            ax_trend_usage.set_title("Usage Proxies (7 days)", fontsize=12)
            ax_trend_usage.set_xlabel("date")
            ax_trend_usage.tick_params(axis="x", labelrotation=45)
            ax_trend_usage.legend()
        else:
            ax_trend_usage.axis("off")
            ax_trend_usage.set_title("Usage Proxies (not available)", fontsize=12)
    else:
        ax_trend_usage.axis("off")
        ax_trend_usage.set_title("Usage Proxies (no data)", fontsize=12)

    # --- Histogram (from histogram table bins) ---
    # histogram_avg_cost_per_account table has bin_left/bin_right/count
    if not hist.empty and {"bin_left", "bin_right", "count"}.issubset(hist.columns):
        h = hist.copy()
        # bar centers and widths
        centers = (h["bin_left"].astype(float) + h["bin_right"].astype(float)) / 2.0
        widths = h["bin_right"].astype(float) - h["bin_left"].astype(float)
        ax_hist.bar(
            centers.values,
            h["count"].astype(int).values,
            width=widths.values,
            align="center",
        )
        ax_hist.set_title(
            "Distribution: Total Cost per Account (yesterday)", fontsize=12
        )
        ax_hist.set_xlabel("total_cost_per_account (bins)")
        ax_hist.set_ylabel("count")
    else:
        ax_hist.axis("off")
        ax_hist.set_title("Distribution (no data)", fontsize=12)

    # Add exceptions info as a small footer text (optional)
    if exc is not None and isinstance(exc, pd.DataFrame) and not exc.empty:
        fig.text(0.01, 0.01, f"Exceptions flagged: {len(exc)}", fontsize=10)
    else:
        fig.text(0.01, 0.01, "Exceptions flagged: 0", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return DashboardBuildResult(png_path=str(Path(out_path).resolve()))
