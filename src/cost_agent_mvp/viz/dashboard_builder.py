from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cost_agent_mvp.viz.chart_specs import SERVICE_COLORS, DashboardLayout

CURRENCY = "USD"
COST_CAP_USD: float = 30.0
PRICE_PER_ACCOUNT_DAY_USD: float = 100.0

# Robust mapping: labels in data may vary (e.g., 'AmoCRM Calls', 'amocrm', 'amo crm calls')
_SERVICE_ALIASES: dict[str, str] = {
    "amocrm": "amocrm calls",
    "amocrm call": "amocrm calls",
    "amocrm calls": "amocrm calls",
    "amo crm calls": "amocrm calls",
    "amo crm": "amocrm calls",
    "tasks": "tasks",
    "task": "tasks",
    "classifications": "classifications",
    "classification": "classifications",
}

SERVICE_LABEL_TO_COL: dict[str, str] = {
    # display label -> dataframe column key
    "tasks": "total_cost_tasks",
    "task": "total_cost_tasks",
    "classifications": "total_cost_classifications",
    "classification": "total_cost_classifications",
    "amocrm calls": "cost_amocrm_call",
    "amocrm": "cost_amocrm_call",
    "amo crm calls": "cost_amocrm_call",
    "amo crm": "cost_amocrm_call",
}
DEFAULT_BAR_COLOR = "#4C78A8"


def normalize_label(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def color_for_service(label_or_col: object) -> str:
    """
    Accepts either:
      - a dataframe column name (e.g. total_cost_tasks)
      - a display label (e.g. Tasks, AmoCRM Calls)
    and returns the configured color.
    """
    raw = str(label_or_col).strip()
    # direct column name hit
    if raw in SERVICE_COLORS:
        return SERVICE_COLORS[raw]

    key = normalize_label(raw)
    col = SERVICE_LABEL_TO_COL.get(key)
    if col and col in SERVICE_COLORS:
        return SERVICE_COLORS[col]

    return DEFAULT_BAR_COLOR


def build_standard_daily_dashboard(
    evidence: dict[str, pd.DataFrame],
    out_path: str | Path,
    *,
    title: str | None = None,
    layout: DashboardLayout | None = None,
) -> Path:
    """
    Builds the dashboard matching tiles + plots A–F.
    Expects evidence pack keys produced by build_standard_daily_evidence().
    """
    layout = layout or DashboardLayout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kpis = evidence["kpis"].iloc[0].to_dict()
    trend = evidence["trend_daily"].copy()
    service_breakdown = evidence["service_breakdown"].copy()
    service_totals = evidence["service_totals"].copy()
    service_combos = evidence["service_combinations"].copy()
    dials_cost_trend = evidence["dials_cost_trend"].copy()

    dist_hist = evidence["distribution_hist"].copy()
    dist_stats = evidence["distribution_stats"].iloc[0].to_dict()

    report_day = kpis["date"]

    fig = plt.figure(figsize=layout.figsize)
    fig.subplots_adjust(
        left=0.03,  # default ~0.125
        right=0.99,  # default ~0.9
        top=0.94,  # default ~0.88 (leave room for title)
        bottom=0.04,  # default ~0.11
        wspace=0.35,  # spacing between columns
        hspace=0.55,  # spacing between rows
    )
    gs = fig.add_gridspec(
        3,
        8,
        hspace=0.4,
        wspace=0.35,
        height_ratios=[0.72, 1.05, 1.05],
    )

    def kpi_card(
        ax,
        title_txt,
        value_txt,
        *,
        color="#3498db",
        change=None,
        change_pct=None,
        subtitle="",
    ):
        ax.axis("off")
        rect = plt.Rectangle(
            (0.05, 0.05),
            0.9,
            0.9,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#e0e0e0",
            linewidth=2,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            0.85,
            title_txt,
            ha="center",
            va="center",
            fontsize=layout.kpi_title_fontsize,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.55,
            value_txt,
            ha="center",
            va="center",
            fontsize=layout.kpi_value_fontsize,
            fontweight="bold",
            color=color,
        )

        if change is not None:
            change_color = "#27ae60" if change >= 0 else "#e74c3c"
            change_symbol = "+" if change >= 0 else ""
            txt = f"{change_symbol}{change:,.0f}"
            if change_pct is not None:
                txt += f" ({change_symbol}{change_pct * 100:.1f}%)"
            ax.text(
                0.5,
                0.25,
                f"vs yesterday: {txt}",
                ha="center",
                va="center",
                fontsize=layout.small_fontsize,
                color=change_color,
                fontweight="bold",
            )

        if subtitle:
            ax.text(
                0.5,
                0.1,
                subtitle,
                ha="center",
                va="center",
                fontsize=layout.small_fontsize,
                color="#7f8c8d",
            )

    def churn_card(ax, new_users: float, churned_users: float, net: float):
        ax.axis("off")
        rect = plt.Rectangle(
            (0.05, 0.05),
            0.9,
            0.9,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#e0e0e0",
            linewidth=2,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            0.90,
            "User Churn",
            ha="center",
            va="center",
            fontsize=layout.kpi_title_fontsize,
            fontweight="bold",
        )
        ax.text(0.3, 0.65, "New:", ha="center", va="center", fontsize=10)
        ax.text(
            0.3,
            0.50,
            f"{int(new_users):,}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#27ae60",
        )
        ax.text(0.7, 0.65, "Churned:", ha="center", va="center", fontsize=10)
        ax.text(
            0.7,
            0.50,
            f"{int(churned_users):,}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#e74c3c",
        )

        net_color = "#27ae60" if net >= 0 else "#e74c3c"
        net_symbol = "+" if net >= 0 else ""
        ax.text(
            0.5,
            0.25,
            f"Net: {net_symbol}{int(net):,}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=net_color,
        )
        ax.text(
            0.5,
            0.10,
            "vs yesterday",
            ha="center",
            va="center",
            fontsize=layout.small_fontsize,
            color="#7f8c8d",
        )

    # --- Row 1: KPI cards
    ax1 = fig.add_subplot(gs[0, 0])  # Active
    kpi_card(
        ax1,
        "Total Active\nUsers",
        f"{int(kpis['active_users']):,}",
        color="#3498db",
        change=kpis["active_users_delta_abs"],
        change_pct=kpis["active_users_delta_pct"],
    )

    ax2 = fig.add_subplot(gs[0, 1])  # Active
    churn_card(ax2, kpis["new_users"], kpis["churned_users"], kpis["net_user_change"])

    ax3 = fig.add_subplot(gs[0, 2])
    kpi_card(
        ax3,
        "Avg Cost per\nActive Account-Day",
        f"{kpis['avg_cost_per_active_account_day']:,.2f} {CURRENCY}",
        color="#e74c3c",
        change=kpis.get("avg_cost_per_active_account_day_delta_abs"),
        change_pct=kpis.get("avg_cost_per_active_account_day_delta_pct"),
        subtitle="Diluted:\ntotal_cost / active_accounts",
    )

    ax4 = fig.add_subplot(gs[0, 3])
    kpi_card(
        ax4,
        "p95 Cost per\nAccount-Day",
        f"{kpis['p95_cost_per_account_day']:,.2f} {CURRENCY}",
        color="#8e44ad",
        change=kpis.get("p95_cost_per_account_day_delta_abs"),
        change_pct=kpis.get("p95_cost_per_account_day_delta_pct"),
        subtitle=f"Tail risk | cap={COST_CAP_USD:,.0f} {CURRENCY}",
    )

    ax5 = fig.add_subplot(gs[0, 4])
    kpi_card(
        ax5,
        "Total Daily Costs",
        f"{kpis['total_cost_services']:,.0f} {CURRENCY}",
        color="#27ae60",
        change=kpis.get("total_cost_services_delta_abs"),
        change_pct=kpis.get("total_cost_services_delta_pct"),
        subtitle="Tasks + Classif + AmoCRM",
    )
    ax6k = fig.add_subplot(gs[0, 5])  # % above cap
    # % accounts above cap
    kpi_card(
        ax6k,
        f"% Accounts >\n{COST_CAP_USD:,.0f} {CURRENCY}",
        f"{kpis['pct_accounts_above_cap'] * 100:,.1f}%",
        color="#e67e22",
        change=kpis.get("pct_accounts_above_cap_delta_abs") * 100
        if kpis.get("pct_accounts_above_cap_delta_abs") is not None
        else None,
        change_pct=kpis.get("pct_accounts_above_cap_delta_pct"),
        subtitle="Tail prevalence",
    )

    ax7k = fig.add_subplot(gs[0, 6])  # Avg margin
    # Avg margin per account-day (price - avg cost)
    kpi_card(
        ax7k,
        "Avg Margin per\nAccount-Day",
        f"{kpis['avg_margin_per_account_day']:,.2f} {CURRENCY}",
        color="#16a085",
        change=kpis.get("avg_margin_per_account_day_delta_abs"),
        change_pct=kpis.get("avg_margin_per_account_day_delta_pct"),
        subtitle=f"Price={PRICE_PER_ACCOUNT_DAY_USD:,.0f} {CURRENCY}",
    )
    ax8k = fig.add_subplot(gs[0, 7])  # Dials analyzed
    # Dials analyzed (today)
    kpi_card(
        ax8k,
        "Dials Analyzed",
        f"{int(kpis['dials_analyzed']):,}",
        color="#2980b9",
        change=kpis.get("dials_analyzed_delta_abs"),
        change_pct=kpis.get("dials_analyzed_delta_pct"),
        subtitle="Non-zero service lines",
    )
    # --- Row 2: A) Distribution hist + median, B) Mix pie, C) Avg cost trend + 7d avg
    ax6 = fig.add_subplot(gs[1, :3])
    if not dist_hist.empty and dist_stats.get("n", 0) > 0:
        # Render hist from pre-binned table so it's stable/reproducible
        centers = (dist_hist["bin_left"] + dist_hist["bin_right"]) / 2
        widths = dist_hist["bin_right"] - dist_hist["bin_left"]
        ax6.bar(
            centers,
            dist_hist["count"],
            width=widths,
            align="center",
            alpha=0.7,
            edgecolor="black",
        )
        median = float(dist_stats["median"])
        ax6.axvline(
            median,
            linestyle="--",
            linewidth=2,
            color="red",
            label=f"Median: {median:,.2f} {CURRENCY}",
        )
        ax6.axvline(
            COST_CAP_USD,
            linestyle=":",
            linewidth=2,
            color="#555555",
            label=f"Cap: {COST_CAP_USD:,.0f} {CURRENCY}",
        )
        ax6.legend()
    ax6.set_title("Distribution of Cost per Account-Day\n(Today)", fontweight="bold")
    ax6.set_xlabel(f"Cost per Account-Day ({CURRENCY})")
    ax6.set_ylabel("Number of Accounts")
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[1, 3])
    if not service_breakdown.empty and service_breakdown["cost"].sum() > 0:
        sizes = service_breakdown["cost"].values
        labels = service_breakdown["label"].values
        colors = [color_for_service(c) for c in service_breakdown["component"].values]
        ax7.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
    else:
        ax7.text(0.5, 0.5, "No costs", ha="center", va="center")
    ax7.set_title("Cost Distribution\nby Service", fontweight="bold")

    ax8 = fig.add_subplot(gs[1, 4:])
    if not trend.empty:
        x = np.arange(len(trend))
        ax8.plot(
            x,
            trend["avg_cost_per_active_account_day"],
            marker="o",
            linewidth=2,
            label="Daily Average",
        )
        avg7 = float(trend["avg_cost_per_active_account_day"].mean()) if len(trend) else 0.0
        ax8.axhline(
            avg7,
            linestyle="--",
            linewidth=2,
            color="red",
            label=f"7-Day Avg: {avg7:,.4f}",
        )
        ax8.set_xticks(x)
        ax8.set_xticklabels(
            [pd.to_datetime(d).strftime("%m/%d") for d in trend["date"]], fontsize=9
        )
        ax8.set_title("Cost Trend\n(Last 7 Days)", fontweight="bold")
        ax8.set_ylabel(f"Cost ({CURRENCY})")
        ax8.grid(True, alpha=0.3)
        ax8.legend()
    else:
        ax8.text(0.5, 0.5, "No data", ha="center", va="center")

    # --- Row 3: D) service combos, E) service totals, F) dials & cost trends
    ax9 = fig.add_subplot(gs[2, :3])
    if not service_combos.empty:
        labels = service_combos["label"].tolist()
        values = service_combos["active_users"].tolist()
        bars = ax9.bar(labels, values, alpha=0.8, edgecolor="black", linewidth=1.2)
        ax9.set_title("Active Users by Service Combination", fontweight="bold")
        ax9.set_ylabel("Active Users")
        ax9.grid(True, axis="y", alpha=0.3)
        ax9.tick_params(axis="x", labelrotation=0)
        for bar, v in zip(bars, values, strict=True):
            ax9.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    else:
        ax9.text(0.5, 0.5, "No data", ha="center", va="center")

    ax10 = fig.add_subplot(gs[2, 3])
    if not service_totals.empty:
        labels = service_totals["label"].tolist()
        values = service_totals["cost"].tolist()
        bars = ax10.bar(labels, values, alpha=0.8, edgecolor="black", linewidth=1.2)
        ax10.set_title("Service Usage Breakdown\n(Total Costs)", fontweight="bold")
        ax10.set_ylabel(f"Total Cost ({CURRENCY})")
        ax10.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values, strict=True):
            ax10.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:,.0f} {CURRENCY}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax10.text(0.5, 0.5, "No data", ha="center", va="center")

    ax11 = fig.add_subplot(gs[2, 4:])
    if not dials_cost_trend.empty:
        x = np.arange(len(dials_cost_trend))
        ax11.plot(
            x,
            dials_cost_trend["dials_analyzed"],
            marker="o",
            linewidth=2,
            label="Dials Analyzed",
            color="#1f77b4",
        )
        ax11.set_xticks(x)
        ax11.set_xticklabels(
            [pd.to_datetime(d).strftime("%m/%d") for d in dials_cost_trend["date"]],
            fontsize=9,
        )
        ax11.set_ylabel("Dials")
        ax11.grid(True, alpha=0.3)

        ax11_t = ax11.twinx()
        ax11_t.plot(
            x,
            dials_cost_trend["total_cost_services"],
            marker="s",
            linewidth=2,
            label=f"Total Cost ({CURRENCY})",
            color="#2ca02c",
        )
        ax11_t.set_ylabel(f"Total Cost ({CURRENCY})")

        lines1, labels1 = ax11.get_legend_handles_labels()
        lines2, labels2 = ax11_t.get_legend_handles_labels()
        ax11.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
        ax11.set_title("Dials & Cost Trends\n(Last 7 Days)", fontweight="bold")
    else:
        ax11.text(0.5, 0.5, "No data", ha="center", va="center")

    fig.suptitle(
        title or f"Daily Dashboard - {report_day}",
        fontsize=layout.title_fontsize,
        fontweight="bold",
        y=0.995,
    )
    # plt.tight_layout(rect=[0, 0, 1, 0.99])

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
