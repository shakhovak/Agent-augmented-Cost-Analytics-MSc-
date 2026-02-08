from __future__ import annotations

from dataclasses import dataclass

# Services INCLUDED in KPIs / mixes / combos (dialog is excluded by design)
SERVICE_COST_COLUMNS: tuple[str, str, str] = (
    "total_cost_tasks",
    "total_cost_classifications",
    "cost_amocrm_call",
)

SERVICE_LABELS: dict[str, str] = {
    "total_cost_tasks": "Tasks",
    "total_cost_classifications": "Classifications",
    "cost_amocrm_call": "AmoCRM Calls",
}

SERVICE_COLORS: dict[str, str] = {
    # Optional: if you want stable colors; safe to remove if you prefer matplotlib defaults
    "total_cost_tasks": "#3498db",
    "total_cost_classifications": "#27ae60",
    "cost_amocrm_call": "#9b59b6",
}


@dataclass(frozen=True)
class DashboardLayout:
    figsize: tuple[int, int] = (26, 14)
    title_fontsize: int = 18
    kpi_value_fontsize: int = 24
    kpi_title_fontsize: int = 11
    small_fontsize: int = 9
    hist_bins: int = 30
    top_accounts_n: int = 10
    trend_days: int = 7
