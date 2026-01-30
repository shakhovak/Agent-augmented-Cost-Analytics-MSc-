"""Declarative chart spec objects (optional)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChartType(Enum):
    """Chart types."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    HISTOGRAM = "histogram"
    DUAL_AXIS = "dual_axis"
    KPI_CARD = "kpi_card"


@dataclass
class ChartSpec:
    """Specification for a chart."""

    chart_type: ChartType
    title: str
    x_axis: str | None = None
    y_axis: list[str] | None = None
    data_source: str | None = None
    filters: dict[str, Any] | None = None
    style: dict[str, Any] | None = None
