"""Declarative chart spec objects (optional)."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


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
    x_axis: Optional[str] = None
    y_axis: Optional[List[str]] = None
    data_source: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None

