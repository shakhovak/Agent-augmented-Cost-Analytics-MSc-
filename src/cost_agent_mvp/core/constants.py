"""Constants and enums for dimensions, metrics, and chart types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class SourceType(str, Enum):
    CSV = "csv"
    POSTGRES = "postgres"


class TimeWindowType(str, Enum):
    YESTERDAY = "yesterday"
    LAST_N_DAYS = "last_n_days"
    RANGE = "range"


class Aggregation(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class ChartType(str, Enum):
    KPI_CARDS = "kpi_cards"
    LINE_TREND = "line_trend"
    BAR_TOP_N = "bar_top_n"
    PIE_COMPOSITION = "pie_composition"
    HIST_DISTRIBUTION = "hist_distribution"
    TABLE = "table"


# Default operational constraints (can be overridden by semantic_layer.yaml)
DEFAULT_MAX_ROWS: int = 2000
HARD_MAX_ROWS: int = 20000

DEFAULT_MAX_DAYS: int = 30
HARD_MAX_DAYS: int = 180

DEFAULT_TOP_N: int = 10

# Drilldown safety defaults
DRILLDOWN_MAX_DAYS: int = 7
DRILLDOWN_REQUIRED_FILTERS: FrozenSet[str] = frozenset({"account_id"})


@dataclass(frozen=True)
class SafetyLimits:
    """Runtime safety caps. Typically loaded/overridden from config."""

    max_rows_default: int = DEFAULT_MAX_ROWS
    max_rows_hard: int = HARD_MAX_ROWS
    max_days_default: int = DEFAULT_MAX_DAYS
    max_days_hard: int = HARD_MAX_DAYS
    default_top_n: int = DEFAULT_TOP_N
    drilldown_max_days: int = DRILLDOWN_MAX_DAYS
