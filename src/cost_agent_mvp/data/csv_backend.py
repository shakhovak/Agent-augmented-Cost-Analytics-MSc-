"""CSV backend for loading and querying data."""

from __future__ import annotations

from datetime import date as dt_date  # Add at top of file if missing
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

from cost_agent_mvp.core.constants import (
    Aggregation,
    SafetyLimits,
    SortDirection,
    SourceType,
)
from cost_agent_mvp.core.errors import (
    RowLimitExceeded,
    UnsupportedQuery,
    ValidationError,
)
from cost_agent_mvp.core.types import (
    AggregationSpec,
    Lineage,
    QueryFilters,
    QuerySpec,
    TimeWindow,
)
from cost_agent_mvp.core.utils_dates import normalize_time_window
from cost_agent_mvp.core.utils_hash import sha256_file
from cost_agent_mvp.data.schema import DatasetSchema, default_joint_schema, load_csv

_AGG_MAP = {
    Aggregation.SUM: "sum",
    Aggregation.MEAN: "mean",
    Aggregation.MEDIAN: "median",
    Aggregation.COUNT: "count",
    # COUNT_DISTINCT is handled separately
}


class CsvBackend:
    """
    Safe querying over a cached DataFrame loaded from CSV.

    Design goals:
    - predictable runtime (caps)
    - no arbitrary code execution
    - reproducible lineage
    """

    def __init__(
        self,
        csv_path: str,
        dataset_name: str = "joint_costs_daily",
        schema: DatasetSchema | None = None,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.schema = schema or default_joint_schema()
        self.limits = limits or SafetyLimits()

        # Load once and cache
        self._df: pd.DataFrame = load_csv(csv_path, schema=self.schema)
        self._dataset_version: str = sha256_file(csv_path)

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    # ---------------------------
    # Public API
    # ---------------------------

    def query(self, spec: QuerySpec) -> Any:  # ← Change return type to Any
        """
        Execute a QuerySpec safely and return object with .df and .lineage attributes.
        """
        tw = normalize_time_window(spec.time_window)
        self._enforce_time_window(tw, spec)
        self._enforce_supported_fields(spec)

        df0 = self._apply_time_window(self._df, tw)
        df1 = self._apply_filters(df0, spec.filters)

        if spec.group_by:
            df = self._group_and_aggregate(df1, spec.group_by, spec.aggregations)
        else:
            # No group_by: still allow aggregations (returns single-row table)
            df = self._aggregate_no_group(df1, spec.aggregations)

        df = self._apply_sort_topn_and_caps(df, spec)

        lineage = self._make_lineage(
            tw=tw,
            spec=spec,
            row_count=len(df),
        )

        # ✅ CRITICAL FIX: Return object with .df + lineage AS LIST
        # Convert Lineage object → list of dicts for orchestrator compatibility
        lineage_list = (
            [lineage.dict()]
            if hasattr(lineage, "dict")
            else (
                [lineage.model_dump()]
                if hasattr(lineage, "model_dump")
                else (
                    [vars(lineage)]
                    if hasattr(lineage, "__dict__")
                    else [lineage]
                    if isinstance(lineage, dict)
                    else []
                )
            )
        )

        return SimpleNamespace(df=df, lineage=lineage_list)

    # ---------------------------
    # Safety + validation
    # ---------------------------

    def _enforce_supported_fields(self, spec: QuerySpec) -> None:
        cols = set(self._df.columns)

        for g in spec.group_by:
            if g not in cols:
                raise UnsupportedQuery(f"Unsupported group_by field: '{g}'")

        for a in spec.aggregations:
            if a.field not in cols:
                raise UnsupportedQuery(f"Unsupported aggregation field: '{a.field}'")

        if (
            spec.sort is not None
            and spec.sort.by not in cols
            and spec.sort.by not in self._derived_names(spec)
        ):
            # sort.by could be an alias produced by aggregations; handle that later in sort stage
            pass

    # src/cost_agent_mvp/data/csv_backend.py

    def _enforce_time_window(self, tw: TimeWindow, spec: QuerySpec) -> None:
        # ✅ CRITICAL FIX: Parse string dates to date objects BEFORE arithmetic
        start_val = tw.start
        end_val = tw.end

        # Convert strings to date objects if needed
        if isinstance(start_val, str):
            start_date = dt_date.fromisoformat(start_val)
        else:
            start_date = start_val  # Already a date object (or None)

        if isinstance(end_val, str):
            end_date = dt_date.fromisoformat(end_val)
        else:
            end_date = end_val  # Already a date object (or None)

        # Now safe to do date arithmetic (only if both dates exist)
        if start_date and end_date:
            days = (end_date - start_date).days + 1
        else:
            days = 1  # Fallback for non-range types (yesterday, last_n_days)

        # Get max_days limit from semantic layer (passed during CsvBackend init)
        # Common patterns - try these in order:
        max_days = 14  # Safe default

        # Pattern 1: backend has .semantic_layer attribute
        if hasattr(self, "semantic_layer") and self.semantic_layer:
            max_days = self.semantic_layer.get("limits", {}).get("max_days_hard", 14)

        # Pattern 2: backend has ._semantic_layer attribute
        elif hasattr(self, "_semantic_layer") and self._semantic_layer:
            max_days = self._semantic_layer.get("limits", {}).get("max_days_hard", 14)

        # Pattern 3: spec has limits (less common)
        elif hasattr(spec, "limits"):
            max_days = spec.limits.get("max_days_hard", 14)

        # Enforce limit
        if days > max_days:
            from cost_agent_mvp.agent.plan_compiler import TimeWindowExceeded

            raise TimeWindowExceeded(days, max_days)

    # ---------------------------
    # Query execution steps
    # ---------------------------

    def _apply_time_window(self, df: pd.DataFrame, tw: TimeWindow) -> pd.DataFrame:
        # ✅ CRITICAL FIX: Convert string boundaries → date objects BEFORE comparison
        start_boundary = dt_date.fromisoformat(tw.start) if isinstance(tw.start, str) else tw.start
        end_boundary = dt_date.fromisoformat(tw.end) if isinstance(tw.end, str) else tw.end

        assert start_boundary is not None and end_boundary is not None

        # 'date' is already datetime.date (from schema coercion)
        mask = (df[self.schema.date_column] >= start_boundary) & (
            df[self.schema.date_column] <= end_boundary
        )
        return df.loc[mask]

    def _apply_filters(self, df: pd.DataFrame, flt: QueryFilters) -> pd.DataFrame:
        out = df

        if flt.account_id:
            out = out[out["account_id"].isin(flt.account_id)]

        if flt.chat_type:
            out = out[out["chat_type"].isin(flt.chat_type)]

        if flt.chat_id:
            out = out[out["chat_id"].isin(flt.chat_id)]

        # Flags: stored as Int64 nullable; treat True as ==1, False as ==0
        if flt.has_tasks is not None and "has_tasks" in out.columns:
            out = out[out["has_tasks"].fillna(0).astype(int) == (1 if flt.has_tasks else 0)]

        if flt.has_classifications is not None and "has_classifications" in out.columns:
            out = out[
                out["has_classifications"].fillna(0).astype(int)
                == (1 if flt.has_classifications else 0)
            ]

        if flt.has_both is not None and "has_both" in out.columns:
            out = out[out["has_both"].fillna(0).astype(int) == (1 if flt.has_both else 0)]

        return out

    def _group_and_aggregate(
        self,
        df: pd.DataFrame,
        group_by: list[str],
        aggs: list[AggregationSpec],
    ) -> pd.DataFrame:
        if not aggs:
            # If you group without aggregations, that's usually dangerous (can explode rows).
            # Force a count as a safe default.
            aggs = [AggregationSpec(field=group_by[0], agg=Aggregation.COUNT, as_name="row_count")]

        gb = df.groupby(group_by, dropna=False)

        # Build aggregation dict
        agg_dict: dict[str, Any] = {}
        distinct_specs: list[AggregationSpec] = []

        for a in aggs:
            out_name = a.as_name or f"{a.agg.value}_{a.field}"
            if a.agg == Aggregation.COUNT_DISTINCT:
                distinct_specs.append(AggregationSpec(field=a.field, agg=a.agg, as_name=out_name))
            else:
                if a.field not in agg_dict:
                    agg_dict[a.field] = {}
                agg_dict[a.field][out_name] = _AGG_MAP[a.agg]

        # pandas supports named aggregations via tuples or dict forms; use NamedAgg pattern
        named_aggs: dict[str, pd.NamedAgg] = {}
        for field, sub in agg_dict.items():
            for out_name, func in sub.items():
                named_aggs[out_name] = pd.NamedAgg(column=field, aggfunc=func)

        res = gb.agg(**named_aggs).reset_index()

        # Handle count distinct fields (slower, but controlled)
        for ds in distinct_specs:
            out_name = ds.as_name or f"count_distinct_{ds.field}"
            tmp = gb[ds.field].nunique(dropna=True).reset_index(name=out_name)
            res = res.merge(tmp, on=group_by, how="left")

        return res

    def _aggregate_no_group(self, df: pd.DataFrame, aggs: list[AggregationSpec]) -> pd.DataFrame:
        if not aggs:
            # Safe default single-row count
            return pd.DataFrame({"row_count": [len(df)]})

        out: dict[str, Any] = {}
        for a in aggs:
            name = a.as_name or f"{a.agg.value}_{a.field}"
            if a.agg == Aggregation.COUNT_DISTINCT:
                out[name] = df[a.field].nunique(dropna=True)
            else:
                func = _AGG_MAP[a.agg]
                out[name] = getattr(df[a.field], func)()
        return pd.DataFrame([out])

    def _apply_sort_topn_and_caps(self, df: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
        out = df

        # Sorting: allow sorting by produced aggregation alias too
        if spec.sort is not None:
            by = spec.sort.by
            ascending = spec.sort.direction == SortDirection.ASC

            if by not in out.columns:
                raise UnsupportedQuery(
                    f"Cannot sort by '{by}': not present in result columns {list(out.columns)}"
                )

            out = out.sort_values(by=by, ascending=ascending, kind="mergesort")

        # Top-N
        top_n = spec.top_n
        if top_n is not None:
            if top_n <= 0:
                raise ValidationError("top_n must be positive if provided.")
            out = out.head(top_n)

        # Row caps
        max_rows = spec.max_rows or self.limits.max_rows_default
        hard = self.limits.max_rows_hard
        if len(out) > hard:
            raise RowLimitExceeded(f"Query result has {len(out)} rows, exceeds hard limit {hard}.")
        if len(out) > max_rows:
            # Soft cap: truncate deterministically
            out = out.head(max_rows)

        return out.reset_index(drop=True)

    # ---------------------------
    # Helpers
    # ---------------------------

    def _derived_names(self, spec: QuerySpec) -> list[str]:
        names: list[str] = []
        for a in spec.aggregations:
            names.append(a.as_name or f"{a.agg.value}_{a.field}")
        return names

    def _make_lineage(self, tw: TimeWindow, spec: QuerySpec, row_count: int) -> Lineage:
        applied_filters: dict[str, Any] = {
            "account_id": spec.filters.account_id,
            "chat_type": spec.filters.chat_type,
            "chat_id": spec.filters.chat_id,
            "has_tasks": spec.filters.has_tasks,
            "has_classifications": spec.filters.has_classifications,
            "has_both": spec.filters.has_both,
        }
        return Lineage(
            source_type=SourceType.CSV,
            dataset_name=self.dataset_name,
            dataset_version=self._dataset_version,
            generated_at_utc=datetime.utcnow(),
            time_window=tw,
            applied_filters=applied_filters,
            group_by=list(spec.group_by),
            row_count=row_count,
            notes="csv_backend.safe_query",
        )
