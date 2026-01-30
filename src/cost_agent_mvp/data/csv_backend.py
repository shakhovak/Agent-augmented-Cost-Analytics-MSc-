"""CSV backend for loading and querying data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from src.core.constants import (
    Aggregation,
    SafetyLimits,
    SortDirection,
    SourceType,
)
from src.core.errors import (
    RowLimitExceeded,
    TimeWindowExceeded,
    UnsupportedQuery,
    ValidationError,
)
from src.core.types import (
    AggregationSpec,
    Lineage,
    QueryFilters,
    QuerySpec,
    TimeWindow,
)
from src.core.utils_dates import normalize_time_window
from src.core.utils_hash import sha256_file
from src.data.schema import DatasetSchema, default_joint_schema, load_csv

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

    def query(self, spec: QuerySpec) -> tuple[pd.DataFrame, Lineage]:
        """
        Execute a QuerySpec safely and return (result_df, lineage).
        """
        tw = normalize_time_window(spec.time_window)

        self._enforce_time_window(tw, spec)
        self._enforce_supported_fields(spec)

        df0 = self._apply_time_window(self._df, tw)
        df1 = self._apply_filters(df0, spec.filters)

        if spec.group_by:
            result = self._group_and_aggregate(df1, spec.group_by, spec.aggregations)
        else:
            # No group_by: still allow aggregations (returns single-row table)
            result = self._aggregate_no_group(df1, spec.aggregations)

        result = self._apply_sort_topn_and_caps(result, spec)

        lineage = self._make_lineage(
            tw=tw,
            spec=spec,
            row_count=len(result),
        )
        return result, lineage

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

    def _enforce_time_window(self, tw: TimeWindow, spec: QuerySpec) -> None:
        if tw.start is None or tw.end is None:
            raise ValidationError("TimeWindow must be normalized before enforcement.")

        days = (tw.end - tw.start).days + 1
        hard = self.limits.max_days_hard
        if days > hard:
            raise TimeWindowExceeded(f"Requested {days} days exceeds hard limit {hard} days.")

        # Drilldown safety: if chat_id is filtered or grouped, require account_id and restrict days
        uses_chat_id = ("chat_id" in spec.group_by) or (spec.filters.chat_id is not None)
        if uses_chat_id:
            if not spec.filters.account_id:
                raise UnsupportedQuery("Drilldown by chat_id requires an account_id filter.")
            if days > self.limits.drilldown_max_days:
                raise TimeWindowExceeded(
                    f"Drilldown queries are limited to {self.limits.drilldown_max_days} days."
                )

    # ---------------------------
    # Query execution steps
    # ---------------------------

    def _apply_time_window(self, df: pd.DataFrame, tw: TimeWindow) -> pd.DataFrame:
        assert tw.start is not None and tw.end is not None
        # 'date' is already datetime.date (from schema coercion)
        mask = (df[self.schema.date_column] >= tw.start) & (df[self.schema.date_column] <= tw.end)
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
