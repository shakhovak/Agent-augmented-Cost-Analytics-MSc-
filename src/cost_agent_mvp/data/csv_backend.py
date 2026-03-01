"""CSV backend for loading and querying data."""

from __future__ import annotations

from datetime import date as dt_date  # Add at top of file if missing
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
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

SERVICE_COST_COLS = (
    "total_cost_tasks",
    "total_cost_classifications",
    "cost_amocrm_call",
)
DERIVED_FIELDS = {
    "total_cost_services",
    "active_users",
    "dials_analyzed",
    "avg_cost_per_active_account_day",
    "p95_cost_per_account_day",
    "pct_accounts_above_cap",
    "avg_margin_per_account_day",
    "delta_abs",
    "delta_pct",
    "share_of_total",
}
POST_DERIVED = {"delta_abs", "delta_pct", "share_of_total"}


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

    def _prepare_for_executor(self, df: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
        """
        Materialize derived fields + virtual dimensions needed for this spec.
        """
        out = df

        def _ensure_total_cost_services(local_df: pd.DataFrame) -> pd.DataFrame:
            if "total_cost_services" in local_df.columns:
                return local_df
            out = local_df.copy()
            for c in SERVICE_COST_COLS:
                if c not in out.columns:
                    out[c] = 0.0
            out["total_cost_services"] = out[list(SERVICE_COST_COLS)].sum(axis=1)
            return out

        # (A) Ensure row-level derived column exists (needed for many derived KPIs)
        out = _ensure_total_cost_services(out)

        # (B) If we need service_component dimension/filter, unpivot now
        if self._needs_service_component(spec):
            out = self._unpivot_service_component(out)

            # Apply service_component filter after unpivot
            vals = getattr(spec.filters, "service_component", None)
            if isinstance(vals, list) and vals:
                out = out[out["service_component"].isin(vals)]

        return out

    def query(self, spec: QuerySpec) -> Any:
        """
        Execute a QuerySpec safely and return object with .df and .lineage attributes.
        """
        tw = normalize_time_window(spec.time_window)
        self._enforce_time_window(tw, spec)
        self._enforce_supported_fields(
            spec
        )  # after you update 3.5 to allow derived + service_component

        # 1) Window + "base" filters on wide DF
        df0 = self._apply_time_window(self._df, tw)
        df1 = self._apply_filters(df0, spec.filters)

        # 2) Prepare derived/virtual structures BEFORE aggregation
        df1 = self._prepare_for_executor(df1, spec)
        # inside _prepare_for_executor after unpivot:
        vals = getattr(spec.filters, "service_component", None)
        if isinstance(vals, list) and vals:
            df1 = df1[df1["service_component"].isin(vals)]

        # 3) Aggregate
        if spec.group_by:
            df = self._group_and_aggregate(df1, spec.group_by, spec.aggregations)
        else:
            df = self._aggregate_no_group(df1, spec.aggregations)

        # 4) Sort / top-n / caps
        df = self._apply_sort_topn_and_caps(df, spec)

        lineage = self._make_lineage(tw=tw, spec=spec, row_count=len(df))

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

        # Derived metrics supported by backend executor (semantic layer derived_metrics)
        DERIVED_ALLOWED = {
            "total_cost_services",
            "active_users",
            "dials_analyzed",
            "avg_cost_per_active_account_day",
            "p95_cost_per_account_day",
            "pct_accounts_above_cap",
            "avg_margin_per_account_day",
            "delta_abs",
            "delta_pct",
            "share_of_total",
        }

        # Virtual dimension: only service_component is allowed (we materialize it by unpivot)
        VIRTUAL_GROUP_BY_ALLOWED = {"service_component"}

        for g in spec.group_by:
            if g in cols:
                continue
            if g in VIRTUAL_GROUP_BY_ALLOWED:
                continue
            raise UnsupportedQuery(f"Unsupported group_by field: '{g}'")

        for a in spec.aggregations:
            if a.field in cols:
                continue
            if a.field in DERIVED_ALLOWED:
                continue
            raise UnsupportedQuery(f"Unsupported aggregation field: '{a.field}'")

        # sort.by could be:
        # - a raw column
        # - an alias produced by aggregations (handled later)
        # - a derived name produced by executor
        if spec.sort is not None:
            if spec.sort.by in cols:
                return
            if spec.sort.by in self._derived_names(spec):
                return
            if spec.sort.by in DERIVED_ALLOWED:
                return
            # else: allow alias to be resolved later (existing behavior)
            # (Don't raise here.)
            return

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
        # ----- helpers / constants -----
        DERIVED_ALLOWED = {
            "total_cost_services",
            "active_users",
            "dials_analyzed",
            "avg_cost_per_active_account_day",
            "p95_cost_per_account_day",
            "pct_accounts_above_cap",
            "avg_margin_per_account_day",
            "share_of_total",
            "delta_abs",
            "delta_pct",
        }
        POST_WINDOW_DERIVED = {
            "delta_abs",
            "delta_pct",
        }  # needs comparison window; handled in query()
        SERVICE_COST_COLS = ("total_cost_tasks", "total_cost_classifications", "cost_amocrm_call")

        def _ensure_total_cost_services(local_df: pd.DataFrame) -> pd.DataFrame:
            if "total_cost_services" in local_df.columns:
                return local_df
            out = local_df.copy()
            for c in SERVICE_COST_COLS:
                if c not in out.columns:
                    out[c] = 0.0
            out["total_cost_services"] = out[list(SERVICE_COST_COLS)].sum(axis=1)
            return out

        def _business_rule_float(key: str, default: float) -> float:
            # Reads from semantic layer if present; otherwise falls back
            sl = getattr(self, "semantic_layer", None) or getattr(self, "_semantic_layer", None)
            if isinstance(sl, dict):
                try:
                    return float((sl.get("business_rules", {}) or {}).get(key, default))
                except Exception:
                    return float(default)
            return float(default)

        cap_usd = _business_rule_float("cost_cap_per_active_account_day_usd", 30.0)
        price_usd = _business_rule_float("price_per_active_account_day_usd", 100.0)

        # ----- split aggs: base vs derived -----
        if not aggs:
            # still safe default if absolutely nothing requested
            aggs = [AggregationSpec(field=group_by[0], agg=Aggregation.COUNT, as_name="row_count")]

        # Ensure we can compute service-based derived metrics
        df = _ensure_total_cost_services(df)

        derived_specs: list[AggregationSpec] = []
        base_specs: list[AggregationSpec] = []

        for a in aggs:
            if a.field in POST_WINDOW_DERIVED:
                raise UnsupportedQuery(
                    f"Derived metric '{a.field}' requires comparison window; implement in query() (delta)."
                )
            if a.field in DERIVED_ALLOWED and a.field not in df.columns:
                derived_specs.append(a)
            else:
                base_specs.append(a)

        # ----- run base aggregations (existing behavior) -----
        gb = df.groupby(group_by, dropna=False)

        # If there are no base specs but we do have derived specs, start from distinct groups
        if not base_specs:
            res = gb.size().reset_index().drop(columns=[0])  # just the group keys
        else:
            agg_dict: dict[str, Any] = {}
            distinct_specs: list[AggregationSpec] = []

            for a in base_specs:
                out_name = a.as_name or f"{a.agg.value}_{a.field}"
                if a.agg == Aggregation.COUNT_DISTINCT:
                    distinct_specs.append(
                        AggregationSpec(field=a.field, agg=a.agg, as_name=out_name)
                    )
                else:
                    if a.field not in agg_dict:
                        agg_dict[a.field] = {}
                    agg_dict[a.field][out_name] = _AGG_MAP[a.agg]

            named_aggs: dict[str, pd.NamedAgg] = {}
            for field, sub in agg_dict.items():
                for out_name, func in sub.items():
                    named_aggs[out_name] = pd.NamedAgg(column=field, aggfunc=func)

            res = gb.agg(**named_aggs).reset_index()

            for ds in distinct_specs:
                out_name = ds.as_name or f"count_distinct_{ds.field}"
                tmp = gb[ds.field].nunique(dropna=True).reset_index(name=out_name)
                res = res.merge(tmp, on=group_by, how="left")

        # ----- compute derived specs and merge into res -----
        if derived_specs:
            active = df[df["total_cost_services"] > 0].copy()
            g_active = active.groupby(group_by, dropna=False)

            # intermediates reused across multiple derived metrics
            # sum services per group
            svc_sum = g_active["total_cost_services"].sum()

            # active users per group
            au = g_active["account_id"].nunique(dropna=True)

            # dials analyzed per group (row count among active service rows)
            dials = g_active.size()

            # per-account totals per group for p95 and pct_above_cap
            per_acc = (
                active.groupby(group_by + ["account_id"], dropna=False)["total_cost_services"]
                .sum()
                .reset_index()
            )
            per_acc_g = per_acc.groupby(group_by, dropna=False)

            p95 = per_acc_g["total_cost_services"].quantile(0.95)
            pct_above = per_acc_g["total_cost_services"].apply(
                lambda s: float((s > float(cap_usd)).mean()) if len(s) else 0.0
            )

            avg_cost = (svc_sum / au.replace(0, np.nan)).fillna(0.0)
            avg_margin = float(price_usd) - avg_cost

            # Build derived df
            ddf = pd.DataFrame({k: res[k] for k in group_by}) if group_by else pd.DataFrame([{}])
            # Ensure unique group rows
            if group_by:
                ddf = ddf.drop_duplicates(subset=group_by).reset_index(drop=True)

            # Map field -> computed series
            computed = {
                "active_users": au,
                "dials_analyzed": dials,
                "avg_cost_per_active_account_day": avg_cost,
                "p95_cost_per_account_day": p95.fillna(0.0),
                "pct_accounts_above_cap": pct_above.fillna(0.0),
                "avg_margin_per_account_day": avg_margin.fillna(float(price_usd)),
            }

            # share_of_total: only meaningful when grouped
            share = None
            total = float(svc_sum.sum()) if len(svc_sum) else 0.0
            if total > 0:
                share = (svc_sum / total).fillna(0.0)
            else:
                share = svc_sum.copy()
                share[:] = 0.0
            computed["share_of_total"] = share

            # Attach requested derived outputs using their as_name
            # Use merge to align by group keys
            tmp = (
                pd.DataFrame({**{k: svc_sum.index.get_level_values(k) for k in group_by}})
                if group_by
                else pd.DataFrame([{}])
            )
            # The above is messy for MultiIndex; safer: reset_index() on each series:
            # We'll do it per metric to be robust.
            for a in derived_specs:
                out_name = a.as_name or f"{a.agg.value}_{a.field}"
                s = computed.get(a.field)
                if s is None:
                    raise UnsupportedQuery(
                        f"Derived metric '{a.field}' not implemented in backend."
                    )
                t = s.reset_index(name=out_name)
                ddf = ddf.merge(t, on=group_by, how="left")

            # Merge derived columns into result
            res = res.merge(ddf, on=group_by, how="left")

        return res

    def _ensure_total_cost_services(self, local_df: pd.DataFrame) -> pd.DataFrame:
        if "total_cost_services" in local_df.columns:
            return local_df
        out = local_df.copy()
        for c in SERVICE_COST_COLS:
            if c not in out.columns:
                out[c] = 0.0
        out["total_cost_services"] = out[list(SERVICE_COST_COLS)].sum(axis=1)
        return out

    def _needs_service_component(self, spec: QuerySpec) -> bool:
        if "service_component" in (spec.group_by or []):
            return True
        vals = getattr(spec.filters, "service_component", None)
        return isinstance(vals, list) and len(vals) > 0

    def _aggregate_no_group(self, df: pd.DataFrame, aggs: list[AggregationSpec]) -> pd.DataFrame:
        if not aggs:
            # Safe default single-row count
            return pd.DataFrame({"row_count": [len(df)]})

        DERIVED_FIELDS = {
            "total_cost_services",
            "active_users",
            "dials_analyzed",
            "avg_cost_per_active_account_day",
            "p95_cost_per_account_day",
            "pct_accounts_above_cap",
            "avg_margin_per_account_day",
            "delta_abs",
            "delta_pct",
            "share_of_total",
        }
        POST_WINDOW_DERIVED = {
            "delta_abs",
            "delta_pct",
        }  # needs prev window; handled in query()

        SERVICE_COST_COLS = (
            "total_cost_tasks",
            "total_cost_classifications",
            "cost_amocrm_call",
        )

        def _ensure_total_cost_services(local_df: pd.DataFrame) -> pd.DataFrame:
            if "total_cost_services" in local_df.columns:
                return local_df
            out = local_df.copy()
            for c in SERVICE_COST_COLS:
                if c not in out.columns:
                    out[c] = 0.0
            out["total_cost_services"] = out[list(SERVICE_COST_COLS)].sum(axis=1)
            return out

        def _business_rule_float(key: str, default: float) -> float:
            sl = getattr(self, "semantic_layer", None) or getattr(self, "_semantic_layer", None)
            if isinstance(sl, dict):
                try:
                    return float((sl.get("business_rules", {}) or {}).get(key, default))
                except Exception:
                    return float(default)
            return float(default)

        cap_usd = _business_rule_float(
            "cost_cap_per_active_account_day_usd", 30.0
        )  # semantic layer :contentReference[oaicite:7]{index=7}
        price_usd = _business_rule_float(
            "price_per_active_account_day_usd", 100.0
        )  # semantic layer :contentReference[oaicite:8]{index=8}

        # Ensure row-level services total exists (so downstream is consistent)
        df = _ensure_total_cost_services(df)

        # Active rows for derived metrics are defined by services > 0
        active = df[df["total_cost_services"] > 0].copy()

        # Precompute derived scalars once (cheap)
        services_sum = float(df["total_cost_services"].sum())  # zeros don't affect sum

        active_users = int(active["account_id"].nunique(dropna=True)) if not active.empty else 0

        # semantic layer: distinct (account_id, chat_id, chat_type) where services > 0 :contentReference[oaicite:9]{index=9}
        if not active.empty and all(
            c in active.columns for c in ["account_id", "chat_id", "chat_type"]
        ):
            dials_analyzed = int(
                active[["account_id", "chat_id", "chat_type"]].drop_duplicates().shape[0]
            )
        else:
            dials_analyzed = 0

        avg_cost = float(services_sum / active_users) if active_users > 0 else 0.0

        # p95 over per-account-day totals (account_id × date), within current filtered window :contentReference[oaicite:10]{index=10}
        if not active.empty and "date" in active.columns:
            per_acc_day = active.groupby(["date", "account_id"], dropna=False)[
                "total_cost_services"
            ].sum()
            vals = per_acc_day[per_acc_day > 0].to_numpy()
            p95_cost = float(np.percentile(vals, 95)) if len(vals) else 0.0
            # fraction of active account-days above cap (cap is per account-day)
            pct_above = float((vals > float(cap_usd)).mean()) if len(vals) else 0.0
        else:
            p95_cost = 0.0
            pct_above = 0.0

        avg_margin = float(price_usd) - float(avg_cost)

        # Build outputs
        out: dict[str, Any] = {}

        # Helper to return derived scalar for a.field
        def _derived_value(field: str) -> float | int:
            if field == "total_cost_services":
                return float(services_sum)
            if field == "active_users":
                return int(active_users)
            if field == "dials_analyzed":
                return int(dials_analyzed)
            if field == "avg_cost_per_active_account_day":
                return float(avg_cost)
            if field == "p95_cost_per_account_day":
                return float(p95_cost)
            if field == "pct_accounts_above_cap":
                # NOTE: returns fraction 0..1. If you want percent 0..100, multiply by 100 here.
                return float(pct_above)
            if field == "avg_margin_per_account_day":
                return float(avg_margin)
            if field == "share_of_total":
                # In no-group case share is only meaningful if you also compute a numerator.
                # We'll fill later once we see what else was requested.
                return float("nan")
            raise UnsupportedQuery(
                f"Derived metric '{field}' not implemented in _aggregate_no_group"
            )

        # First pass: compute everything except share_of_total and delta_*
        requested_names: list[tuple[str, str]] = []  # (field, out_name)
        for a in aggs:
            name = a.as_name or f"{a.agg.value}_{a.field}"
            requested_names.append((a.field, name))

            if a.field in POST_WINDOW_DERIVED:
                # must be computed in query() since it needs comparison window boundaries
                continue

            if a.field in DERIVED_FIELDS and a.field not in df.columns:
                out[name] = _derived_value(a.field)
                continue

            # Base columns (including total_cost_services which we materialized above)
            if a.agg == Aggregation.COUNT_DISTINCT:
                out[name] = int(df[a.field].nunique(dropna=True))
            else:
                func = _AGG_MAP[a.agg]
                out[name] = getattr(df[a.field], func)()

        # Second pass: share_of_total if requested
        # semantic layer: metric_value / total_cost_services_value :contentReference[oaicite:11]{index=11}
        for field, name in requested_names:
            if field != "share_of_total":
                continue

            denom = float(services_sum)
            if denom <= 0:
                out[name] = 0.0
                continue

            # Choose numerator:
            # if exactly one other numeric output exists, use it; otherwise default to denom (=> 1.0)
            other_keys = [
                n
                for (f, n) in requested_names
                if f not in {"share_of_total", "delta_abs", "delta_pct"}
            ]
            if len(other_keys) == 1 and other_keys[0] in out:
                num = float(out[other_keys[0]])
                out[name] = float(num / denom)
            else:
                out[name] = 1.0

        # delta_* left for query() to fill
        for field, name in requested_names:
            if field in POST_WINDOW_DERIVED:
                out[name] = np.nan

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

    def _unpivot_service_component(self, df: pd.DataFrame) -> pd.DataFrame:
        SERVICE_COST_COLS = (
            "total_cost_tasks",
            "total_cost_classifications",
            "cost_amocrm_call",
        )

        # Make sure the base columns exist (avoid KeyError)
        out = df.copy()
        for c in SERVICE_COST_COLS:
            if c not in out.columns:
                out[c] = 0.0

        # IMPORTANT: value_name must NOT collide with existing columns
        tmp_value = "__component_cost__"

        # id_vars should exclude the service columns and also exclude the temporary
        id_vars = [c for c in out.columns if c not in SERVICE_COST_COLS and c != tmp_value]

        long = out.melt(
            id_vars=id_vars,
            value_vars=list(SERVICE_COST_COLS),
            var_name="service_component",
            value_name=tmp_value,
        )

        mapping = {
            "total_cost_tasks": "tasks",
            "total_cost_classifications": "classifications",
            "cost_amocrm_call": "amocrm_calls",
        }
        long["service_component"] = (
            long["service_component"].map(mapping).fillna(long["service_component"])
        )

        # Now define total_cost_services as the per-component cost
        long["total_cost_services"] = long[tmp_value].fillna(0.0)
        long = long.drop(columns=[tmp_value])

        # keep only active component rows (optional but usually desired)
        long = long[long["total_cost_services"] > 0].copy()

        return long
