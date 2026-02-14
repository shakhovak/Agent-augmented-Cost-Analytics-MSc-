from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

# Planner models are used only for typing. Importing them at runtime is OK
# (planner does not import plan_compiler, so there is no circular import).
from cost_agent_mvp.agent.planner import FilterSpec, PlannerPlan  # noqa: E402
from cost_agent_mvp.core.errors import UnsupportedQuery
from cost_agent_mvp.core.types import (
    Aggregation,
    AggregationSpec,
    QueryFilters,
    QuerySpec,
    SortDirection,
    SortSpec,
    TimeWindow,
    TimeWindowType,
)


class TimeWindowExceeded(UnsupportedQuery):
    """Raised when the requested time window exceeds semantic limits."""


class RowLimitExceeded(UnsupportedQuery):
    """Raised when requested row/top-N settings exceed semantic limits."""


# -------------------------
# Small helpers
# -------------------------


def _raise_unsupported(msg: str) -> None:
    raise UnsupportedQuery(msg)


def _supported_aggs_by_backend() -> set[Aggregation]:
    """CsvBackend supports a fixed set of aggregations."""
    supported: set[Aggregation] = set()
    for name in ("SUM", "MEAN", "MEDIAN", "COUNT", "COUNT_DISTINCT"):
        if hasattr(Aggregation, name):
            supported.add(getattr(Aggregation, name))
    # Fallback: if enum differs, assume all enum values are supported.
    return supported or set(Aggregation)


def _get_row_safety(semantic_layer: Mapping[str, Any]) -> dict[str, int]:
    """Extract safety limits from semantic_layer.yaml (supports a few layouts)."""
    # Preferred: semantic_layer.dataset.row_safety
    rs = semantic_layer.get("dataset", {}).get("row_safety", {})  # type: ignore[call-arg]
    if isinstance(rs, dict) and rs:
        return {k: int(v) for k, v in rs.items() if isinstance(v, int | float | str)}

    # Back-compat: semantic_layer.limits
    rs2 = semantic_layer.get("limits", {})  # type: ignore[call-arg]
    if isinstance(rs2, dict) and rs2:
        return {k: int(v) for k, v in rs2.items() if isinstance(v, int | float | str)}

    return {}


def _get_allowed_ops(semantic_layer: Mapping[str, Any]) -> dict[str, Any]:
    ops = semantic_layer.get("allowed_operations", {})  # type: ignore[call-arg]
    return ops if isinstance(ops, dict) else {}


def _allowed_dimensions(semantic_layer: Mapping[str, Any]) -> set[str]:
    dims = semantic_layer.get("dimensions", [])  # type: ignore[call-arg]
    out: set[str] = set()
    if isinstance(dims, list):
        for d in dims:
            if isinstance(d, dict) and isinstance(d.get("name"), str):
                out.add(d["name"])
    return out


def _allowed_measures(semantic_layer: Mapping[str, Any]) -> set[str]:
    ms = semantic_layer.get("measures", [])  # type: ignore[call-arg]
    out: set[str] = set()
    if isinstance(ms, list):
        for m in ms:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                out.add(m["name"])
    return out


def _derived_metrics(semantic_layer: Mapping[str, Any]) -> set[str]:
    dm = semantic_layer.get("derived_metrics", [])  # type: ignore[call-arg]
    out: set[str] = set()
    if isinstance(dm, list):
        for m in dm:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                out.add(m["name"])
    return out


def _virtual_dimensions(semantic_layer: Mapping[str, Any]) -> set[str]:
    vd = semantic_layer.get("virtual_dimensions", [])  # type: ignore[call-arg]
    out: set[str] = set()
    if isinstance(vd, list):
        for d in vd:
            if isinstance(d, dict) and isinstance(d.get("name"), str):
                out.add(d["name"])
    return out


def _allowed_filter_names(semantic_layer: Mapping[str, Any]) -> set[str]:
    ops = _get_allowed_ops(semantic_layer)
    raw = ops.get("filters", [])

    out: set[str] = set()
    if isinstance(raw, dict):
        out |= {str(k) for k in raw.keys()}
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                out.add(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                out.add(item["name"])
    return out


def _allowed_aggs(semantic_layer: Mapping[str, Any]) -> set[Aggregation]:
    ops = _get_allowed_ops(semantic_layer)
    raw = ops.get("aggregations", [])

    backend = _supported_aggs_by_backend()
    allowed: set[Aggregation] = set()

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Aggregation):
                allowed.add(item)
            elif isinstance(item, str):
                try:
                    allowed.add(Aggregation(item))
                except Exception:
                    # allow aliases like "avg"
                    if item.lower() == "avg" and hasattr(Aggregation, "MEAN"):
                        allowed.add(Aggregation.MEAN)

    allowed = {a for a in allowed if a in backend}

    # Safe default: SUM (if supported)
    if not allowed:
        if hasattr(Aggregation, "SUM"):
            allowed = {Aggregation.SUM}
        else:
            allowed = backend

    return allowed


def _coerce_yyyy_mm_dd(v: Any) -> str:
    if v is None:
        raise ValueError("date is required")
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip()
    # Basic validation
    datetime.strptime(s, "%Y-%m-%d")
    return s


_METRIC_CALL_RE = re.compile(r"^(sum|mean|median|count|count_distinct)\((.+)\)$", re.IGNORECASE)
_METRIC_DUNDER_RE = re.compile(r"^(.+?)__(sum|mean|median|count|count_distinct)$", re.IGNORECASE)
_METRIC_PREFIX_RE = re.compile(r"^(sum|mean|median|count|count_distinct)_(.+)$", re.IGNORECASE)


def _parse_metric_token(token: str) -> tuple[str, Aggregation | None]:
    """Parse metric token into (field, agg_override).

    Supported forms:
      - total_cost_tasks
      - sum_total_cost_tasks
      - total_cost_tasks__sum
      - sum(total_cost_tasks)
    """
    t = str(token).strip()

    m = _METRIC_CALL_RE.match(t)
    if m:
        agg_s = m.group(1).lower()
        field = m.group(2).strip()
        return field, Aggregation(agg_s)

    m = _METRIC_DUNDER_RE.match(t)
    if m:
        field = m.group(1).strip()
        agg_s = m.group(2).lower()
        return field, Aggregation(agg_s)

    m = _METRIC_PREFIX_RE.match(t)
    if m:
        agg_s = m.group(1).lower()
        field = m.group(2).strip()
        return field, Aggregation(agg_s)

    return t, None


def _compile_time_window(
    tw: Any,
    *,
    semantic_layer: Mapping[str, Any],
    anchor_day: date,
) -> tuple[TimeWindow, int]:
    """Return (TimeWindow, span_days)."""
    if tw is None:
        return TimeWindow(type=TimeWindowType.RANGE, start=str(anchor_day), end=str(anchor_day))

    tw_type = str(getattr(tw, "type", "yesterday")).strip().lower()
    max_days_hard = semantic_layer.get("limits", {}).get("max_days_hard", 14)
    # allow aliases
    if tw_type in {"range", "date_range"}:
        tw_type = "date_range"

    allowed = _get_allowed_ops(semantic_layer).get("time_windows", [])
    allowed_set = {str(x).lower() for x in allowed} if isinstance(allowed, list) else set()
    if allowed_set and tw_type not in allowed_set and tw_type != "single_day":
        _raise_unsupported(f"Unsupported time_window.type: {tw_type}")

    if tw_type == "yesterday":
        return (
            TimeWindow(type=TimeWindowType.YESTERDAY, n_days=None, start=None, end=None),
            1,
        )

    if tw_type == "last_n_days":
        n_days = getattr(tw, "n_days", None)
        if n_days is None:
            _raise_unsupported("last_n_days requires time_window.n_days")
        n = int(n_days)
        if n <= 0:
            _raise_unsupported("last_n_days.n_days must be > 0")
        return (
            TimeWindow(type=TimeWindowType.LAST_N_DAYS, n_days=n, start=None, end=None),
            n,
        )
    # === FIX: Handle single_day → convert to RANGE with matching start/end ===
    if tw_type == "single_day":
        day_str = str(getattr(tw, "day", "")).strip()
        if not day_str:
            _raise_unsupported("single_day requires 'day' field with ISO date (YYYY-MM-DD)")
        # Validate date format
        try:
            from datetime import datetime

            datetime.strptime(day_str, "%Y-%m-%d")
        except ValueError:
            _raise_unsupported(
                f"single_day 'day' field must be ISO date YYYY-MM-DD, got: {day_str}"
            )
        start = day_str
        end = day_str
        span_days = 1  # ← Critical: define span_days
        if span_days > max_days_hard:
            raise TimeWindowExceeded(span_days, max_days_hard)
        # ✅ FIXED: Return TUPLE (TimeWindow, int) like all other branches
        return (
            TimeWindow(type=TimeWindowType.RANGE, start=start, end=end, n_days=None),
            span_days,  # ← ADD THIS (comma makes it a tuple)
        )

    if tw_type == "date_range":
        start_s = _coerce_yyyy_mm_dd(getattr(tw, "start", None))
        end_s = _coerce_yyyy_mm_dd(getattr(tw, "end", None))
        sd = date.fromisoformat(start_s)
        ed = date.fromisoformat(end_s)
        if ed < sd:
            _raise_unsupported("date_range requires end >= start")
        span = (ed - sd).days + 1
        return (
            TimeWindow(type=TimeWindowType.RANGE, n_days=None, start=start_s, end=end_s),
            span,
        )

    _raise_unsupported(f"Unsupported time_window.type: {tw_type}")


def _enforce_cap(*, requested: int | None, default: int, hard_max: int, label: str) -> int:
    v = default if requested is None else int(requested)
    if v < 1:
        v = default
    if v > hard_max:
        v = hard_max
    return v


def _compile_filters(
    filters: Sequence[FilterSpec],
    *,
    semantic_layer: Mapping[str, Any],
) -> QueryFilters:
    allowed = _allowed_filter_names(semantic_layer)
    virtual_dims = _virtual_dimensions(semantic_layer)

    # QueryFilters is frozen -> construct in one go
    data: dict[str, Any] = {}

    for f in filters or []:
        # ✅ CRITICAL FIX: Normalize name for robust matching
        raw_name = str(getattr(f, "name", "")).strip()
        # Convert to consistent format: lowercase + underscores only
        norm_name = raw_name.lower().replace("-", "_").replace(" ", "_")

        # Skip time window filters (they belong in time_window field, not filters)
        if norm_name in {
            "single_day",
            "yesterday",
            "last_n_days",
            "date_range",
            "range",
        }:
            continue  # Silently ignore invalid filter types
        args = getattr(f, "args", {}) or {}

        if norm_name not in allowed:
            _raise_unsupported(f"Unsupported filter: {raw_name}")

        # prevent virtual dimension filters
        if norm_name in virtual_dims or norm_name.startswith("service_component"):
            _raise_unsupported(
                "Filter on virtual dimension 'service_component' is not supported by CsvBackend yet."
            )

        # membership filters
        if norm_name.endswith("_in"):
            vals = args.get("values", None)
            if vals is None:
                vals = args.get("value", None)
            if not isinstance(vals, list) or not vals:
                _raise_unsupported(f"Filter {norm_name} expects args.values as a non-empty list")

            # cap filter cardinality
            rs = _get_row_safety(semantic_layer)
            max_vals = int(rs.get("max_filter_values_in", 100))
            vals = vals[:max_vals]

            if norm_name == "account_id_in":
                data["account_id"] = [int(x) for x in vals]
            elif norm_name == "chat_id_in":
                # backend expects list[str]
                data["chat_id"] = [str(x) for x in vals]
            elif norm_name == "chat_type_in":
                data["chat_type"] = [str(x) for x in vals]
            else:
                _raise_unsupported(f"Filter not yet mapped to backend: {norm_name}")
            continue

        # boolean flags
        if norm_name in {"has_tasks", "has_classifications", "has_both"}:
            v = args.get("value", None)
            if not isinstance(v, bool):
                _raise_unsupported(f"Filter {norm_name} expects args.value as boolean")
            data[norm_name] = v
            continue

        _raise_unsupported(f"Unhandled filter mapping: {norm_name}")

    return QueryFilters(**data)


def compile_plan_to_queryspec(
    plan: PlannerPlan,
    semantic_layer: Mapping[str, Any],
    *,
    anchor_day: date | None = None,
) -> list[QuerySpec]:
    """Convert PlannerPlan -> list[QuerySpec] safely.

    Naming convention for output metrics (matches updated unit tests):
      as_name = "{agg}_{field}" e.g. "sum_total_cost_tasks".

    sort_by normalization accepts:
      - "sum_total_cost_tasks"
      - "total_cost_tasks" (-> sum_...)
      - "total_cost_tasks__sum" (-> sum_...)
      - "sum(total_cost_tasks)" (-> sum_...)
    """
    if anchor_day is None:
        anchor_day = date.today()
    print(f"compiler: {anchor_day}")

    status = str(getattr(plan, "status", "")).upper()
    if status != "OK":
        reason = getattr(plan, "reason", "") or ""
        _raise_unsupported(f"Planner status={status}. {reason}".strip())

    allowed_dims = _allowed_dimensions(semantic_layer)
    allowed_measures = _allowed_measures(semantic_layer)
    derived = _derived_metrics(semantic_layer)
    virtual_dims = _virtual_dimensions(semantic_layer)

    # --- group_by ---
    group_by = list(getattr(plan, "group_by", []) or [])
    for dim in group_by:
        if dim in virtual_dims:
            _raise_unsupported(f"Group-by '{dim}' is a virtual dimension and not supported yet.")
        if dim not in allowed_dims:
            _raise_unsupported(f"Unsupported group-by dimension: {dim}")

    # --- metrics -> aggregations ---
    metrics = list(getattr(plan, "metrics", []) or [])
    if not metrics:
        _raise_unsupported("Plan has no metrics; at least one metric is required")

    allowed_aggs = _allowed_aggs(semantic_layer)
    aggs: list[AggregationSpec] = []
    for tok in metrics:
        field, agg_override = _parse_metric_token(tok)

        if field in derived:
            _raise_unsupported(f"Metric '{field}' is derived and not queryable yet")
        if field not in allowed_measures:
            _raise_unsupported(f"Unsupported metric: {field}")

        agg = agg_override or (
            Aggregation.SUM if hasattr(Aggregation, "SUM") else next(iter(allowed_aggs))
        )
        if agg not in allowed_aggs:
            _raise_unsupported(f"Aggregation '{agg.value}' is not allowed")

        as_name = f"{agg.value}_{field}"
        aggs.append(AggregationSpec(field=field, agg=agg, as_name=as_name))

    output_cols = {a.as_name for a in aggs}

    # --- time window ---
    tw_core, span_days = _compile_time_window(
        getattr(plan, "time_window", None),
        semantic_layer=semantic_layer,
        anchor_day=anchor_day,
    )

    rs = _get_row_safety(semantic_layer)
    max_days_hard = int(rs.get("max_days_hard", 14))
    if span_days > max_days_hard:
        raise TimeWindowExceeded(
            f"Requested time window is {span_days} days, exceeds max_days_hard={max_days_hard}"
        )

    # --- caps ---
    top_n = _enforce_cap(
        requested=getattr(plan, "top_n", None),
        default=int(rs.get("default_top_n", 10)),
        hard_max=int(rs.get("max_top_n", 50)),
        label="top_n",
    )
    max_rows = _enforce_cap(
        requested=(getattr(plan, "max_rows", None) if hasattr(plan, "max_rows") else None),
        default=int(rs.get("max_rows_default", 5000)),
        hard_max=int(rs.get("max_rows_hard", 20000)),
        label="max_rows",
    )
    if top_n > max_rows:
        raise RowLimitExceeded(f"top_n={top_n} exceeds max_rows={max_rows}")

    # --- filters ---
    qf = _compile_filters(getattr(plan, "filters", []) or [], semantic_layer=semantic_layer)

    # --- sort ---
    def _normalize_sort_by(raw: str) -> str:
        s = str(raw).strip()
        if not s:
            return s
        if s in group_by or s in output_cols:
            return s

        # field__sum -> sum_field
        m = _METRIC_DUNDER_RE.match(s)
        if m:
            field = m.group(1).strip()
            agg_s = m.group(2).lower()
            cand = f"{agg_s}_{field}"
            if cand in output_cols:
                return cand

        # sum(field)
        m = _METRIC_CALL_RE.match(s)
        if m:
            agg_s = m.group(1).lower()
            field = m.group(2).strip()
            cand = f"{agg_s}_{field}"
            if cand in output_cols:
                return cand

        # bare field -> assume sum_field
        cand = f"sum_{s}"
        if cand in output_cols:
            return cand

        return s

    sort: SortSpec | None = None
    raw_sort_by = getattr(plan, "sort_by", None)
    if raw_sort_by:
        normalized = _normalize_sort_by(raw_sort_by)
        if normalized not in group_by and normalized not in output_cols:
            _raise_unsupported(f"sort_by='{raw_sort_by}' must be in group_by or in output metrics")

        direction = SortDirection.DESC
        raw_dir = getattr(plan, "sort_dir", None)
        if raw_dir and str(raw_dir).lower().startswith("asc"):
            direction = SortDirection.ASC

        sort = SortSpec(by=normalized, direction=direction)

    spec = QuerySpec(
        time_window=tw_core,
        group_by=group_by,
        aggregations=aggs,
        filters=qf,
        sort=sort,
        top_n=top_n,
        max_rows=max_rows,
    )

    return [spec]
