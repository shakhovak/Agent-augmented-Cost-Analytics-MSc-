from datetime import date

import pandas as pd
import pytest

from cost_agent_mvp.core.constants import Aggregation, SortDirection, SourceType, TimeWindowType
from cost_agent_mvp.core.errors import (
    RowLimitExceeded,
    TimeWindowExceeded,
    UnsupportedQuery,
)
from cost_agent_mvp.core.types import (
    AggregationSpec,
    QueryFilters,
    QuerySpec,
    SortSpec,
    TimeWindow,
)
from cost_agent_mvp.data.csv_backend import CsvBackend
from cost_agent_mvp.data.schema import default_joint_schema


def _write_valid_csv(path, df: pd.DataFrame) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _base_df():
    """
    Minimal rows that should pass default_joint_schema() and support backend operations.
    We build all required columns with safe defaults, plus the columns used in query tests.
    """
    schema = default_joint_schema()
    rows = []

    # Create 4 rows over 2 days, 2 accounts, 2 chat_ids
    for d, acc, chat_id, chat_type, cost in [
        ("2026-01-01", 1, "c1", "support", 10.0),
        ("2026-01-01", 1, "c2", "support", 5.0),
        ("2026-01-02", 2, "c3", "sales", 7.0),
        ("2026-01-02", 2, "c4", "sales", 3.0),
    ]:
        r = {}
        for c in schema.required_columns:
            if c == "date":
                r[c] = d
            elif c == "account_id":
                r[c] = acc
            elif c == "chat_id":
                r[c] = chat_id
            elif c == "chat_type":
                r[c] = chat_type
            elif c == "total_cost":
                r[c] = cost
            else:
                # fill everything else required with 0 / empty string
                r[c] = 0
        rows.append(r)

    return pd.DataFrame(rows)


def test_basic_group_aggregation_sum(tmp_path):
    p = tmp_path / "data.csv"
    _write_valid_csv(p, _base_df())

    backend = CsvBackend(str(p))
    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2026, 1, 1), end=date(2026, 1, 2)
        ),
        filters=QueryFilters(account_id=[1]),
        group_by=["date"],
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM, as_name="cost_sum")],
        sort=SortSpec(by="date", direction=SortDirection.ASC),
    )

    out, lineage = backend.query(spec)

    assert lineage.source_type == SourceType.CSV
    assert lineage.dataset_name == "joint_costs_daily"
    assert lineage.dataset_version  # non-empty
    assert lineage.generated_at_utc.tzinfo is None

    assert lineage.time_window.start == date(2026, 1, 1)
    assert lineage.time_window.end == date(2026, 1, 2)

    assert lineage.applied_filters["account_id"] == [1]
    assert lineage.group_by == ["date"]
    assert lineage.row_count == len(out)

    assert list(out.columns) == ["date", "cost_sum"]
    assert len(out) == 1
    assert out["cost_sum"].iloc[0] == pytest.approx(15.0)

    assert lineage.dataset_name == "joint_costs_daily"
    assert lineage.row_count == 1
    assert lineage.applied_filters["account_id"] == [1]


def test_time_window_hard_limit_enforced(tmp_path):
    p = tmp_path / "data.csv"
    _write_valid_csv(p, _base_df())

    backend = CsvBackend(str(p))
    # huge range to trigger max_days_hard
    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2020, 1, 1), end=date(2026, 1, 1)
        ),
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM)],
    )

    with pytest.raises(TimeWindowExceeded):
        backend.query(spec)


def test_drilldown_chat_id_requires_account_id(tmp_path):
    p = tmp_path / "data.csv"
    _write_valid_csv(p, _base_df())

    backend = CsvBackend(str(p))
    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2026, 1, 1), end=date(2026, 1, 2)
        ),
        filters=QueryFilters(chat_id=["c1"]),  # drilldown without account_id
        group_by=["chat_id"],
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM)],
    )

    with pytest.raises(UnsupportedQuery):
        backend.query(spec)


def test_unsupported_group_by_raises(tmp_path):
    p = tmp_path / "data.csv"
    _write_valid_csv(p, _base_df())

    backend = CsvBackend(str(p))
    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2026, 1, 1), end=date(2026, 1, 2)
        ),
        group_by=["not_a_column"],
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM)],
    )

    with pytest.raises(UnsupportedQuery):
        backend.query(spec)


def test_soft_row_cap_truncates(tmp_path):
    p = tmp_path / "data.csv"

    # Build many distinct chat_ids but within account_id=1 to satisfy drilldown rule
    base = _base_df()
    big = pd.concat([base] * 50, ignore_index=True)

    # force account_id=1 for all rows and generate many distinct chat_id values
    big["account_id"] = 1
    big["chat_id"] = [f"c{i}" for i in range(len(big))]

    _write_valid_csv(p, big)

    backend = CsvBackend(str(p))

    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2026, 1, 1), end=date(2026, 1, 2)
        ),
        filters=QueryFilters(account_id=[1]),  # <-- required for chat_id drilldown
        group_by=["chat_id"],
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM, as_name="s")],
        max_rows=5,  # soft cap
        sort=SortSpec(by="s", direction=SortDirection.DESC),
    )

    out, _ = backend.query(spec)
    assert len(out) == 5


def test_hard_row_cap_raises(tmp_path):
    from cost_agent_mvp.core.constants import (
        SafetyLimits,
    )  # import here to keep scope local

    p = tmp_path / "data.csv"

    # Make lots of distinct groups (chat_id) within account_id=1
    n = 2000
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"] * n,
            "account_id": [1] * n,
            "chat_id": [f"c{i}" for i in range(n)],
            "chat_type": ["support"] * n,
            "total_cost": [1.0] * n,
        }
    )

    # Fill any other required schema columns
    schema = default_joint_schema()
    for c in schema.required_columns:
        if c not in df.columns:
            df[c] = 0

    _write_valid_csv(p, df)

    # Force a low hard cap so the test is deterministic
    limits = SafetyLimits(max_rows_hard=100, max_rows_default=100)

    backend = CsvBackend(str(p), limits=limits)

    spec = QuerySpec(
        time_window=TimeWindow(
            type=TimeWindowType.RANGE, start=date(2026, 1, 1), end=date(2026, 1, 1)
        ),
        filters=QueryFilters(account_id=[1]),  # required for chat_id drilldown
        group_by=["chat_id"],
        aggregations=[AggregationSpec(field="total_cost", agg=Aggregation.SUM, as_name="s")],
        # ensure we don't soft-truncate before we can exceed hard (max_rows_default also 100 here)
        max_rows=1000,
    )

    with pytest.raises(RowLimitExceeded):
        backend.query(spec)
