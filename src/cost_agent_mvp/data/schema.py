"""Column types and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from cost_agent_mvp.core.errors import DataSourceError, ValidationError


@dataclass(frozen=True)
class DatasetSchema:
    """
    Schema/validation rules for the joint CSV table.

    Notes:
    - We keep this strict enough to avoid silent failures,
      but flexible enough to accept minor dtype differences.
    - 'date' is normalized to python datetime.date (daily grain).
    """

    required_columns: tuple[str, ...]
    date_column: str = "date"

    # Optional: columns we will attempt to coerce to numeric if present
    numeric_columns: tuple[str, ...] = (
        "cost_dialog",
        "cost_task",
        "total_cost_tasks",
        "cost_classification",
        "cost_qc",
        "cost_check_list",
        "total_cost_classifications",
        "total_cost",
    )

    # Optional: columns we will attempt to coerce to int if present
    int_columns: tuple[str, ...] = (
        "account_id",
        "has_tasks",
        "has_classifications",
        "has_both",
    )

    # Optional: columns we will coerce to string if present
    str_columns: tuple[str, ...] = (
        "chat_id",
        "chat_type",
    )


def default_joint_schema() -> DatasetSchema:
    """
    Default schema for your current joint table (15 columns).
    If your CSV changes, update here (or load from semantic_layer.yaml later).
    """
    cols = (
        "account_id",
        "chat_id",
        "chat_type",
        "date",
        "cost_dialog",
        "cost_task",
        "total_cost_tasks",
        "cost_classification",
        "cost_qc",
        "cost_check_list",
        "total_cost_classifications",
        "total_cost",
        "has_tasks",
        "has_classifications",
        "has_both",
    )
    return DatasetSchema(required_columns=cols, date_column="date")


def validate_columns(df: pd.DataFrame, schema: DatasetSchema) -> None:
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise DataSourceError(
            f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}"
        )


def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parse a date column into python datetime.date.
    Accepts ISO strings or pandas datetime-like. Keeps NaT as error.
    """
    # Convert to datetime (UTC-agnostic; date-only)
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.isna().any():
        bad_count = int(dt.isna().sum())
        examples = s[dt.isna()].astype(str).head(5).tolist()
        raise ValidationError(f"Failed to parse {bad_count} date values. Examples: {examples}")
    return dt.dt.date


def coerce_types(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """
    Returns a new DataFrame with normalized types:
    - date column → datetime.date
    - known numeric columns → float (NaN allowed)
    - known int columns → Int64 (nullable) then filled if desired by callers
    - known string columns → string
    """
    out = df.copy()

    # Date normalization
    if schema.date_column not in out.columns:
        raise DataSourceError(f"Date column '{schema.date_column}' not found.")
    out[schema.date_column] = _parse_date_series(out[schema.date_column])

    # Numeric coercion
    for c in schema.numeric_columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Int coercion
    for c in schema.int_columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    # String coercion
    for c in schema.str_columns:
        if c in out.columns:
            out[c] = out[c].astype("string")

    return out


def load_csv(
    csv_path: str,
    schema: DatasetSchema | None = None,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Loads the CSV and validates/coerces according to schema.
    """
    schema = schema or default_joint_schema()
    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except Exception as e:
        raise DataSourceError(f"Failed to read CSV: {csv_path}") from e

    validate_columns(df, schema)
    return coerce_types(df, schema)
