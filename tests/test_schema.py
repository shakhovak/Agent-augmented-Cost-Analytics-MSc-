import pandas as pd
import pytest

from cost_agent_mvp.core.errors import DataSourceError, ValidationError
from cost_agent_mvp.data.schema import (
    coerce_types,
    default_joint_schema,
    validate_columns,
)


def make_minimal_valid_df():
    return pd.DataFrame(
        {
            "account_id": [123],
            "chat_id": ["c1"],
            "chat_type": ["whatsapp"],
            "date": ["2025-01-01"],
            "cost_dialog": [1.0],
            "cost_task": [2.0],
            "total_cost_tasks": [2.0],
            "cost_classification": [3.0],
            "cost_qc": [0.5],
            "cost_check_list": [0.2],
            "total_cost_classifications": [3.7],
            "cost_amocrm_call": [0.9],
            "has_amocrm_call": [0],
            "total_cost": [1.0 + 2.0 + 3.7 + 0.9],
            "num_services": [2],
            "has_tasks": [1],
            "has_classifications": [1],
            "has_both": [1],
        }
    )


def test_validate_columns_passes_on_valid_minimal_df():
    schema = default_joint_schema()
    df = make_minimal_valid_df()
    validate_columns(df, schema)


def test_validate_columns_fails_on_missing_required_column():
    schema = default_joint_schema()
    df = make_minimal_valid_df().drop(columns=["total_cost"])
    with pytest.raises(DataSourceError):
        validate_columns(df, schema)


def test_coerce_types_parses_date_and_numeric():
    schema = default_joint_schema()
    df = make_minimal_valid_df()

    # make some numeric columns string-typed
    df["cost_task"] = df["cost_task"].astype(str)
    df["total_cost"] = df["total_cost"].astype(str)

    out = coerce_types(df, schema)

    assert isinstance(out.loc[0, "date"], __import__("datetime").date)
    assert pd.api.types.is_float_dtype(out["cost_task"])
    assert pd.api.types.is_float_dtype(out["total_cost"])


def test_coerce_types_raises_on_bad_numeric_type():
    schema = default_joint_schema()
    df = make_minimal_valid_df()

    # simulate bad raw input as string column (common when reading CSV)
    df["cost_task"] = df["cost_task"].astype("string")
    df.loc[0, "cost_task"] = "not-a-number"

    with pytest.raises(ValidationError):
        _ = coerce_types(df, schema)


def test_optional_columns_can_be_missing():
    schema = default_joint_schema()
    df = make_minimal_valid_df()

    # Ensure some known optional columns are NOT present
    assert "has_all_three" not in df.columns
    assert "has_1_service" not in df.columns

    # Should still validate
    validate_columns(df, schema)

    # And coercion should still work
    out = coerce_types(df, schema)
    assert "date" in out.columns
