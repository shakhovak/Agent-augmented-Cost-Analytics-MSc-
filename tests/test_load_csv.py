import pandas as pd
import pytest

from cost_agent_mvp.core.errors import DataSourceError, ValidationError
from cost_agent_mvp.data.schema import default_joint_schema, load_csv


def _write_csv(path, df: pd.DataFrame) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _valid_min_df():
    """
    Build a minimal valid DataFrame that satisfies default_joint_schema().
    """
    schema = default_joint_schema()
    # start with required cols, fill with safe defaults
    data = {}
    for c in schema.required_columns:
        if c == "date":
            data[c] = ["2026-01-01"]
        elif c in ("chat_id", "chat_type"):
            data[c] = ["x"]
        else:
            data[c] = [0]

    # Ensure at least one numeric column is a string -> should coerce
    data["total_cost"] = ["12.34"]  # string to float
    # Ensure at least one int column is a string -> should coerce
    data["account_id"] = ["123"]  # string to Int64

    return pd.DataFrame(data)


def test_load_csv_valid_success_and_dtypes(tmp_path):
    df = _valid_min_df()
    p = tmp_path / "sample.csv"
    _write_csv(p, df)

    out = load_csv(str(p))

    # date parsed to python datetime.date
    assert out["date"].iloc[0].__class__.__name__ == "date"

    # numeric -> float64
    assert str(out["total_cost"].dtype) == "float64"
    assert out["total_cost"].iloc[0] == pytest.approx(12.34)

    # int -> pandas nullable Int64
    assert str(out["account_id"].dtype) == "Int64"
    assert int(out["account_id"].iloc[0]) == 123

    # strings -> pandas string
    assert str(out["chat_id"].dtype) == "string"
    assert str(out["chat_type"].dtype) == "string"


def test_load_csv_missing_required_columns_raises(tmp_path):
    df = _valid_min_df().drop(columns=["total_cost"])
    p = tmp_path / "missing_col.csv"
    _write_csv(p, df)

    with pytest.raises(DataSourceError) as e:
        load_csv(str(p))

    assert "missing required columns" in str(e.value).lower()
    assert "total_cost" in str(e.value)


def test_load_csv_invalid_date_raises(tmp_path):
    df = _valid_min_df()
    df.loc[0, "date"] = "not-a-date"
    p = tmp_path / "bad_date.csv"
    _write_csv(p, df)

    with pytest.raises(ValidationError) as e:
        load_csv(str(p))

    assert "failed to parse" in str(e.value).lower()
    assert "date" in str(e.value).lower()


def test_load_csv_invalid_numeric_raises(tmp_path):
    df = _valid_min_df()
    df.loc[0, "total_cost"] = "abc"
    p = tmp_path / "bad_numeric.csv"
    _write_csv(p, df)

    with pytest.raises(ValidationError) as e:
        load_csv(str(p))

    msg = str(e.value).lower()
    assert "non-numeric" in msg
    assert "total_cost" in msg


def test_load_csv_missing_file_raises():
    with pytest.raises(DataSourceError):
        load_csv("this_file_does_not_exist_12345.csv")
