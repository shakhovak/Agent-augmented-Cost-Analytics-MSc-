import hashlib
from pathlib import Path

import pandas as pd

from cost_agent_mvp.data.schema import default_joint_schema, validate_columns
from data.generator import generate_joint_dataset, load_yaml


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_generate_sample_data_writes_file_and_is_deterministic(tmp_path: Path):
    out_csv_1 = tmp_path / "sample_1.csv"
    out_csv_2 = tmp_path / "sample_2.csv"
    cfg_path = "data/samples/sample_config.yaml"

    cfg = load_yaml(Path(cfg_path))

    # Run twice with same seed
    df1 = generate_joint_dataset(cfg)
    df1.to_csv(out_csv_1, index=False)

    df2 = generate_joint_dataset(cfg)
    df2.to_csv(out_csv_2, index=False)

    # File exists and non-empty
    assert out_csv_1.exists()
    assert out_csv_1.stat().st_size > 0

    # Determinism: same seed -> same hash (CSV bytes)
    h1 = _sha256_bytes(out_csv_1.read_bytes())
    h2 = _sha256_bytes(out_csv_2.read_bytes())
    assert h1 == h2

    # Validate required columns match schema
    schema = default_joint_schema()
    validate_columns(df1, schema)  # fails fast on missing required columns
    assert set(schema.required_columns).issubset(set(df1.columns))

    # Validate the outpiut file is not empty
    df_check = pd.read_csv(out_csv_1)
    assert len(df_check) > 0
