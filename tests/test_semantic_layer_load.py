from __future__ import annotations

from pathlib import Path

import yaml


def test_semantic_layer_load_and_basic_schema() -> None:
    path = Path("configs/semantic_layer.yaml")
    assert path.exists(), "configs/semantic_layer.yaml is missing"

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    # required top-level keys
    for k in (
        "version",
        "dataset",
        "dimensions",
        "measures",
        "allowed_operations",
        "limits",
    ):
        assert k in cfg, f"Missing required key: {k}"

    assert isinstance(cfg["dimensions"], list) and len(cfg["dimensions"]) > 0
    assert isinstance(cfg["measures"], list) and len(cfg["measures"]) > 0

    # sanity: ensure core fields exist
    ds = cfg["dataset"]
    assert "name" in ds and "date_column" in ds

    limits = cfg["limits"]
    assert limits["max_days_hard"] >= limits["max_days_default"]
    assert limits["max_rows_hard"] >= limits["max_rows_default"]

    ops = cfg["allowed_operations"]
    assert "filters" in ops and "group_by" in ops and "aggregations" in ops
