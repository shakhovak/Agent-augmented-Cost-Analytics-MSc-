from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REQUIRED_LABEL_FIELDS = {
    "event_id",
    "scenario_id",
    "seed",
    "anomaly_type",
    "affected_services",
    "start_date",
    "end_date",
    "driver_template",
}


def _repo_root() -> Path:
    # tests/ is at repo_root/tests/
    return Path(__file__).resolve().parents[1]


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    import sys

    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader, f"Could not load module spec from {path}"
    mod = importlib.util.module_from_spec(spec)

    # Important: register before exec so decorators (dataclass) can resolve cls.__module__
    sys.modules[name] = mod

    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_generator_config() -> Path:
    root = _repo_root()
    # generator default is configs/synth_2025.yaml (per generator.py)
    candidates = [
        root / "data" / "config_data_generator.yaml",
        root / "config_data_generator.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find generator config in: {candidates}")


def _make_small_generator_cfg(generator_mod, base_cfg_path: Path, out_csv: Path) -> dict:
    # Use generator's YAML loader if present; otherwise parse with PyYAML via generator.load_yaml
    cfg = generator_mod.load_yaml(str(base_cfg_path))

    # Make it fast for tests:
    # - fewer accounts
    # - shorter horizon
    cfg["dataset"]["seed"] = 123
    cfg["accounts"]["n_accounts"] = 50

    # use only first 30 days of 2025 (generator supports overriding via cfg directly)
    cfg["dataset"]["start_date"] = "2025-01-01"
    cfg["dataset"]["end_date"] = "2025-01-30"

    cfg.setdefault("output", {})
    cfg["output"]["path"] = str(out_csv)

    return cfg


def _write_min_anomaly_yaml(path: Path) -> None:
    # Keep dates within the test generator window (Jan 2025)
    text = """\
version: 0.1
base_seed: 123
dataset:
  max_concurrent_per_account: 50

scenarios:
  - scenario_id: test_cost_spike
    anomaly_type: cost_spike
    seed: 1001
    account_selection: { mode: random_k, k: 5 }
    window: { start_date: 2025-01-10, duration_days: 2 }
    params: { component: dialog, magnitude: 3.0 }
    driver_template: "Unit-cost spike in dialog."

  - scenario_id: test_volume_spike
    anomaly_type: volume_spike
    seed: 1002
    account_selection: { mode: random_k, k: 8 }
    window: { start_date: 2025-01-15, duration_days: 3 }
    params: { component: tasks, magnitude: 2.0, min_base_count: 3 }
    driver_template: "Usage spike in tasks."

  - scenario_id: test_cap_spike
    anomaly_type: cap_spike
    seed: 1003
    account_selection: { mode: random_k, k: 2 }
    window: { start_date: 2025-01-20, duration_days: 1 }
    params: { exceed_by: 5 }
    driver_template: "Cap violation spike."
"""
    path.write_text(text, encoding="utf-8")


def _read_labels_jsonl(path: Path) -> list[dict]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(json.loads(line))
    return labels


def test_anomaly_injection_end_to_end(tmp_path: Path):
    root = _repo_root()

    # --- load generator module from file path (generator.py may live in data/ or elsewhere)
    candidate_paths = [
        root / "generator.py",
        root / "data" / "generator.py",
        root / "scripts" / "generator.py",
        root / "src" / "cost_agent_mvp" / "data" / "generator.py",
    ]

    generator_path = next((p for p in candidate_paths if p.exists()), None)
    if generator_path is None:
        pytest.fail(f"Could not find generator.py in: {candidate_paths}")

    generator_mod = _load_module_from_path("generator", generator_path)

    assert hasattr(
        generator_mod, "generate_joint_dataset"
    ), "generate_joint_dataset() not found in generator.py"
    assert hasattr(generator_mod, "load_yaml"), "load_yaml() not found in generator.py"

    generate_joint_dataset = generator_mod.generate_joint_dataset
    load_yaml = generator_mod.load_yaml

    # make sure these are present
    assert callable(generate_joint_dataset)
    assert callable(load_yaml)

    # --- import injector from scripts/inject_anomalies.py (script may not be a package)
    injector_path = root / "data" / "inject_anomalies.py"
    if not injector_path.exists():
        pytest.fail(f"Missing injector script at {injector_path}")

    injector_mod = _load_module_from_path("inject_anomalies", injector_path)
    assert hasattr(injector_mod, "inject_anomalies"), "inject_anomalies() not found in script"
    inject_anomalies = injector_mod.inject_anomalies

    # --- generate base data into tmp_path
    base_csv = tmp_path / "base.csv"
    cfg_path = _find_generator_config()
    cfg = _make_small_generator_cfg(generator_mod, cfg_path, base_csv)

    df_base = generate_joint_dataset(cfg)
    df_base.to_csv(base_csv, index=False)
    assert base_csv.exists(), "Base CSV was not created"
    assert len(df_base) > 0, "Base dataset is unexpectedly empty"

    # --- write anomaly config + run injection
    anomaly_cfg = tmp_path / "anomaly_scenarios.yaml"
    _write_min_anomaly_yaml(anomaly_cfg)

    out_csv = tmp_path / "sample_with_anomalies.csv"
    labels_path = tmp_path / "labels.jsonl"

    inject_anomalies(
        input_csv=str(base_csv),
        config_path=str(anomaly_cfg),
        output_csv=str(out_csv),
        labels_jsonl=str(labels_path),
    )

    assert out_csv.exists(), "Anomalies output CSV was not created"
    assert labels_path.exists(), "Labels JSONL was not created"

    # --- basic label schema checks
    labels = _read_labels_jsonl(labels_path)
    assert len(labels) > 0, "No labels were produced"

    for ev in labels:
        missing = REQUIRED_LABEL_FIELDS - set(ev.keys())
        assert not missing, f"Label missing required fields: {missing}"

        assert isinstance(ev["affected_services"], list), "affected_services should be a JSON list"
        # sanity: isoformat dates
        pd.to_datetime(ev["start_date"], errors="raise")
        pd.to_datetime(ev["end_date"], errors="raise")

    # --- consistency: each label window exists within data range
    df_out = pd.read_csv(out_csv)
    assert "date" in df_out.columns, "Output CSV missing 'date' column"
    dates = pd.to_datetime(df_out["date"], errors="raise").dt.date

    data_min = dates.min()
    data_max = dates.max()
    assert data_min is not None and data_max is not None

    for ev in labels:
        s = pd.to_datetime(ev["start_date"]).date()
        e = pd.to_datetime(ev["end_date"]).date()
        assert (
            data_min <= s <= data_max
        ), f"Label start_date {s} outside data range [{data_min}, {data_max}]"
        assert (
            data_min <= e <= data_max
        ), f"Label end_date {e} outside data range [{data_min}, {data_max}]"
        assert s <= e, "Label has start_date after end_date"
