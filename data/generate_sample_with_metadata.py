from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.generator import (
    generate_joint_dataset,
    load_yaml,
)  # adjust import to your actual location

DATASET_VERSION = "0.1.1"  # bump when schema/scenarios change


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _best_effort_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def main(config_path: str = "data/samples/sample_config.yaml") -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / config_path).resolve()
    cfg = load_yaml(config_path)

    # Generate dataframe (deterministic given cfg["dataset"]["seed"])
    df: pd.DataFrame = generate_joint_dataset(cfg)

    # Write sample.csv
    out_csv = repo_root / cfg.get("output", {}).get("path", "data/samples/joint_info_sample.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Write metadata sidecar
    meta_path = out_csv.with_suffix(".meta.json")
    config_text = cfg_path.read_bytes()
    meta: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": _best_effort_git_commit(repo_root),
        "config_path": str(Path(config_path)),
        "config_sha256": _sha256_bytes(config_text),
        "output_csv": str(out_csv.relative_to(repo_root)),
        "output_csv_sha256": _sha256_file(out_csv),
        "seeds": {"dataset_seed": int(cfg["dataset"]["seed"])},
        "params": cfg,  # OK for synthetic; if you prefer, store only selected sections
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
