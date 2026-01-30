from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RunMode = Literal["deterministic", "llm"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    # Example: run_20260130T120501Z
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def get_git_commit_hash() -> str | None:
    """Best-effort git commit hash. Returns None if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_python_version() -> str:
    return platform.python_version()


class RunRecord(BaseModel):
    run_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    mode: RunMode

    input_path: str | None = None
    report_day: str | None = None  # store as ISO date string if used later

    git_commit: str | None = Field(default_factory=get_git_commit_hash)
    python_version: str = Field(default_factory=get_python_version)

    config_hashes: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


def ensure_run_dir(base_outputs_dir: Path, run_id: str) -> Path:
    run_dir = base_outputs_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_record(run_dir: Path, record: RunRecord) -> Path:
    path = run_dir / "run_record.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_run_record(path: Path) -> RunRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord.model_validate(data)
