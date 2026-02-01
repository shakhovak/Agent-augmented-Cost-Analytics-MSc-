from __future__ import annotations

import os
import sys
from pathlib import Path

from cost_agent_mvp.core.run_record import (
    RunRecord,
    ensure_run_dir,
    generate_run_id,
    write_run_record,
)


def _package_version() -> str:
    # Prefer importlib.metadata so it works when installed
    try:
        from importlib.metadata import version

        return version("agent-augmented-cost-analytics")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    # Output directory override for tests/CI
    outputs_base = Path(os.environ.get("COST_AGENT_OUTPUTS_DIR", "outputs")).resolve()

    run_id = generate_run_id(prefix="doctor")
    run_dir = ensure_run_dir(outputs_base, run_id)

    # Check presence of env vars (don’t print values)
    env_keys = ["OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"]
    env_presence = {k: bool(os.environ.get(k)) for k in env_keys}

    print("cost_agent_mvp doctor")
    print(f"python: {sys.version.split()[0]}")
    print(f"package_version: {_package_version()}")
    for k, present in env_presence.items():
        print(f"env:{k}: {present}")

    record = RunRecord(
        run_id=run_id,
        mode="deterministic",
        input_path=None,
        report_day=None,
    )
    # Include doctor info as lightweight “artifacts/config” placeholders
    record.config_hashes["doctor_env_presence"] = str(env_presence)
    record.artifacts["run_record"] = str((run_dir / "run_record.json").resolve())

    path = write_run_record(run_dir, record)
    print(f"run_record: {path}")

    return 0
