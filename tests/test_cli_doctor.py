from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_doctor_writes_run_record(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["COST_AGENT_OUTPUTS_DIR"] = str(tmp_path / "outputs")

    # Run: python -m cost_agent_mvp.cli doctor
    result = subprocess.run(
        [sys.executable, "-m", "cost_agent_mvp.cli", "doctor"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr

    runs_dir = Path(env["COST_AGENT_OUTPUTS_DIR"]) / "runs"
    assert runs_dir.exists()

    # Find run_record.json somewhere under runs/<run_id>/
    run_records = list(runs_dir.glob("*/run_record.json"))
    assert len(run_records) >= 1, f"No run_record.json found. stdout:\n{result.stdout}"
