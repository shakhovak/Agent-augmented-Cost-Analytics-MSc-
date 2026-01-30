from __future__ import annotations

from pathlib import Path

from cost_agent_mvp.core.run_record import (
    RunRecord,
    ensure_run_dir,
    generate_run_id,
    read_run_record,
    write_run_record,
)


def test_run_record_write_and_read(tmp_path: Path) -> None:
    base_outputs = tmp_path / "outputs"
    run_id = generate_run_id(prefix="test")
    run_dir = ensure_run_dir(base_outputs, run_id)

    record = RunRecord(
        run_id=run_id,
        mode="deterministic",
        input_path="data/example.csv",
        report_day="2026-01-30",
    )

    record_path = write_run_record(run_dir, record)
    assert record_path.exists()

    loaded = read_run_record(record_path)

    # Required fields exist and match expectations
    assert loaded.run_id == run_id
    assert loaded.mode == "deterministic"
    assert loaded.timestamp  # non-empty
    assert loaded.python_version
    # git_commit is best-effort and may be None in some test environments
