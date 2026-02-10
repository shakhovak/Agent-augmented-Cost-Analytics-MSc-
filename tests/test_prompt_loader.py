from __future__ import annotations

from pathlib import Path

import pytest

from cost_agent_mvp.agent.prompt_loader import PromptLoader


def _write_bundle(dir_path: Path, name: str, *, system: str, user: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}_system.md").write_text(system, encoding="utf-8")
    (dir_path / f"{name}_user.md").write_text(user, encoding="utf-8")


def test_load_bundle_reads_files(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    _write_bundle(
        prompts_dir,
        "analyst",
        system="SYSTEM\nMode={{MODE}}\n",
        user="USER\nQuestion={{USER_QUESTION}}\n",
    )

    loader = PromptLoader(prompts_dir=str(prompts_dir))
    bundle = loader.load_bundle("analyst")

    assert "SYSTEM" in bundle.system
    assert "{{MODE}}" in bundle.system  # render not applied here
    assert "{{USER_QUESTION}}" in bundle.user


def test_load_and_render_substitutes_placeholders(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    _write_bundle(
        prompts_dir,
        "analyst",
        system="SYSTEM\nMode={{MODE}}\n",
        user="USER\nEvidence={{EVIDENCE}}\nQuestion={{USER_QUESTION}}\n",
    )

    loader = PromptLoader(prompts_dir=str(prompts_dir))
    out = loader.load_and_render(
        "analyst",
        system_vars={"MODE": "daily_report"},
        user_vars={"EVIDENCE": "kpi=123", "USER_QUESTION": "Why did it spike?"},
    )

    assert "{{MODE}}" not in out.system
    assert "Mode=daily_report" in out.system
    assert "Evidence=kpi=123" in out.user
    assert "Why did it spike?" in out.user


def test_render_missing_vars_is_clear_or_left_unresolved() -> None:
    """
    Spec allows either:
    - strict behavior: raise a clear exception for missing variables, OR
    - permissive behavior: leave unresolved placeholders (documented behavior).
    This test passes for both implementations.
    """
    template = "A={{A}} B={{B}}"
    try:
        rendered = PromptLoader.render(template, {"A": "1"})
    except KeyError:
        # Strict implementation is acceptable.
        return

    # Permissive implementation: unresolved placeholders remain.
    assert "A=1" in rendered
    assert "{{B}}" in rendered


def test_load_bundle_missing_file_raises(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Only system exists; user file missing
    (prompts_dir / "analyst_system.md").write_text("SYSTEM", encoding="utf-8")

    loader = PromptLoader(prompts_dir=str(prompts_dir))

    with pytest.raises(FileNotFoundError) as e:
        loader.load_bundle("analyst")

    assert "Prompt file not found" in str(e.value)


def test_render_allows_none_values_as_empty_string() -> None:
    template = "X{{A}}Y"
    # Current implementation treats None as empty string; keep regression coverage.
    rendered = PromptLoader.render(template, {"A": None})  # type: ignore[arg-type]
    assert rendered == "XY"
