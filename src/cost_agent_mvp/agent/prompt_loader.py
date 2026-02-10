from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Z0-9_]+)\s*}}")


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


class PromptLoader:
    """
    Loads markdown prompt templates from disk and renders them with placeholders.

    Placeholders convention: {{PLACEHOLDER_NAME}} inside markdown.
    """

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        if prompts_dir is None:
            # Stable default regardless of working directory
            prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

    def _read(self, filename: str) -> str:
        key = str(self.prompts_dir / filename)
        if key in self._cache:
            return self._cache[key]

        path = self.prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        text = path.read_text(encoding="utf-8")
        self._cache[key] = text
        return text

    @staticmethod
    def render(template: str, vars: dict[str, str] | None = None, *, strict: bool = True) -> str:
        vars = vars or {}
        placeholders = set(_PLACEHOLDER_RE.findall(template))

        missing = sorted(placeholders - set(vars.keys()))
        if strict and missing:
            raise KeyError(f"Missing placeholder variables: {missing}")

        out = template
        for k, v in vars.items():
            out = out.replace(f"{{{{{k}}}}}", "" if v is None else str(v))

        # After replacement, ensure nothing unresolved remains
        if strict:
            unresolved = sorted(set(_PLACEHOLDER_RE.findall(out)))
            if unresolved:
                raise KeyError(f"Unresolved placeholders remain: {unresolved}")

        return out

    def load_bundle(self, name: str) -> PromptBundle:
        """
        name examples:
          - "planner" -> planner_system.md + planner_user.md
          - "analyst" -> analyst_system.md + analyst_user.md
        """
        system = self._read(f"{name}_system.md")
        user = self._read(f"{name}_user.md")
        return PromptBundle(system=system, user=user)

    def load_and_render(
        self,
        name: str,
        user_vars: dict[str, str] | None = None,
        system_vars: dict[str, str] | None = None,
    ) -> PromptBundle:
        bundle = self.load_bundle(name)
        return PromptBundle(
            system=self.render(bundle.system, system_vars),
            user=self.render(bundle.user, user_vars),
        )
