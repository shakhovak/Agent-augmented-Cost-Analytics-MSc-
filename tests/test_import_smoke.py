"""
Smoke test: verify that the package is installable and importable.

This test intentionally does NOT modify sys.path.
It should pass only if the project is installed (e.g., `pip install -e .`)
and packaging configuration includes the package code.

Milestone: M0
Issue: M0-01
"""

from __future__ import annotations

import importlib
from importlib import metadata


# 1) PACKAGE_NAME = your import name (folder under src/)
# Example: src/cost_agent_mvp/__init__.py  -> PACKAGE_NAME = "cost_agent_mvp"
PACKAGE_NAME = "cost_agent_mvp"

# 2) DIST_NAME = the distribution name from [project].name in pyproject.toml
# Example: name = "agent-augmented-cost-analytics" -> DIST_NAME = "agent-augmented-cost-analytics"
DIST_NAME = "agent-augmented-cost-analytics"


def test_import_package() -> None:
    """Package can be imported after installation."""
    module = importlib.import_module(PACKAGE_NAME)
    assert module is not None


def test_distribution_version_available() -> None:
    """
    Distribution metadata is available (proves the package is installed).

    This does NOT require __version__ inside the package; it reads installed metadata.
    """
    version = metadata.version(DIST_NAME)
    assert isinstance(version, str)
    assert version.strip() != ""


def test_package_version_attribute_optional() -> None:
    """
    Optional check: if you expose __version__ in your package, verify it looks sane.

    If you do NOT expose __version__, this test will be skipped gracefully.
    """
    module = importlib.import_module(PACKAGE_NAME)

    if not hasattr(module, "__version__"):
        # This is fine—version can be taken from importlib.metadata instead.
        return

    value = getattr(module, "__version__")
    assert isinstance(value, str)
    assert value.strip() != ""
