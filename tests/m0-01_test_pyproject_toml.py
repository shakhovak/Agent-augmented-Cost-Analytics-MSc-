"""
Test script for validating pyproject.toml configuration.

Issue: M0-01
Purpose: Validate pyproject.toml syntax, structure, and basic package setup.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Get the project root directory (parent of tests/)
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path so we can import test_results_manager
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_results_manager import ConsolidatedResultsManager

PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
LOG_FILE = Path(__file__).with_suffix(".log")


class TestLogger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def log(self, message: str, status: str = "INFO") -> None:
        """Log a message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{status}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def record_test(self, test_name: str, passed: bool, message: str = "") -> None:
        """Record a test result."""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        self.log(f"TEST: {test_name} - {status} - {message}", status)

    def write_summary(self, results_manager: Any) -> None:
        """Write summary of all test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
        }

        self.log("=" * 60)
        self.log(f"SUMMARY: {passed}/{total} tests passed")
        if failed > 0:
            self.log(f"FAILED: {failed} test(s) failed", "FAIL")
        else:
            self.log("All tests passed!", "PASS")
        self.log("=" * 60)

        # Write to consolidated results file
        results_manager.add_test_suite_results(
            test_suite_name="pyproject_toml_validation",
            results=self.results,
            summary=summary,
        )
        results_manager.save()
        self.log(f"Results saved to consolidated file: {results_manager.get_results_file_path()}")


def _load_toml():
    """Helper to load TOML file with fallback options."""
    import sys
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Try Python 3.11+ tomllib first
    try:
        import tomllib

        with open(PYPROJECT_TOML, "rb") as f:
            return tomllib.load(f), f"tomllib (Python {python_version})"
    except ImportError:
        # Fallback to tomli for older Python versions
        try:
            import tomli

            with open(PYPROJECT_TOML, "rb") as f:
                return tomli.load(f), f"tomli (Python {python_version})"
        except ImportError:
            raise ImportError(
                f"No TOML parser available. Python version: {python_version}. "
                f"For Python < 3.11, install with: pip install tomli"
            )


def test_toml_syntax(logger: TestLogger) -> bool:
    """Test 1: Validate TOML syntax."""
    logger.log("Testing TOML syntax...")
    try:
        data, parser = _load_toml()
        logger.record_test("TOML Syntax", True, f"File is valid TOML (using {parser})")
        return True
    except ImportError as e:
        logger.record_test("TOML Syntax", False, str(e))
        return False
    except Exception as e:
        logger.record_test("TOML Syntax", False, f"Invalid TOML: {str(e)}")
        return False


def test_pyproject_structure(logger: TestLogger) -> bool:
    """Test 2: Validate pyproject.toml structure."""
    logger.log("Testing pyproject.toml structure...")
    try:
        data, _ = _load_toml()
    except ImportError as e:
        logger.record_test("Structure", False, str(e))
        return False

    required_sections = ["build-system", "project"]
    missing = [s for s in required_sections if s not in data]

    if missing:
        logger.record_test("Structure", False, f"Missing sections: {missing}")
        return False

    # Check project section
    project = data.get("project", {})
    required_fields = ["name", "version"]
    missing_fields = [f for f in required_fields if f not in project]

    if missing_fields:
        logger.record_test("Structure", False, f"Missing project fields: {missing_fields}")
        return False

    logger.record_test(
        "Structure",
        True,
        f"Project: {project.get('name')} v{project.get('version')}",
    )
    return True


def test_package_structure(logger: TestLogger) -> bool:
    """Test 3: Validate package structure (src/ exists and is importable)."""
    logger.log("Testing package structure...")
    src_dir = PROJECT_ROOT / "src"
    src_init = src_dir / "__init__.py"

    if not src_dir.exists():
        logger.record_test("Package Structure", False, "src/ directory not found")
        return False

    if not src_init.exists():
        logger.record_test("Package Structure", False, "src/__init__.py not found")
        return False

    # Try to import (add project root to path temporarily)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import src  # noqa: F401

        logger.record_test("Package Structure", True, "src package is importable")
        return True
    except ImportError as e:
        logger.record_test("Package Structure", False, f"Import failed: {str(e)}")
        return False
    finally:
        if str(PROJECT_ROOT) in sys.path:
            sys.path.remove(str(PROJECT_ROOT))


def test_build_system(logger: TestLogger) -> bool:
    """Test 4: Validate build system configuration."""
    logger.log("Testing build system configuration...")
    try:
        data, _ = _load_toml()
    except ImportError as e:
        logger.record_test("Build System", False, str(e))
        return False

    build_system = data.get("build-system", {})
    if "requires" not in build_system or "build-backend" not in build_system:
        logger.record_test("Build System", False, "Missing build-system configuration")
        return False

    backend = build_system.get("build-backend", "")
    if "hatchling" not in backend:
        logger.record_test(
            "Build System",
            False,
            f"Unexpected build backend: {backend}",
        )
        return False

    logger.record_test("Build System", True, f"Using backend: {backend}")
    return True


def test_entry_points_structure(logger: TestLogger) -> bool:
    """Test 5: Validate entry points structure (not functionality)."""
    logger.log("Testing entry points structure...")
    try:
        data, _ = _load_toml()
    except ImportError as e:
        logger.record_test("Entry Points", False, str(e))
        return False

    project = data.get("project", {})
    scripts = project.get("scripts", {})

    if not scripts:
        logger.record_test("Entry Points", True, "No entry points defined (optional)")
        return True

    # Check format: command = "module.path:function"
    for cmd, path in scripts.items():
        if ":" not in path:
            logger.record_test(
                "Entry Points",
                False,
                f"Invalid entry point format: {cmd} = {path}",
            )
            return False

    logger.record_test("Entry Points", True, f"Found {len(scripts)} entry point(s)")
    return True


def main() -> int:
    """Run all validation tests."""
    import sys
    
    # Initialize results manager for milestone M0, issue 01
    results_manager = ConsolidatedResultsManager(milestone="M0", issue="01")
    
    logger = TestLogger(LOG_FILE)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    logger.log("=" * 60)
    logger.log("pyproject.toml Validation Tests (Milestone M0, Issue 01)")
    logger.log(f"Python version: {python_version}")
    logger.log(f"Project root: {PROJECT_ROOT}")
    logger.log(f"pyproject.toml: {PYPROJECT_TOML}")
    logger.log(f"Log file: {LOG_FILE}")
    logger.log(f"Consolidated results: {results_manager.get_results_file_path()}")
    logger.log("=" * 60)

    # Run all tests
    tests = [
        test_toml_syntax,
        test_pyproject_structure,
        test_package_structure,
        test_build_system,
        test_entry_points_structure,
    ]

    for test_func in tests:
        try:
            test_func(logger)
        except Exception as e:
            logger.record_test(test_func.__name__, False, f"Test error: {str(e)}")

    # Write summary to consolidated results file
    logger.write_summary(results_manager)

    # Return exit code: 0 if all passed, 1 if any failed
    failed = sum(1 for r in logger.results if not r["passed"])
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
