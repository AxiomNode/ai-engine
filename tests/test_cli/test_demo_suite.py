"""Tests for the CLI demo suite."""

from __future__ import annotations

from ai_engine.cli import demo_suite


def test_catalog_has_expected_keys() -> None:
    """The demo catalog should expose all expected demo entries."""
    expected = {
        "config",
        "llm",
        "kbd",
        "rag",
        "games",
        "observability",
        "api",
        "integration",
    }
    assert expected.issubset(set(demo_suite.DEMO_CATALOG.keys()))


def test_run_selected_single_demo_returns_success() -> None:
    """Running a lightweight demo should return zero exit code."""
    exit_code = demo_suite.run_selected(["config"])
    assert exit_code == 0


def test_main_run_all_returns_success() -> None:
    """Running all demos should succeed in the default test environment."""
    exit_code = demo_suite.main(["run", "all"])
    assert exit_code == 0
