"""Tests for the CLI demo suite."""

from __future__ import annotations

import pytest

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


def test_supports_color_respects_environment_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Color support should disable on NO_COLOR/dumb TERM and enable for TTY output."""

    class _FakeStdout:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(demo_suite.sys, "stdout", _FakeStdout())
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert demo_suite._supports_color() is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert demo_suite._supports_color() is False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert demo_suite._supports_color() is False


def test_colorize_and_format_status_cover_all_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status formatting should colorize PASS/FAIL/SKIP when the terminal supports it."""
    monkeypatch.setattr(demo_suite, "_supports_color", lambda: True)

    assert demo_suite._colorize("ok", demo_suite._ANSI_GREEN).startswith(
        demo_suite._ANSI_GREEN
    )
    assert demo_suite._format_status("PASS").startswith(demo_suite._ANSI_GREEN)
    assert demo_suite._format_status("FAIL").startswith(demo_suite._ANSI_RED)
    assert demo_suite._format_status("SKIP").startswith(demo_suite._ANSI_YELLOW)


def test_run_demo_reports_pass_fail_and_skip() -> None:
    """The demo runner should classify successful, skipped and failed demos."""
    passed = demo_suite._run_demo("pass-demo", lambda: "done")
    skipped = demo_suite._run_demo(
        "skip-demo", lambda: (_ for _ in ()).throw(ImportError("optional"))
    )
    failed = demo_suite._run_demo(
        "fail-demo", lambda: (_ for _ in ()).throw(RuntimeError("broken"))
    )

    assert passed.status == "PASS"
    assert passed.details == "done"
    assert skipped.status == "SKIP"
    assert "optional" in skipped.details
    assert failed.status == "FAIL"
    assert failed.details == "broken"


def test_print_catalog_lists_all_and_summary_renders_totals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catalog and summary printers should render stable text output without ANSI noise."""
    monkeypatch.setattr(demo_suite, "_supports_color", lambda: False)

    demo_suite._print_catalog()
    demo_suite._print_summary(
        [
            demo_suite.DemoResult("config", "PASS", "ok", 12.3),
            demo_suite.DemoResult("llm", "FAIL", "missing", 45.6),
            demo_suite.DemoResult("rag", "SKIP", "optional", 7.8),
        ]
    )

    output = capsys.readouterr().out
    assert "[DEMO] Available Demos" in output
    assert "- all          Run all demos and print a summary table" in output
    assert "[DEMO] Demo Summary" in output
    assert "TOTAL: 3 | PASS=1 FAIL=1 SKIP=1" in output


def test_run_selected_returns_failure_when_any_demo_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_selected should surface a non-zero exit code when at least one demo fails."""
    monkeypatch.setattr(
        demo_suite,
        "DEMO_CATALOG",
        {
            "ok": ("Passing demo", lambda: "done"),
            "boom": (
                "Failing demo",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        },
    )

    assert demo_suite.run_selected(["ok", "boom"]) == 1


def test_main_list_and_invalid_demo_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI main should print the catalog for list and reject unknown demos."""
    monkeypatch.setattr(demo_suite, "_supports_color", lambda: False)

    assert demo_suite.main(["list"]) == 0
    assert "Available Demos" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        demo_suite.main(["run", "unknown-demo"])

    assert exc.value.code == 2


def test_demo_config_and_registry_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config and registry demos should raise on empty embedding model or empty registry."""
    monkeypatch.setattr(
        demo_suite,
        "get_settings",
        lambda: type(
            "_Settings",
            (),
            {"embedding_model": "", "models_dir": "models", "api_key": None},
        )(),
    )

    with pytest.raises(ValueError, match="embedding_model is empty"):
        demo_suite.demo_config()

    monkeypatch.setattr(demo_suite, "list_models", lambda: [])

    with pytest.raises(ValueError, match="No registered models found"):
        demo_suite.demo_llm_registry()
