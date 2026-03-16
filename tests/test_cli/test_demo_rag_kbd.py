"""Unit tests for the RAG+KBD demo CLI module."""

from __future__ import annotations

import sys
import types

import pytest

from ai_engine.cli import demo_rag_kbd


def test_locate_model_path_uses_first_available_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should return the first model path that exists in priority order."""

    calls: list[str] = []

    def fake_model_path(name: str) -> str:
        calls.append(name)
        if name == "qwen2.5-7b":
            raise FileNotFoundError("missing")
        return "models/phi.gguf"

    fake_model_manager = types.SimpleNamespace(
        MODELS={"qwen2.5-7b": object(), "phi-3.5-mini": object()},
        model_path=fake_model_path,
    )
    monkeypatch.setitem(sys.modules, "ai_engine.llm.model_manager", fake_model_manager)

    resolved = demo_rag_kbd.locate_model_path()

    assert resolved == "models/phi.gguf"
    assert calls == ["qwen2.5-7b", "phi-3.5-mini"]


def test_locate_model_path_raises_when_no_models_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should exit with status 1 when no local model can be resolved."""

    def always_missing(_: str) -> str:
        raise FileNotFoundError("missing")

    fake_model_manager = types.SimpleNamespace(
        MODELS={
            "qwen2.5-7b": object(),
            "phi-3.5-mini": object(),
            "qwen2.5-3b": object(),
        },
        model_path=always_missing,
    )
    monkeypatch.setitem(sys.modules, "ai_engine.llm.model_manager", fake_model_manager)

    with pytest.raises(SystemExit) as exc:
        demo_rag_kbd.locate_model_path()

    assert exc.value.code == 1


def test_main_orchestrates_all_stages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main should call every stage in order and return success code."""

    observed: list[str] = []
    fake_kb = object()
    fake_pipeline = object()

    def fake_locate_model_path() -> str:
        observed.append("locate")
        return "models/fake.gguf"

    def fake_run_kbd_demo() -> object:
        observed.append("kbd")
        return fake_kb

    def fake_run_rag_demo(kb: object) -> object:
        assert kb is fake_kb
        observed.append("rag")
        return fake_pipeline

    def fake_run_game_demo(model_path: str, pipeline: object) -> None:
        assert model_path == "models/fake.gguf"
        assert pipeline is fake_pipeline
        observed.append("game")

    monkeypatch.setattr(demo_rag_kbd, "locate_model_path", fake_locate_model_path)
    monkeypatch.setattr(demo_rag_kbd, "run_kbd_demo", fake_run_kbd_demo)
    monkeypatch.setattr(demo_rag_kbd, "run_rag_demo", fake_run_rag_demo)
    monkeypatch.setattr(demo_rag_kbd, "run_game_demo", fake_run_game_demo)

    assert demo_rag_kbd.main() == 0
    assert observed == ["locate", "kbd", "rag", "game"]
    assert "Demo complete." in capsys.readouterr().out
