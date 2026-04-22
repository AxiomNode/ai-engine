"""Tests for lightweight import surfaces and module entry points."""

from __future__ import annotations

import importlib
import runpy


def test_llm_module_entrypoint_invokes_cli(monkeypatch) -> None:
    called = {"value": False}

    def _fake_cli() -> None:
        called["value"] = True

    monkeypatch.setattr("ai_engine.llm.model_manager._cli", _fake_cli)

    runpy.run_module("ai_engine.llm.__main__", run_name="__main__")

    assert called["value"] is True


def test_kbd_init_exports_expected_symbols() -> None:
    module = importlib.import_module("ai_engine.kbd")

    assert module.KnowledgeEntry.__name__ == "KnowledgeEntry"
    assert module.KnowledgeBase.__name__ == "KnowledgeBase"
    assert hasattr(module, "TinyDBKnowledgeBase")


def test_rag_init_exports_expected_symbols() -> None:
    module = importlib.import_module("ai_engine.rag")

    assert module.Document.__name__ == "Document"
    assert module.RAGPipeline.__name__ == "RAGPipeline"
    assert hasattr(module, "ChromaVectorStore")


def test_vectorstore_modules_export_chroma_symbol() -> None:
    legacy = importlib.import_module("ai_engine.rag.vectorstore")
    modern = importlib.import_module("ai_engine.rag.vectorstores")

    assert hasattr(legacy, "ChromaVectorStore")
    assert hasattr(modern, "ChromaVectorStore")
