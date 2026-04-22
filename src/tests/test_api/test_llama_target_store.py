"""Tests for persisted llama target storage."""

from __future__ import annotations

import asyncio
import json

from ai_engine.api.llama_target_store import LlamaTargetStore, PersistedLlamaTarget


def _run(coro):
    return asyncio.run(coro)


def test_load_returns_none_when_file_is_missing(tmp_path) -> None:
    store = LlamaTargetStore(str(tmp_path / "runtime" / "llama-target.json"))

    assert _run(store.load()) is None


def test_load_returns_none_for_invalid_payload_shapes(tmp_path) -> None:
    path = tmp_path / "llama-target.json"
    path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
    store = LlamaTargetStore(str(path))
    assert _run(store.load()) is None

    path.write_text(json.dumps({"url": 123, "updatedAt": None}), encoding="utf-8")
    assert _run(store.load()) is None


def test_load_normalizes_blank_label_to_none(tmp_path) -> None:
    path = tmp_path / "llama-target.json"
    path.write_text(
        json.dumps(
            {
                "url": "http://localhost:8080",
                "label": "   ",
                "updatedAt": "2026-04-22T19:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    store = LlamaTargetStore(str(path))

    loaded = _run(store.load())

    assert loaded == PersistedLlamaTarget(
        url="http://localhost:8080",
        label=None,
        updated_at="2026-04-22T19:00:00Z",
    )


def test_save_creates_parent_directories_and_reset_is_idempotent(tmp_path) -> None:
    path = tmp_path / "runtime" / "targets" / "llama-target.json"
    store = LlamaTargetStore(str(path))

    _run(
        store.save(
            PersistedLlamaTarget(
                url="http://localhost:11434",
                label="Local llama",
                updated_at="2026-04-22T19:05:00Z",
            )
        )
    )

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "url": "http://localhost:11434",
        "label": "Local llama",
        "updatedAt": "2026-04-22T19:05:00Z",
    }

    _run(store.reset())
    assert not path.exists()

    _run(store.reset())
