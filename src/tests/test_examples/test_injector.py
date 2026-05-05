"""Tests for the example injection system."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai_engine.examples.corpus import (
    EDUCATIONAL_RESOURCES,
    QUIZ_EXAMPLES,
    TRUE_FALSE_EXAMPLES,
    WORD_PASS_EXAMPLES,
    get_full_corpus,
)
from ai_engine.examples.injector import ExampleInjector
from ai_engine.examples.rag_seed import load_seed_corpus_entries
from ai_engine.rag.document import Document

# ── Corpus integrity ──────────────────────────────────────────────────


class TestCorpusIntegrity:
    """Validate every entry in the corpus is well-formed."""

    def test_full_corpus_is_non_empty(self):
        corpus = get_full_corpus()
        assert len(corpus) > 0

    def test_full_corpus_equals_sum_of_parts(self):
        source_entries = (
            QUIZ_EXAMPLES
            + WORD_PASS_EXAMPLES
            + TRUE_FALSE_EXAMPLES
            + EDUCATIONAL_RESOURCES
            + load_seed_corpus_entries()
        )
        total = sum(
            1
            for entry in source_entries
            if entry.get("metadata", {}).get("language") == "en"
        )
        assert len(get_full_corpus()) == total

    @pytest.mark.parametrize(
        "entry", get_full_corpus(), ids=lambda e: e.get("doc_id", "?")
    )
    def test_each_entry_has_content_and_doc_id(self, entry):
        assert "content" in entry
        assert isinstance(entry["content"], str)
        assert len(entry["content"]) > 0
        assert "doc_id" in entry

    @pytest.mark.parametrize(
        "entry", get_full_corpus(), ids=lambda e: e.get("doc_id", "?")
    )
    def test_each_entry_has_metadata(self, entry):
        meta = entry.get("metadata", {})
        assert "kind" in meta
        assert meta["kind"] in ("game_example", "educational_resource")

    @pytest.mark.parametrize(
        "entry", get_full_corpus(), ids=lambda e: e.get("doc_id", "?")
    )
    def test_doc_ids_are_unique(self, entry):
        """Collect all doc_ids and assert no duplicates."""
        # This test runs per-entry but we do a global check.
        all_ids = [e["doc_id"] for e in get_full_corpus()]
        assert (
            all_ids.count(entry["doc_id"]) == 1
        ), f"duplicate doc_id: {entry['doc_id']}"


class TestGameExamples:
    """Validate game example structures match the expected schemas."""

    @pytest.mark.parametrize("entry", QUIZ_EXAMPLES, ids=lambda e: e.get("doc_id", "?"))
    def test_quiz_example_contains_valid_json(self, entry):
        meta = entry["metadata"]
        assert meta["game_type"] == "quiz"
        # Extract JSON from content (between ```json and ```)
        content = entry["content"]
        json_start = content.index("```json\n") + len("```json\n")
        json_end = content.index("\n```", json_start)
        game = json.loads(content[json_start:json_end])
        assert game["game_type"] == "quiz"
        assert len(game["questions"]) >= 3
        for q in game["questions"]:
            assert "question" in q
            assert len(q["options"]) == 4
            assert 0 <= q["correct_index"] <= 3

    @pytest.mark.parametrize(
        "entry", WORD_PASS_EXAMPLES, ids=lambda e: e.get("doc_id", "?")
    )
    def test_word_pass_example_contains_valid_json(self, entry):
        meta = entry["metadata"]
        assert meta["game_type"] == "word-pass"
        content = entry["content"]
        json_start = content.index("```json\n") + len("```json\n")
        json_end = content.index("\n```", json_start)
        game = json.loads(content[json_start:json_end])
        assert game["game_type"] == "word-pass"
        assert len(game["words"]) >= 10
        for w in game["words"]:
            assert "letter" in w
            assert "hint" in w
            assert "answer" in w
            assert "starts_with" in w

    @pytest.mark.parametrize(
        "entry", TRUE_FALSE_EXAMPLES, ids=lambda e: e.get("doc_id", "?")
    )
    def test_true_false_example_contains_valid_json(self, entry):
        meta = entry["metadata"]
        assert meta["game_type"] == "true_false"
        content = entry["content"]
        json_start = content.index("```json\n") + len("```json\n")
        json_end = content.index("\n```", json_start)
        game = json.loads(content[json_start:json_end])
        assert game["game_type"] == "true_false"
        assert len(game["statements"]) >= 3
        for s in game["statements"]:
            assert "statement" in s
            assert "is_true" in s


# ── Injector ──────────────────────────────────────────────────────────


class TestExampleInjector:
    """Test the ExampleInjector class."""

    def _mock_pipeline(self) -> MagicMock:
        pipeline = MagicMock()
        pipeline.ingest = MagicMock()
        return pipeline

    def test_inject_all_calls_pipeline_ingest(self):
        pipeline = self._mock_pipeline()
        injector = ExampleInjector(pipeline)
        count = injector.inject_all()
        assert count > 0
        pipeline.ingest.assert_called_once()
        docs = pipeline.ingest.call_args[0][0]
        assert all(isinstance(d, Document) for d in docs)
        assert len(docs) == count

    def test_inject_by_kind_game_example(self):
        pipeline = self._mock_pipeline()
        injector = ExampleInjector(pipeline)
        count = injector.inject_by_kind("game_example")
        assert count > 0
        docs = pipeline.ingest.call_args[0][0]
        assert all(d.metadata["kind"] == "game_example" for d in docs)

    def test_inject_by_kind_educational_resource(self):
        pipeline = self._mock_pipeline()
        injector = ExampleInjector(pipeline)
        count = injector.inject_by_kind("educational_resource")
        assert count > 0
        docs = pipeline.ingest.call_args[0][0]
        assert all(d.metadata["kind"] == "educational_resource" for d in docs)

    def test_inject_by_game_type(self):
        pipeline = self._mock_pipeline()
        injector = ExampleInjector(pipeline)
        count = injector.inject_by_game_type("quiz")
        assert count > 0
        docs = pipeline.ingest.call_args[0][0]
        assert all(d.metadata.get("game_type") == "quiz" for d in docs)

    def test_inject_by_game_type_no_match(self):
        pipeline = self._mock_pipeline()
        injector = ExampleInjector(pipeline)
        count = injector.inject_by_game_type("nonexistent")
        assert count == 0
        pipeline.ingest.assert_not_called()
