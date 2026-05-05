from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai_engine.examples.corpus import get_full_corpus
from ai_engine.examples.rag_seed import (
    RagSeedCorpusError,
    describe_seed_corpus,
    load_seed_corpus_entries,
    load_seed_documents,
)
from ai_engine.games.schemas import GameEnvelope

_JSON_BLOCK_RE = re.compile(r"```json\n(?P<payload>.*?)\n```", re.DOTALL)


def test_load_seed_documents_returns_versioned_rag_documents() -> None:
    documents = load_seed_documents()

    assert len(documents) >= 10
    assert all(document.doc_id for document in documents)
    sources = {document.metadata["source"] for document in documents}
    assert "ai-engine-curated-seed" in sources
    assert "ai-engine-curated-expansion" in sources
    assert "ai-engine-approved-examples" in sources
    assert "Science & Nature" in {
        document.metadata["category"] for document in documents
    }


def test_seed_corpus_is_part_of_full_runtime_corpus() -> None:
    corpus = get_full_corpus()

    assert any(entry.get("doc_id") == "seed-computers-databases-en" for entry in corpus)


def test_describe_seed_corpus_reports_coverage() -> None:
    summary = describe_seed_corpus()

    assert summary["documents"] >= 10
    assert "en" in summary["languages"]
    assert "ai-engine-curated-seed" in summary["sources"]
    assert "History" in summary["categories"]


def test_load_seed_corpus_entries_rejects_missing_metadata(tmp_path: Path) -> None:
    corpus_file = tmp_path / "bad.jsonl"
    corpus_file.write_text(
        '{"doc_id":"bad","content":"Missing category","metadata":{"kind":"educational_resource"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(RagSeedCorpusError, match="missing metadata keys"):
        load_seed_corpus_entries([corpus_file])


def test_approved_game_examples_validate_against_game_envelope() -> None:
    examples = [
        entry
        for entry in load_seed_corpus_entries()
        if entry["metadata"].get("kind") == "game_example"
        and entry["metadata"].get("source") == "ai-engine-approved-examples"
    ]

    assert examples
    for entry in examples:
        match = _JSON_BLOCK_RE.search(entry["content"])
        assert match, f"missing JSON block in {entry['doc_id']}"
        payload = json.loads(match.group("payload"))
        envelope = GameEnvelope.from_dict(payload)
        assert envelope.game_type == entry["metadata"]["game_type"]
