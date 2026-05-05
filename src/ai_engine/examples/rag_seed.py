"""Load versioned RAG seed documents from package data."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from ai_engine.rag.document import Document

_DEFAULT_RESOURCE_PACKAGE = "ai_engine.examples.rag_documents"
_REQUIRED_METADATA_KEYS = {"kind", "source", "language", "category", "topic"}


class RagSeedCorpusError(ValueError):
    """Raised when a seed corpus entry is malformed."""


def load_seed_corpus_entries(
    paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Return validated seed corpus entries from package data or JSONL paths."""
    entries = []
    for line_number, raw_line in _iter_seed_lines(paths):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagSeedCorpusError(
                f"Invalid JSONL entry at line {line_number}: {exc}"
            ) from exc
        entries.append(_validate_seed_entry(entry, line_number=line_number))
    return entries


def load_seed_documents(paths: Iterable[str | Path] | None = None) -> list[Document]:
    """Return seed corpus entries as RAG Document objects."""
    return [
        Document(
            content=entry["content"],
            doc_id=entry["doc_id"],
            metadata=dict(entry.get("metadata", {})),
        )
        for entry in load_seed_corpus_entries(paths)
    ]


def describe_seed_corpus(paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Summarize seed corpus coverage for diagnostics and CLI output."""
    entries = load_seed_corpus_entries(paths)
    categories = sorted(
        {str(entry["metadata"].get("category", "")) for entry in entries}
    )
    languages = sorted(
        {str(entry["metadata"].get("language", "")) for entry in entries}
    )
    sources = sorted({str(entry["metadata"].get("source", "")) for entry in entries})
    return {
        "documents": len(entries),
        "categories": categories,
        "languages": languages,
        "sources": sources,
    }


def _iter_seed_lines(paths: Iterable[str | Path] | None) -> Iterable[tuple[int, str]]:
    if paths is None:
        package_root = resources.files(_DEFAULT_RESOURCE_PACKAGE)
        for resource in sorted(package_root.iterdir(), key=lambda item: item.name):
            if resource.name.endswith(".jsonl"):
                with resource.open("r", encoding="utf-8") as handle:
                    yield from enumerate(handle, start=1)
        return

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            yield from enumerate(handle, start=1)


def _validate_seed_entry(entry: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RagSeedCorpusError(f"Seed entry at line {line_number} must be an object")

    doc_id = entry.get("doc_id")
    content = entry.get("content")
    metadata = entry.get("metadata")

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise RagSeedCorpusError(f"Seed entry at line {line_number} is missing doc_id")
    if not isinstance(content, str) or not content.strip():
        raise RagSeedCorpusError(f"Seed entry at line {line_number} is missing content")
    if not isinstance(metadata, dict):
        raise RagSeedCorpusError(
            f"Seed entry at line {line_number} is missing metadata"
        )

    missing_metadata = sorted(
        key for key in _REQUIRED_METADATA_KEYS if not metadata.get(key)
    )
    if missing_metadata:
        raise RagSeedCorpusError(
            f"Seed entry at line {line_number} is missing metadata keys: {', '.join(missing_metadata)}"
        )

    return {
        "doc_id": doc_id.strip(),
        "content": content.strip(),
        "metadata": dict(metadata),
    }
