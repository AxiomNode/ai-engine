"""Document model for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents a piece of text with associated metadata.

    Attributes:
        content: The raw text content of the document.
        metadata: Arbitrary key-value pairs (source, page, etc.).
        doc_id: Optional unique identifier.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")

    def __len__(self) -> int:
        return len(self.content)
