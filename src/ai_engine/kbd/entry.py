"""Knowledge base entry model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge base.

    Attributes:
        entry_id: Unique identifier for this entry.
        title: Short human-readable title.
        content: Full text content of the entry.
        tags: List of tags for filtering.
        metadata: Arbitrary extra metadata.
    """

    entry_id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a plain dictionary."""
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "content": self.content,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        """Deserialize an entry from a plain dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A :class:`KnowledgeEntry` instance.
        """
        return cls(
            entry_id=data["entry_id"],
            title=data["title"],
            content=data["content"],
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )
