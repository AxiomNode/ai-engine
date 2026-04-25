"""Pydantic request and response schemas for the generation API.

All models are used for automatic request validation and OpenAPI
documentation generation by FastAPI.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator


class DocumentInput(BaseModel):
    """A single document to ingest into the RAG pipeline.

    Attributes:
        content: The raw text content of the document.
        doc_id: Optional unique identifier. Auto-generated if omitted.
        metadata: Arbitrary key-value pairs (source file, page number, etc.).
    """

    content: str
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``.

    Attributes:
        documents: List of documents to chunk, embed, and store.
    """

    documents: list[DocumentInput]


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``.

    Attributes:
        query: Optional topic override used to retrieve relevant context.
        game_type: One of ``"quiz"``, ``"word-pass"``, ``"true_false"``.
        item_count: Number of items to generate.
        max_tokens: Maximum tokens for the LLM generation call.
    """

    query: str | None = None
    game_type: str = "quiz"
    item_count: int = Field(
        default=5,
        ge=1,
        le=50,
        validation_alias=AliasChoices("item_count", "num_questions"),
    )
    difficulty_percentage: int = Field(default=50, ge=0, le=100)
    category_id: str | None = None
    category_name: str | None = None
    letters: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
    )
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    use_cache: bool = True
    force_refresh: bool = False

    @model_validator(mode="after")
    def validate_generation_target(self) -> "GenerateRequest":
        self.query = self.query.strip() if isinstance(self.query, str) else None
        self.category_id = (
            self.category_id.strip() if isinstance(self.category_id, str) else None
        )
        self.category_name = (
            self.category_name.strip() if isinstance(self.category_name, str) else None
        )
        if not self.query and not self.category_id and not self.category_name:
            raise ValueError("query or category_id/category_name is required")
        return self

    @property
    def num_questions(self) -> int:
        """Backward-compatible alias for the historical item count name."""
        return self.item_count

    @property
    def resolved_topic(self) -> str:
        """Return the best available topic seed for retrieval and prompting."""
        if self.query:
            return self.query
        if self.category_name:
            return self.category_name
        return self.category_id or ""


class GenerateSDKResponse(BaseModel):
    """Typed SDK-oriented payload returned by ``POST /generate/sdk``."""

    model_type: str
    metadata: dict[str, Any]
    data: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
