"""Pydantic request and response schemas for the generation API.

All models are used for automatic request validation and OpenAPI
documentation generation by FastAPI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
        query: Keywords used to retrieve relevant context from the RAG pipeline.
        topic: Educational topic for the game content.
        game_type: One of ``"quiz"``, ``"pasapalabra"``, ``"true_false"``.
        language: ISO 639-1 language code for the generated content.
        num_questions: Number of questions / statements to generate.
        letters: Comma-separated letters for pasapalabra rosco.
        max_tokens: Maximum tokens for the LLM generation call.
    """

    query: str
    topic: str
    game_type: str = "quiz"
    language: str = "es"
    num_questions: int = Field(default=5, ge=1, le=50)
    letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z"
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    use_cache: bool = True
    force_refresh: bool = False


class GenerateSDKResponse(BaseModel):
    """Typed SDK-oriented payload returned by ``POST /generate/sdk``."""

    model_type: str
    metadata: dict[str, Any]
    data: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
