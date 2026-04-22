"""Tests for RAGPipeline.generate() – JSON extraction from LLM output."""

import asyncio

import pytest

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore


class DummyEmbedder(Embedder):
    """Embedder that returns a fixed vector."""

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class MockLLM:
    """Mock LLM that returns chain-of-thought followed by JSON."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: object) -> str:
        return (
            "Thinking step 1...\nThinking step 2...\n\n{"
            + '"title": "TestGame", "questions": []}'
        )


class AsyncMockLLM:
    async def generate(
        self, prompt: str, max_tokens: int = 256, **kwargs: object
    ) -> str:
        await asyncio.sleep(0)
        return '{"title": "AsyncGame", "questions": []}'


class BrokenMockLLM:
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: object) -> str:
        return "No JSON here"


def test_pipeline_generate_json_extraction():
    emb = DummyEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store, llm_client=MockLLM())

    doc = Document(
        content="Some useful context about rivers",
        metadata={"source": "textbook"},
        doc_id="r1",
    )
    pipeline.ingest([doc])

    result = pipeline.generate(query="rivers?", goal="Create a short quiz")
    assert isinstance(result, dict)
    assert result.get("title") == "TestGame"


def test_pipeline_generate_supports_async_llm_without_generate_sync():
    emb = DummyEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store, llm_client=AsyncMockLLM())

    pipeline.ingest(
        [Document(content="Async context", metadata={"source": "doc"}, doc_id="a1")]
    )

    result = pipeline.generate(query="async?", goal="Create a short quiz")

    assert result["title"] == "AsyncGame"


def test_pipeline_generate_raises_when_model_output_has_no_json():
    emb = DummyEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store, llm_client=BrokenMockLLM())

    pipeline.ingest(
        [Document(content="Some context", metadata={"source": "doc"}, doc_id="b1")]
    )

    with pytest.raises(ValueError, match="Failed to extract JSON"):
        pipeline.generate(query="broken?", goal="Create a short quiz")
