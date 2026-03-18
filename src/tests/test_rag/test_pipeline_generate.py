"""Tests for RAGPipeline.generate() – JSON extraction from LLM output."""

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
