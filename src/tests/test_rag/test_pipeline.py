"""Integration tests for ai_engine.rag.pipeline (RAGPipeline)."""

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore


class _FixedEmbedder(Embedder):
    """Deterministic embedder for testing: returns one-hot vectors by doc content."""

    _MAP = {
        "python": [1.0, 0.0, 0.0],
        "ml": [0.0, 1.0, 0.0],
        "data": [0.0, 0.0, 1.0],
    }

    def embed_text(self, text: str) -> list[float]:
        for key, vec in self._MAP.items():
            if key in text.lower():
                return vec
        return [0.33, 0.33, 0.33]


def _build_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=_FixedEmbedder(),
        vector_store=InMemoryVectorStore(),
    )


def test_ingest_and_retrieve():
    pipeline = _build_pipeline()
    docs = [
        Document(content="Python is a high-level language", doc_id="1"),
        Document(content="ML is machine learning", doc_id="2"),
    ]
    pipeline.ingest(docs)
    results = pipeline.retrieve("tell me about python", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0].content


def test_build_context_returns_string():
    pipeline = _build_pipeline()
    pipeline.ingest([Document(content="Python is great for data science", doc_id="1")])
    context = pipeline.build_context("python question")
    assert isinstance(context, str)
    assert "Python" in context


def test_retrieve_prefers_matching_metadata():
    pipeline = _build_pipeline()
    pipeline.ingest(
        [
            Document(
                content="Python question example in English",
                doc_id="en-quiz",
                metadata={"language": "en", "game_type": "quiz", "kind": "game_example"},
            ),
            Document(
                content="Python question example in Spanish",
                doc_id="es-quiz",
                metadata={"language": "es", "game_type": "quiz", "kind": "game_example"},
            ),
        ]
    )

    results = pipeline.retrieve(
        "python question",
        top_k=1,
        metadata_preferences={"language": "es", "game_type": "quiz"},
    )

    assert len(results) == 1
    assert results[0].doc_id.startswith("es-quiz")


def test_build_context_adds_headers_and_honors_limit():
    pipeline = _build_pipeline()
    pipeline.ingest(
        [
            Document(
                content="Python is great for data science and scripting.",
                doc_id="1",
                metadata={"language": "es", "kind": "educational_resource", "topic": "python"},
            )
        ]
    )

    context = pipeline.build_context("python question", max_chars=70)

    assert context.startswith("Source 1")
    assert len(context) <= 73


def test_ingest_empty_list():
    pipeline = _build_pipeline()
    pipeline.ingest([])  # should not raise
    results = pipeline.retrieve("anything")
    assert results == []
