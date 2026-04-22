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


class _CountingEmbedder(_FixedEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed_text(text)


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
                metadata={
                    "language": "en",
                    "game_type": "quiz",
                    "kind": "game_example",
                },
            ),
            Document(
                content="Python question example in Spanish",
                doc_id="es-quiz",
                metadata={
                    "language": "es",
                    "game_type": "quiz",
                    "kind": "game_example",
                },
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
                metadata={
                    "language": "es",
                    "kind": "educational_resource",
                    "topic": "python",
                },
            )
        ]
    )

    context = pipeline.build_context("python question", max_chars=70)

    assert context.startswith("Source 1")
    assert len(context) <= 73


def test_format_context_skips_duplicates_and_blank_documents():
    pipeline = _build_pipeline()
    docs = [
        Document(content="Repeated content", doc_id="1", metadata={"language": "es"}),
        Document(content="Repeated content", doc_id="1", metadata={"language": "es"}),
        Document(content="   ", doc_id="blank"),
        Document(content="Unique content", doc_id="2"),
    ]

    context = pipeline._format_context(docs)

    assert context.count("Repeated content") == 1
    assert "Unique content" in context
    assert "blank" not in context


def test_format_context_stops_when_header_alone_exhausts_budget():
    pipeline = _build_pipeline()
    docs = [Document(content="A useful paragraph", doc_id="1")]

    context = pipeline._format_context(docs, max_chars=1)

    assert context == ""


def test_format_document_header_omits_missing_metadata_fields():
    pipeline = _build_pipeline()

    header = pipeline._format_document_header(3, Document(content="Body", doc_id="doc"))

    assert header == "Source 3"


def test_ingest_empty_list():
    pipeline = _build_pipeline()
    pipeline.ingest([])  # should not raise
    results = pipeline.retrieve("anything")
    assert results == []


def test_retrieve_reuses_cached_query_embedding_until_ingest_changes():
    embedder = _CountingEmbedder()
    pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())
    pipeline.ingest([Document(content="Python is fast", doc_id="1")])
    calls_after_ingest = embedder.calls

    first = pipeline.retrieve("python", top_k=1)
    second = pipeline.retrieve("python", top_k=1)

    assert len(first) == 1
    assert len(second) == 1
    assert embedder.calls == calls_after_ingest + 1

    pipeline.ingest([Document(content="Python can power automation", doc_id="2")])
    third = pipeline.retrieve("python", top_k=1)

    assert len(third) == 1
    assert embedder.calls == calls_after_ingest + 3
