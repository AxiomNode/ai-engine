"""SentenceTransformers-based embedder for the RAG pipeline."""

from __future__ import annotations

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Embedder backed by the ``sentence-transformers`` library.

    Args:
        model_name: Name or path of the sentence-transformers model.
            Defaults to ``"all-MiniLM-L6-v2"`` which is small, fast,
            and produces 384-dimensional embeddings.

    Example:
        >>> emb = SentenceTransformersEmbedder()
        >>> vec = emb.embed_text("hello world")
        >>> len(vec)
        384
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-untyped]
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._st_class = SentenceTransformer
        self._model: SentenceTransformer | None = None

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = self._st_class(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Return a vector embedding for a single text string."""
        embedding = self._get_model().encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Return embeddings for a batch of documents.

        More efficient than the default one-by-one approach because
        sentence-transformers can batch-encode.
        """
        texts = [doc.content for doc in documents]
        embeddings = self._get_model().encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
