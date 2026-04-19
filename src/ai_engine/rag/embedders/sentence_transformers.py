"""SentenceTransformers-based embedder for the RAG pipeline."""

from __future__ import annotations

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Embedder backed by the ``sentence-transformers`` library.

    Args:
        model_name: Name or path of the sentence-transformers model.
            Defaults to ``"paraphrase-multilingual-MiniLM-L12-v2"``
            which offers stronger multilingual retrieval quality while
            remaining practical for CPU-bound deployments.

    Example:
        >>> emb = SentenceTransformersEmbedder()
        >>> vec = emb.embed_text("hello world")
        >>> len(vec)
        384
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
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
        self.device = (device or "cpu").strip() or "cpu"
        self.batch_size = max(1, int(batch_size))
        self.normalize_embeddings = normalize_embeddings
        self._st_class = SentenceTransformer
        self._model: SentenceTransformer | None = None

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = self._st_class(self.model_name, device=self.device)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Return a vector embedding for a single text string."""
        embedding = self._get_model().encode(
            text,
            convert_to_numpy=True,
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Return embeddings for a batch of documents.

        More efficient than the default one-by-one approach because
        sentence-transformers can batch-encode.
        """
        texts = [doc.content for doc in documents]
        embeddings = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]
