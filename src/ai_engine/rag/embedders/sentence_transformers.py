"""Sentence-Transformers based embedder implementation.

Uses `sentence-transformers` for local embeddings. Falls back with a clear
error message if the package is not installed.
"""

from __future__ import annotations

from typing import List

from ai_engine.rag.embedder import Embedder
from ai_engine.rag.document import Document

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install extras 'rag' or 'pip install sentence-transformers'."
            )
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        vec = self.model.encode(text)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        texts = [d.content for d in documents]
        vecs = self.model.encode(texts)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
