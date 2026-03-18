"""Tests for the SentenceTransformers embedder integration."""

import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from ai_engine.rag.document import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_st_module() -> ModuleType:
    """Return a fake sentence_transformers module with a stubbed SentenceTransformer."""
    fake_module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, convert_to_numpy: bool = False):
            if isinstance(texts, str):
                return np.array([0.1, 0.2, 0.3])
            return np.array([[0.1, 0.2, 0.3]] * len(texts))

    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    return fake_module


# ---------------------------------------------------------------------------
# Unit tests using a mocked sentence_transformers dependency
# ---------------------------------------------------------------------------


class TestSentenceTransformersEmbedder:

    @pytest.fixture(autouse=True)
    def patch_sentence_transformers(self):
        """Inject a fake sentence_transformers module for every test."""
        import importlib

        import ai_engine.rag.embedders.sentence_transformers as mod

        fake = _make_fake_st_module()
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            importlib.reload(mod)
            yield mod
            importlib.reload(mod)  # restore after test

    def test_embed_text_returns_list(self, patch_sentence_transformers):
        emb = patch_sentence_transformers.SentenceTransformersEmbedder()
        vec = emb.embed_text("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_documents_batch_returns_correct_count(
        self, patch_sentence_transformers
    ):
        emb = patch_sentence_transformers.SentenceTransformersEmbedder()
        docs = [Document(content="first"), Document(content="second")]
        vecs = emb.embed_documents(docs)
        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)

    def test_default_model_name(self, patch_sentence_transformers):
        emb = patch_sentence_transformers.SentenceTransformersEmbedder()
        assert emb.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self, patch_sentence_transformers):
        emb = patch_sentence_transformers.SentenceTransformersEmbedder(
            model_name="my-custom-model"
        )
        assert emb.model_name == "my-custom-model"

    def test_import_error_raised_when_package_missing(self):
        """ImportError is raised with a helpful message when sentence-transformers is absent."""
        import importlib

        import ai_engine.rag.embedders.sentence_transformers as mod

        with patch.dict(sys.modules, {"sentence_transformers": None}):  # type: ignore[dict-item]
            importlib.reload(mod)
            try:
                with pytest.raises(ImportError, match="sentence-transformers"):
                    mod.SentenceTransformersEmbedder()
            finally:
                importlib.reload(mod)  # restore


# ---------------------------------------------------------------------------
# Integration tests (skipped when sentence_transformers is not installed)
# ---------------------------------------------------------------------------


def test_sentence_transformers_embedder_available():
    pytest.importorskip("sentence_transformers")
    from ai_engine.rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    emb = SentenceTransformersEmbedder()
    vec = emb.embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_embed_documents_batch():
    pytest.importorskip("sentence_transformers")
    from ai_engine.rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    emb = SentenceTransformersEmbedder()
    docs = [Document(content="a b c"), Document(content="d e f")]
    vecs = emb.embed_documents(docs)
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
