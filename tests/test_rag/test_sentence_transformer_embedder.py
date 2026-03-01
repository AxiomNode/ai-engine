import pytest

from ai_engine.rag.document import Document


def test_sentence_transformers_embedder_available():
    pytest.importorskip("sentence_transformers")
    from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder

    emb = SentenceTransformersEmbedder()
    vec = emb.embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_embed_documents_batch():
    pytest.importorskip("sentence_transformers")
    from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder

    emb = SentenceTransformersEmbedder()
    docs = [Document(content="a b c"), Document(content="d e f")]
    vecs = emb.embed_documents(docs)
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
