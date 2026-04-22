"""Unit tests for the RAG+KBD demo CLI module."""

from __future__ import annotations

import sys
import types

import pytest

from ai_engine.cli import demo_rag_kbd


def test_locate_model_path_uses_first_available_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should return the first model path that exists in priority order."""

    calls: list[str] = []

    def fake_model_path(name: str) -> str:
        calls.append(name)
        if name == "qwen2.5-7b":
            raise FileNotFoundError("missing")
        return "models/phi.gguf"

    fake_model_manager = types.SimpleNamespace(
        MODELS={"qwen2.5-7b": object(), "phi-3.5-mini": object()},
        model_path=fake_model_path,
    )
    monkeypatch.setitem(sys.modules, "ai_engine.llm.model_manager", fake_model_manager)

    resolved = demo_rag_kbd.locate_model_path()

    assert resolved == "models/phi.gguf"
    assert calls == ["qwen2.5-7b", "phi-3.5-mini"]


def test_locate_model_path_raises_when_no_models_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should exit with status 1 when no local model can be resolved."""

    def always_missing(_: str) -> str:
        raise FileNotFoundError("missing")

    fake_model_manager = types.SimpleNamespace(
        MODELS={
            "qwen2.5-7b": object(),
            "phi-3.5-mini": object(),
            "qwen2.5-3b": object(),
        },
        model_path=always_missing,
    )
    monkeypatch.setitem(sys.modules, "ai_engine.llm.model_manager", fake_model_manager)

    with pytest.raises(SystemExit) as exc:
        demo_rag_kbd.locate_model_path()

    assert exc.value.code == 1


def test_main_orchestrates_all_stages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main should call every stage in order and return success code."""

    observed: list[str] = []
    fake_kb = object()
    fake_pipeline = object()

    def fake_locate_model_path() -> str:
        observed.append("locate")
        return "models/fake.gguf"

    def fake_run_kbd_demo() -> object:
        observed.append("kbd")
        return fake_kb

    def fake_run_rag_demo(kb: object) -> object:
        assert kb is fake_kb
        observed.append("rag")
        return fake_pipeline

    def fake_run_game_demo(model_path: str, pipeline: object) -> None:
        assert model_path == "models/fake.gguf"
        assert pipeline is fake_pipeline
        observed.append("game")

    monkeypatch.setattr(demo_rag_kbd, "locate_model_path", fake_locate_model_path)
    monkeypatch.setattr(demo_rag_kbd, "run_kbd_demo", fake_run_kbd_demo)
    monkeypatch.setattr(demo_rag_kbd, "run_rag_demo", fake_run_rag_demo)
    monkeypatch.setattr(demo_rag_kbd, "run_game_demo", fake_run_game_demo)

    assert demo_rag_kbd.main() == 0
    assert observed == ["locate", "kbd", "rag", "game"]
    assert "Demo complete." in capsys.readouterr().out


def test_run_kbd_demo_builds_searchable_knowledge_base(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The KBD demo should populate entries and print example search output."""
    kb = demo_rag_kbd.run_kbd_demo()

    biology_titles = [entry.title for entry in kb.search_by_tag("biologia")]
    astronomy_titles = [entry.title for entry in kb.search_by_keyword("planetas")]
    output = capsys.readouterr().out

    assert biology_titles == ["Fotosintesis"]
    assert astronomy_titles == ["Sistema solar"]
    assert "[KBD] Total entries: 3" in output


def test_run_rag_demo_ingests_entries_and_prints_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The RAG demo should build documents, ingest them and print the retrieved context."""

    class _FakeEntry:
        def __init__(self, title: str, content: str, tags: list[str]) -> None:
            self.title = title
            self.content = content
            self.tags = tags

    class _FakeKnowledgeBase:
        def list_all(self) -> list[_FakeEntry]:
            return [
                _FakeEntry("Fotosintesis", "Las plantas usan luz.", ["biologia"]),
                _FakeEntry("Sistema solar", "La Tierra orbita el Sol.", ["astronomia"]),
            ]

    captured: dict[str, object] = {}

    class _FakeChunker:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            captured["chunker"] = (chunk_size, chunk_overlap)

    class _FakeDocument:
        def __init__(self, *, content: str, metadata: dict[str, object]) -> None:
            self.content = content
            self.metadata = metadata

    class _FakeVectorStore:
        pass

    class _FakeEmbedder:
        pass

    class _FakePipeline:
        def __init__(self, *, embedder, vector_store, chunker, top_k: int) -> None:
            captured["init"] = {
                "embedder": embedder,
                "vector_store": vector_store,
                "chunker": chunker,
                "top_k": top_k,
            }

        def ingest(self, docs) -> None:
            captured["docs"] = list(docs)

        def build_context(self, query: str) -> str:
            captured["query"] = query
            return "contexto recuperado"

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.rag.chunker",
        types.SimpleNamespace(Chunker=_FakeChunker),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.rag.document",
        types.SimpleNamespace(Document=_FakeDocument),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.rag.pipeline",
        types.SimpleNamespace(RAGPipeline=_FakePipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.rag.vector_store",
        types.SimpleNamespace(InMemoryVectorStore=_FakeVectorStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.rag.embedders.sentence_transformers",
        types.SimpleNamespace(SentenceTransformersEmbedder=_FakeEmbedder),
    )

    pipeline = demo_rag_kbd.run_rag_demo(_FakeKnowledgeBase())
    output = capsys.readouterr().out

    assert isinstance(pipeline, _FakePipeline)
    assert captured["chunker"] == (300, 50)
    assert captured["query"] == "Que proceso usan las plantas para obtener energia?"
    assert [doc.metadata["title"] for doc in captured["docs"]] == [
        "Fotosintesis",
        "Sistema solar",
    ]
    assert "[RAG] Retrieved context:\ncontexto recuperado" in output


def test_run_rag_demo_exits_when_sentence_transformers_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RAG demo should stop with a helpful exit when the embedder dependency is unavailable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ai_engine.rag.embedders.sentence_transformers":
            raise ImportError("missing sentence-transformers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc:
        demo_rag_kbd.run_rag_demo(object())

    assert exc.value.code == 1


def test_run_game_demo_prints_generated_quiz(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The game demo should format and print quiz questions returned by the generator."""

    class _FakeQuestion:
        def __init__(self, question: str, options: list[str], correct_index: int) -> None:
            self.question = question
            self.options = options
            self.correct_index = correct_index

    class _FakeQuizGame:
        def __init__(self, questions: list[_FakeQuestion]) -> None:
            self.questions = questions

    class _FakeEnvelope:
        game_type = "quiz"

        def __init__(self) -> None:
            self.game = _FakeQuizGame(
                [
                    _FakeQuestion("Que produce la fotosintesis?", ["Glucosa", "Hierro"], 0)
                ]
            )

    captured: dict[str, object] = {}
    envelope = _FakeEnvelope()

    class _FakeLlamaClient:
        def __init__(self, **kwargs) -> None:
            captured["llama_kwargs"] = kwargs

    class _FakeGameGenerator:
        def __init__(self, *, rag_pipeline, llm_client, default_language: str) -> None:
            captured["generator_init"] = {
                "rag_pipeline": rag_pipeline,
                "llm_client": llm_client,
                "default_language": default_language,
            }

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return "awaitable-generate"

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.generator",
        types.SimpleNamespace(GameGenerator=_FakeGameGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.schemas",
        types.SimpleNamespace(QuizGame=_FakeQuizGame),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.llm.llama_client",
        types.SimpleNamespace(LlamaClient=_FakeLlamaClient),
    )
    monkeypatch.setattr(demo_rag_kbd.asyncio, "run", lambda awaitable: envelope)

    demo_rag_kbd.run_game_demo("models/demo.gguf", pipeline="pipeline")
    output = capsys.readouterr().out

    assert captured["llama_kwargs"]["model_path"] == "models/demo.gguf"
    assert captured["generate_kwargs"] == {
        "query": "fotosintesis plantas cloroplastos glucosa",
        "game_type": "quiz",
        "num_questions": 3,
    }
    assert "[GAME] Result:" in output
    assert "[*] Glucosa" in output


def test_run_game_demo_exits_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation errors should be converted into a demo-friendly SystemExit."""

    class _FakeQuizGame:
        pass

    class _FakeLlamaClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeGameGenerator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate(self, **kwargs):
            return "awaitable-generate"

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.generator",
        types.SimpleNamespace(GameGenerator=_FakeGameGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.schemas",
        types.SimpleNamespace(QuizGame=_FakeQuizGame),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.llm.llama_client",
        types.SimpleNamespace(LlamaClient=_FakeLlamaClient),
    )

    def raise_generation_error(awaitable):
        raise RuntimeError("boom")

    monkeypatch.setattr(demo_rag_kbd.asyncio, "run", raise_generation_error)

    with pytest.raises(SystemExit) as exc:
        demo_rag_kbd.run_game_demo("models/demo.gguf", pipeline="pipeline")

    assert exc.value.code == 1


def test_run_game_demo_rejects_non_quiz_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demo should fail fast when the generator returns a non-quiz envelope."""

    class _FakeQuizGame:
        pass

    class _OtherGame:
        pass

    class _FakeEnvelope:
        game_type = "word-pass"
        game = _OtherGame()

    class _FakeLlamaClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeGameGenerator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate(self, **kwargs):
            return "awaitable-generate"

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.generator",
        types.SimpleNamespace(GameGenerator=_FakeGameGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.games.schemas",
        types.SimpleNamespace(QuizGame=_FakeQuizGame),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_engine.llm.llama_client",
        types.SimpleNamespace(LlamaClient=_FakeLlamaClient),
    )
    monkeypatch.setattr(demo_rag_kbd.asyncio, "run", lambda awaitable: _FakeEnvelope())

    with pytest.raises(ValueError, match="Expected quiz game output"):
        demo_rag_kbd.run_game_demo("models/demo.gguf", pipeline="pipeline")
