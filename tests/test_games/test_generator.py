"""Tests for ai_engine.games.generator – GameGenerator with mock LLM."""

import json

import pytest

from ai_engine.games.generator import GameGenerator
from ai_engine.games.schemas import GameEnvelope, QuizGame, PasapalabraGame, TrueFalseGame
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore


# ------------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------------

class _FixedEmbedder(Embedder):
    """Embedder that returns a fixed vector for deterministic tests."""

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _MockLLMQuiz:
    """Mock LLM that returns a valid quiz JSON."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "quiz",
            "title": "Mock Quiz",
            "topic": "Testing",
            "questions": [
                {
                    "question": "What is 1+1?",
                    "options": ["1", "2", "3", "4"],
                    "correct_index": 1,
                    "explanation": "Basic addition.",
                }
            ],
        }
        return json.dumps(data)


class _MockLLMPasapalabra:
    """Mock LLM that returns a valid pasapalabra JSON."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "pasapalabra",
            "title": "Mock Rosco",
            "topic": "Testing",
            "words": [
                {"letter": "A", "hint": "First letter", "answer": "Alpha", "starts_with": True},
                {"letter": "B", "hint": "Second letter", "answer": "Beta", "starts_with": True},
            ],
        }
        return json.dumps(data)


class _MockLLMTrueFalse:
    """Mock LLM that returns a valid true/false JSON."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "true_false",
            "title": "Mock T/F",
            "topic": "Testing",
            "statements": [
                {"statement": "The sky is blue", "is_true": True, "explanation": "Rayleigh scattering."},
            ],
        }
        return json.dumps(data)


class _MockLLMBadOutput:
    """Mock LLM that returns non-JSON garbage."""

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return "I cannot help you with that. Sorry!"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def rag_pipeline():
    emb = _FixedEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store)
    pipeline.ingest([
        Document(content="Python is a high-level language.", doc_id="d1"),
        Document(content="The Earth orbits the Sun.", doc_id="d2"),
    ])
    return pipeline


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestGameGeneratorInit:

    def test_requires_rag_pipeline(self):
        with pytest.raises(ValueError, match="rag_pipeline"):
            GameGenerator(rag_pipeline=None, llm_client=_MockLLMQuiz())

    def test_requires_llm_client(self, rag_pipeline):
        with pytest.raises(ValueError, match="llm_client"):
            GameGenerator(rag_pipeline=rag_pipeline, llm_client=None)


class TestGameGeneratorQuiz:

    def test_generate_quiz(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMQuiz())
        result = gen.generate(query="Python", topic="Programming", game_type="quiz")

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "quiz"
        assert isinstance(result.game, QuizGame)
        assert result.game.title == "Mock Quiz"
        assert len(result.game.questions) == 1
        assert result.game.questions[0].correct_index == 1

    def test_generate_raw_returns_dict(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMQuiz())
        result = gen.generate_raw(query="Python", topic="Programming")
        assert isinstance(result, dict)
        assert result["game_type"] == "quiz"


class TestGameGeneratorPasapalabra:

    def test_generate_pasapalabra(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMPasapalabra())
        result = gen.generate(query="letters", topic="Alphabet", game_type="pasapalabra")

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "pasapalabra"
        assert isinstance(result.game, PasapalabraGame)
        assert len(result.game.words) == 2


class TestGameGeneratorTrueFalse:

    def test_generate_true_false(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMTrueFalse())
        result = gen.generate(query="sky", topic="Science", game_type="true_false")

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "true_false"
        assert isinstance(result.game, TrueFalseGame)
        assert len(result.game.statements) == 1


class TestGameGeneratorErrorHandling:

    def test_bad_llm_output_raises_value_error(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBadOutput())
        with pytest.raises(ValueError, match="Failed to extract JSON"):
            gen.generate(query="anything", topic="X")
