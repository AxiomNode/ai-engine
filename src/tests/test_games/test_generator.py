"""Tests for ai_engine.games.generator – GameGenerator with mock LLM."""

import asyncio
import json

import pytest

from ai_engine.games.generator import GameGenerator
from ai_engine.games.schemas import (
    GameEnvelope,
    QuizGame,
    TrueFalseGame,
    WordPassGame,
)
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore

# ------------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously for test convenience."""
    return asyncio.run(coro)


class _FixedEmbedder(Embedder):
    """Embedder that returns a fixed vector for deterministic tests."""

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _MockLLMQuiz:
    """Mock LLM that returns a valid quiz JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "quiz",
            "title": "Mock Quiz",
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


class _MockLLMWordPass:
    """Mock LLM that returns a valid word-pass JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "word-pass",
            "title": "Mock Rosco",
            "words": [
                {
                    "letter": "A",
                    "hint": "First letter",
                    "answer": "Alpha",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Second letter",
                    "answer": "Beta",
                    "starts_with": True,
                },
            ],
        }
        return json.dumps(data)


class _MockLLMTrueFalse:
    """Mock LLM that returns a valid true/false JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "true_false",
            "title": "Mock T/F",
            "statements": [
                {
                    "statement": "The sky is blue",
                    "is_true": True,
                    "explanation": "Rayleigh scattering.",
                },
            ],
        }
        return json.dumps(data)


class _MockLLMBadOutput:
    """Mock LLM that returns non-JSON garbage."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return "I cannot help you with that. Sorry!"


class _MockLLMRetryThenValid:
    """Mock LLM that fails once, then returns valid JSON on retry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return '{"game_type":"quiz","title":"Broken"'
        data = {
            "game_type": "quiz",
            "title": "Recovered Quiz",
            "questions": [
                {
                    "question": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "correct_index": 1,
                    "explanation": "Basic arithmetic.",
                }
            ],
        }
        return json.dumps(data)


class _MockLLMMalformedQuizPayload:
    """Mock LLM that returns incomplete quiz payload requiring normalization."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "educational-game",
            "questions": [
                {
                    "question": "",
                    "options": [""],
                    "correct_index": 9,
                }
            ],
        }
        return json.dumps(data)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def rag_pipeline():
    emb = _FixedEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store)
    pipeline.ingest(
        [
            Document(content="Python is a high-level language.", doc_id="d1"),
            Document(content="The Earth orbits the Sun.", doc_id="d2"),
        ]
    )
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
        result = _run(gen.generate(query="Python", game_type="quiz"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "quiz"
        assert isinstance(result.game, QuizGame)
        assert result.game.title == "Mock Quiz"
        assert len(result.game.questions) == 1
        assert result.game.questions[0].correct_index == 1

    def test_generate_raw_returns_dict(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMQuiz())
        result = _run(gen.generate_raw(query="Python"))
        assert isinstance(result, dict)
        assert result["game_type"] == "quiz"


class TestGameGeneratorWordPass:

    def test_generate_word_pass(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMWordPass())
        result = _run(gen.generate(query="letters", game_type="word-pass"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "word-pass"
        assert isinstance(result.game, WordPassGame)
        assert len(result.game.words) == 2


class TestGameGeneratorTrueFalse:

    def test_generate_true_false(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMTrueFalse())
        result = _run(gen.generate(query="sky", game_type="true_false"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "true_false"
        assert isinstance(result.game, TrueFalseGame)
        assert len(result.game.statements) == 1


class TestGameGeneratorErrorHandling:

    def test_bad_llm_output_raises_value_error(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBadOutput())
        with pytest.raises(ValueError, match="Failed to extract JSON"):
            _run(gen.generate(query="anything"))

    def test_retries_once_and_recovers_when_first_output_is_invalid(self, rag_pipeline):
        llm = _MockLLMRetryThenValid()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(gen.generate(query="math", game_type="quiz"))

        assert envelope.game_type == "quiz"
        assert envelope.game.title == "Recovered Quiz"
        assert len(llm.calls) == 2
        first_prompt, first_tokens = llm.calls[0]
        second_prompt, second_tokens = llm.calls[1]
        assert first_tokens == 1024
        assert second_tokens > first_tokens
        assert second_prompt.startswith(first_prompt)

    def test_raises_after_retry_if_both_outputs_are_invalid(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBadOutput())

        with pytest.raises(ValueError, match="after retry"):
            _run(gen.generate(query="anything"))

    def test_normalizes_malformed_quiz_payload(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMMalformedQuizPayload(),
        )

        result = _run(gen.generate(query="x", game_type="quiz"))

        assert result.game_type == "quiz"
        assert result.game.title
        assert len(result.game.questions) == 1
        question = result.game.questions[0]
        assert question.question
        assert len(question.options) >= 2
        assert 0 <= question.correct_index < len(question.options)
