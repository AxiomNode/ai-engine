"""Tests for generation optimization service internals."""

from __future__ import annotations

from dataclasses import dataclass

from ai_engine.api.optimization import GenerationOptimizationService
from ai_engine.api.schemas import GenerateRequest
from ai_engine.games.schemas import GameEnvelope, QuizGame, QuizQuestion


@dataclass
class _Doc:
    """Simple retrieval document stub."""

    content: str


class _StubRAGPipeline:
    """Minimal RAG stub for optimization tests."""

    def retrieve(self, query: str) -> list[_Doc]:
        _ = query
        return [_Doc(content="Water cycle context")]


class _StubGenerator:
    """Minimal game generator stub exposing generate_from_context."""

    def __init__(self) -> None:
        self.last_run_metrics = {
            "llm_total_ms": 5.0,
            "parse_total_ms": 1.0,
            "retry_used": False,
        }

    def generate_from_context(
        self,
        context: str,
        topic: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
        num_questions: int = 5,
        letters: str = "A,B,C",
        max_tokens: int | None = None,
    ) -> GameEnvelope:
        _ = (context, topic, game_type, language, num_questions, letters, max_tokens)
        return GameEnvelope(
            game_type="quiz",
            game=QuizGame(
                title="Test Quiz",
                topic="Science",
                questions=[
                    QuizQuestion(
                        question="What is H2O?",
                        options=["Water", "Fire", "Air", "Earth"],
                        correct_index=0,
                        explanation="H2O is water.",
                    )
                ],
            ),
        )


def test_persistent_cache_stats_and_reset_use_cache_index(tmp_path) -> None:
    """Persistent cache stats/reset should track only cache entries efficiently."""
    cache_file = tmp_path / "generation_cache.json"
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=str(cache_file),
    )

    req = GenerateRequest(query="water", topic="Science")
    service.generate(req)

    stats_before = service.cache_stats()
    assert stats_before["persistent_enabled"] is True
    assert stats_before["persistent_entries"] == 1

    removed = service.reset_cache()
    assert removed["removed_persistent"] == 1

    stats_after = service.cache_stats()
    assert stats_after["persistent_entries"] == 0
