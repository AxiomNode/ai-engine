from __future__ import annotations

import asyncio

import pytest

from ai_engine.games.generator import GameGenerator


class _StaticPipeline:
    def build_context(self, query: str, top_k: int | None = None, **kwargs) -> str:
        return query


class _StaticLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        return self.payload


def _build_generator(payload: str) -> GameGenerator:
    return GameGenerator(
        rag_pipeline=_StaticPipeline(),
        llm_client=_StaticLLM(payload),
        default_language="es",
    )


def test_quiz_generation_rejects_missing_question_text() -> None:
    generator = _build_generator(
        """
        {
          "game_type": "quiz",
          "title": "Quiz de prueba",
          "questions": [
            {
              "question": "",
              "options": ["A", "B"],
              "correct_index": 0,
              "explanation": ""
            }
          ]
        }
        """
    )

    with pytest.raises(ValueError, match="missing text"):
        asyncio.run(generator.generate_from_context(context="contexto", game_type="quiz"))


def test_quiz_generation_rejects_missing_options_instead_of_using_placeholders() -> None:
    generator = _build_generator(
        """
        {
          "game_type": "quiz",
          "title": "Quiz de prueba",
          "questions": [
            {
              "question": "Pregunta valida",
              "options": ["Unica opcion"],
              "correct_index": 0,
              "explanation": ""
            }
          ]
        }
        """
    )

    with pytest.raises(ValueError, match="at least 2 options"):
        asyncio.run(generator.generate_from_context(context="contexto", game_type="quiz"))


def test_word_pass_generation_rejects_duplicate_letters() -> None:
    generator = _build_generator(
        """
        {
          "game_type": "word-pass",
          "title": "Rosco de prueba",
          "words": [
            {"letter": "A", "hint": "Primera", "answer": "Arbol", "starts_with": true},
            {"letter": "A", "hint": "Duplicada", "answer": "Avion", "starts_with": true}
          ]
        }
        """
    )

    with pytest.raises(ValueError, match="duplicates letter"):
        asyncio.run(
            generator.generate_from_context(context="contexto", game_type="word-pass")
        )


def test_true_false_generation_requires_boolean_is_true() -> None:
    generator = _build_generator(
        """
        {
          "game_type": "true_false",
          "title": "Verdadero o falso",
          "statements": [
            {"statement": "Texto valido", "is_true": "yes", "explanation": ""}
          ]
        }
        """
    )

    with pytest.raises(ValueError, match="invalid is_true"):
        asyncio.run(
            generator.generate_from_context(context="contexto", game_type="true_false")
        )


def test_quiz_generation_prunes_off_topic_question_after_retry() -> None:
    generator = _build_generator(
        """
        {
          "game_type": "quiz",
          "title": "Fotosintesis",
          "questions": [
            {
              "question": "¿Qué es la fotosíntesis?",
              "options": ["Un proceso", "Un planeta"],
              "correct_index": 0,
              "explanation": "La fotosíntesis convierte luz en energía química."
            },
            {
              "question": "¿Cuál es la capital de Francia?",
              "options": ["Madrid", "París"],
              "correct_index": 1,
              "explanation": "París es la capital francesa."
            }
          ]
        }
        """
    )

    result = asyncio.run(
        generator.generate_from_context(
            context="contexto",
            game_type="quiz",
            topic="fotosintesis",
        )
    )

    assert len(result.game.questions) == 1
    assert "fotos" in result.game.questions[0].explanation.lower()