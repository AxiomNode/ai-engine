"""Tests for ai_engine.sdk models."""

import pytest

from ai_engine.sdk import (
    GeneratedQuiz,
    GeneratedWordPass,
    LanguageCode,
    get_language_info,
    parse_generate_response,
)
from ai_engine.sdk.models import (
    BestAnswerQuestion,
    GenerationMetadata,
    MultipleChoiceQuestion,
)


def test_parse_quiz_payload_to_generated_quiz() -> None:
    payload = {
        "game_type": "quiz",
        "game": {
            "game_type": "quiz",
            "title": "Physics Basics",
            "questions": [
                {
                    "question": "What is gravity?",
                    "options": ["A force", "A color", "A sound", "A number"],
                    "correct_index": 0,
                    "explanation": "Gravity is a force.",
                }
            ],
        },
    }

    result = parse_generate_response(payload, language=LanguageCode.EN)
    assert isinstance(result, GeneratedQuiz)
    assert result.metadata.language == LanguageCode.EN
    assert result.metadata.language_id == "lang-en"
    assert result.title == "Physics Basics"
    assert len(result.questions) == 1


def test_parse_true_false_payload_to_generated_quiz_variant() -> None:
    payload = {
        "game_type": "true_false",
        "game": {
            "game_type": "true_false",
            "title": "Earth Facts",
            "statements": [
                {
                    "statement": "The Earth is round.",
                    "is_true": True,
                    "explanation": "Observed from space.",
                }
            ],
        },
    }

    result = parse_generate_response(payload, language="es")
    assert isinstance(result, GeneratedQuiz)
    assert result.metadata.language == LanguageCode.ES
    assert result.metadata.language_id == "lang-es"
    assert result.questions[0].question_type == "true_false"


def test_parse_word_pass_payload() -> None:
    payload = {
        "game_type": "word-pass",
        "game": {
            "game_type": "word-pass",
            "title": "Science Rosco",
            "words": [
                {
                    "letter": "A",
                    "hint": "Basic unit of matter",
                    "answer": "Atom",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Contains this letter and means tiny life-form",
                    "answer": "Microbe",
                    "starts_with": False,
                },
            ],
        },
    }

    result = parse_generate_response(payload, language="fr")
    assert isinstance(result, GeneratedWordPass)
    assert result.metadata.language_id == "lang-fr"
    assert result.entries[0].relation == "starts_with"
    assert result.entries[1].relation == "contains"


def test_get_language_info_raises_for_unsupported_code() -> None:
    try:
        get_language_info("jp")
    except ValueError as exc:
        assert "Unsupported language" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported language code")


def test_generation_metadata_populates_language_id_from_string_language() -> None:
    metadata = GenerationMetadata(language="fr")

    assert metadata.language == LanguageCode.FR
    assert metadata.language_id == "lang-fr"


def test_multiple_choice_question_validates_correct_index() -> None:
    with pytest.raises(ValueError, match="correct_index out of range"):
        MultipleChoiceQuestion(question="Q", options=["A", "B"], correct_index=3)


def test_best_answer_question_validates_best_index() -> None:
    with pytest.raises(ValueError, match="best_index out of range"):
        BestAnswerQuestion(question="Q", options=["A", "B"], best_index=9)


def test_generated_quiz_parses_best_answer_and_true_false_variants() -> None:
    payload = {
        "game_type": "quiz",
        "metadata": {"difficulty_percentage": 20},
        "game": {
            "title": "Mixed Quiz",
            "difficulty_percentage": 65,
            "questions": [
                {
                    "question_type": "best_answer",
                    "question": "Best option?",
                    "options": ["A", "B", "C"],
                    "correct_index": 2,
                },
                {
                    "question_type": "true_false",
                    "question": "Water boils at 100C",
                    "is_true": True,
                    "explanation": "At sea level.",
                },
                "skip-me",
            ],
        },
    }

    result = GeneratedQuiz.from_generate_payload(payload, language="en")

    assert result.metadata.difficulty_percentage == 65
    assert result.questions[0].question_type == "best_answer"
    assert result.questions[1].question_type == "true_false"
    assert len(result.questions) == 2


def test_generated_quiz_rejects_unsupported_game_type() -> None:
    with pytest.raises(ValueError, match="cannot be mapped to GeneratedQuiz"):
        GeneratedQuiz.from_generate_payload(
            {"game_type": "word-pass", "game": {}}, language="es"
        )


def test_generated_quiz_falls_back_to_top_level_payload_when_game_is_not_a_mapping() -> (
    None
):
    result = GeneratedQuiz.from_generate_payload(
        {
            "game_type": "quiz",
            "game": ["bad"],
            "title": "Top level quiz",
            "questions": [
                {
                    "question": "Fallback question",
                    "options": ["A", "B"],
                    "correct_index": 0,
                }
            ],
        },
        language="es",
    )

    assert result.title == "Top level quiz"
    assert len(result.questions) == 1


def test_generated_word_pass_rejects_wrong_type_and_non_mapping_game() -> None:
    with pytest.raises(ValueError, match="cannot be mapped to GeneratedWordPass"):
        GeneratedWordPass.from_generate_payload(
            {"game_type": "quiz", "game": {}}, language="es"
        )

    result = GeneratedWordPass.from_generate_payload(
        {
            "game_type": "word-pass",
            "game": ["bad"],
            "title": "Top level word pass",
            "words": [
                {
                    "letter": "A",
                    "hint": "Top level definition",
                    "answer": "Atom",
                    "starts_with": True,
                }
            ],
        },
        language="es",
    )

    assert result.title == "Top level word pass"
    assert len(result.entries) == 1


def test_generated_word_pass_uses_game_difficulty_and_skips_invalid_items() -> None:
    payload = {
        "game_type": "word-pass",
        "metadata": {"difficulty_percentage": 10},
        "game": {
            "title": "Rosco",
            "difficulty_percentage": 70,
            "words": [
                {
                    "letter": " a ",
                    "hint": "Definition",
                    "answer": "Atom",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Contains letter",
                    "answer": "Microbe",
                    "starts_with": False,
                },
                "skip-me",
            ],
        },
    }

    result = GeneratedWordPass.from_generate_payload(payload, language="fr")

    assert result.metadata.difficulty_percentage == 70
    assert result.entries[0].letter == "A"
    assert result.entries[1].relation == "contains"
    assert len(result.entries) == 2


def test_parse_generate_response_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported game_type"):
        parse_generate_response({"game_type": "arcade"}, language="es")
