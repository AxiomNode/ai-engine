"""Tests for ai_engine.sdk models."""

from ai_engine.sdk import (
    GeneratedQuiz,
    GeneratedWordPass,
    LanguageCode,
    get_language_info,
    parse_generate_response,
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
