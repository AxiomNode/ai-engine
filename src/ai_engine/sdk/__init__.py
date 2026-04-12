"""Small SDK for typed game objects returned by ``POST /generate``."""

from ai_engine.sdk.languages import LanguageCode, LanguageInfo, get_language_info
from ai_engine.sdk.models import (
    BestAnswerQuestion,
    GeneratedGameModel,
    GeneratedQuiz,
    GeneratedWordPass,
    GenerationMetadata,
    MultipleChoiceQuestion,
    TrueFalseQuestion,
    WordPassEntry,
    parse_generate_response,
)

__all__ = [
    "BestAnswerQuestion",
    "GeneratedGameModel",
    "GeneratedWordPass",
    "GeneratedQuiz",
    "GenerationMetadata",
    "LanguageCode",
    "LanguageInfo",
    "MultipleChoiceQuestion",
    "WordPassEntry",
    "TrueFalseQuestion",
    "get_language_info",
    "parse_generate_response",
]
