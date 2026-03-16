"""Small SDK for typed game objects returned by ``POST /generate``."""

from ai_engine.sdk.languages import LanguageCode, LanguageInfo, get_language_info
from ai_engine.sdk.models import (
    BestAnswerQuestion,
    GeneratedGameModel,
    GeneratedPasapalabra,
    GeneratedQuiz,
    GenerationMetadata,
    MultipleChoiceQuestion,
    PasapalabraEntry,
    TrueFalseQuestion,
    parse_generate_response,
)

__all__ = [
    "BestAnswerQuestion",
    "GeneratedGameModel",
    "GeneratedPasapalabra",
    "GeneratedQuiz",
    "GenerationMetadata",
    "LanguageCode",
    "LanguageInfo",
    "MultipleChoiceQuestion",
    "PasapalabraEntry",
    "TrueFalseQuestion",
    "get_language_info",
    "parse_generate_response",
]
