"""Educational game generation module.

Provides data models, prompt templates, and a generator that uses the
RAG pipeline + a local LLM to produce structured JSON definitions for
educational games (quiz, word-pass, true/false, etc.).
"""

from ai_engine.games.generator import GameGenerator
from ai_engine.games.schemas import (
    GameEnvelope,
    WordPassGame,
    WordPassWord,
    QuizGame,
    QuizQuestion,
    TrueFalseGame,
    TrueFalseStatement,
)

__all__ = [
    "QuizGame",
    "QuizQuestion",
    "WordPassGame",
    "WordPassWord",
    "TrueFalseGame",
    "TrueFalseStatement",
    "GameEnvelope",
    "GameGenerator",
]
