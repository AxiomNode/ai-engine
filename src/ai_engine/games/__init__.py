"""Educational game generation module.

Provides data models, prompt templates, and a generator that uses the
RAG pipeline + a local LLM to produce structured JSON definitions for
educational games (quiz, pasapalabra, true/false, etc.).
"""

from ai_engine.games.schemas import (
    QuizGame,
    QuizQuestion,
    PasapalabraGame,
    PasapalabraWord,
    TrueFalseGame,
    TrueFalseStatement,
    GameEnvelope,
)
from ai_engine.games.generator import GameGenerator

__all__ = [
    "QuizGame",
    "QuizQuestion",
    "PasapalabraGame",
    "PasapalabraWord",
    "TrueFalseGame",
    "TrueFalseStatement",
    "GameEnvelope",
    "GameGenerator",
]
