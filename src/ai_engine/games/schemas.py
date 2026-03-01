"""Data models for structured educational games.

All models use plain dataclasses so they stay free of heavy dependencies
and can be serialised to / deserialised from JSON trivially.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ------------------------------------------------------------------
# Quiz
# ------------------------------------------------------------------

@dataclass
class QuizQuestion:
    """A single multiple-choice question.

    Attributes:
        question: The question text.
        options: List of possible answers (typically 4).
        correct_index: Zero-based index of the correct option.
        explanation: Optional pedagogical explanation of the answer.
    """

    question: str
    options: list[str]
    correct_index: int
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("question must not be empty")
        if len(self.options) < 2:
            raise ValueError("options must contain at least 2 choices")
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError(
                f"correct_index {self.correct_index} out of range "
                f"for {len(self.options)} options"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "correct_index": self.correct_index,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuizQuestion:
        return cls(
            question=data["question"],
            options=list(data["options"]),
            correct_index=int(data["correct_index"]),
            explanation=data.get("explanation", ""),
        )


@dataclass
class QuizGame:
    """A complete quiz game definition.

    Attributes:
        title: Human-readable title for the quiz.
        topic: The educational topic covered.
        questions: Ordered list of quiz questions.
    """

    title: str
    topic: str
    questions: list[QuizQuestion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": "quiz",
            "title": self.title,
            "topic": self.topic,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuizGame:
        return cls(
            title=data["title"],
            topic=data.get("topic", ""),
            questions=[
                QuizQuestion.from_dict(q)
                for q in data.get("questions", [])
            ],
        )


# ------------------------------------------------------------------
# Pasapalabra (rosco)
# ------------------------------------------------------------------

@dataclass
class PasapalabraWord:
    """A single word entry for a Pasapalabra rosco.

    Attributes:
        letter: The letter of the alphabet this entry covers (A–Z).
        hint: Clue or definition read aloud to the player.
        answer: The correct word (starts with or contains *letter*).
        starts_with: True if the answer starts with *letter*;
            False if the answer merely contains *letter*.
    """

    letter: str
    hint: str
    answer: str
    starts_with: bool = True

    def __post_init__(self) -> None:
        if len(self.letter) != 1 or not self.letter.isalpha():
            raise ValueError(f"letter must be a single alphabetic character, got {self.letter!r}")
        self.letter = self.letter.upper()
        if not self.hint:
            raise ValueError("hint must not be empty")
        if not self.answer:
            raise ValueError("answer must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "letter": self.letter,
            "hint": self.hint,
            "answer": self.answer,
            "starts_with": self.starts_with,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PasapalabraWord:
        return cls(
            letter=data["letter"],
            hint=data["hint"],
            answer=data["answer"],
            starts_with=data.get("starts_with", True),
        )


@dataclass
class PasapalabraGame:
    """A full Pasapalabra (rosco) game.

    Attributes:
        title: Human-readable title.
        topic: The educational topic.
        words: List of word entries (ideally one per letter A-Z).
    """

    title: str
    topic: str
    words: list[PasapalabraWord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": "pasapalabra",
            "title": self.title,
            "topic": self.topic,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PasapalabraGame:
        return cls(
            title=data["title"],
            topic=data.get("topic", ""),
            words=[
                PasapalabraWord.from_dict(w)
                for w in data.get("words", [])
            ],
        )


# ------------------------------------------------------------------
# True / False
# ------------------------------------------------------------------

@dataclass
class TrueFalseStatement:
    """A single true/false statement.

    Attributes:
        statement: The assertion to evaluate.
        is_true: Whether the statement is true.
        explanation: Pedagogical explanation.
    """

    statement: str
    is_true: bool
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.statement:
            raise ValueError("statement must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "is_true": self.is_true,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrueFalseStatement:
        return cls(
            statement=data["statement"],
            is_true=bool(data["is_true"]),
            explanation=data.get("explanation", ""),
        )


@dataclass
class TrueFalseGame:
    """A complete true/false game.

    Attributes:
        title: Human-readable title.
        topic: The educational topic.
        statements: Ordered list of true/false statements.
    """

    title: str
    topic: str
    statements: list[TrueFalseStatement] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": "true_false",
            "title": self.title,
            "topic": self.topic,
            "statements": [s.to_dict() for s in self.statements],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrueFalseGame:
        return cls(
            title=data["title"],
            topic=data.get("topic", ""),
            statements=[
                TrueFalseStatement.from_dict(s)
                for s in data.get("statements", [])
            ],
        )


# ------------------------------------------------------------------
# Envelope (wraps any game type)
# ------------------------------------------------------------------

GAME_TYPE_REGISTRY: dict[str, type] = {
    "quiz": QuizGame,
    "pasapalabra": PasapalabraGame,
    "true_false": TrueFalseGame,
}


@dataclass
class GameEnvelope:
    """Generic wrapper that holds any supported game type.

    Attributes:
        game_type: One of ``"quiz"``, ``"pasapalabra"``, ``"true_false"``.
        game: The concrete game instance.
    """

    game_type: str
    game: QuizGame | PasapalabraGame | TrueFalseGame

    def to_dict(self) -> dict[str, Any]:
        return self.game.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameEnvelope:
        """Deserialise a game envelope from a dictionary.

        The ``game_type`` key selects the concrete game class.
        """
        game_type = data.get("game_type", "quiz")
        game_cls = GAME_TYPE_REGISTRY.get(game_type)
        if game_cls is None:
            raise ValueError(
                f"Unknown game_type {game_type!r}. "
                f"Supported: {list(GAME_TYPE_REGISTRY)}"
            )
        game = game_cls.from_dict(data)
        return cls(game_type=game_type, game=game)
