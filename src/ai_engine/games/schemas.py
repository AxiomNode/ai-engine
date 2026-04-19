"""Data models for structured educational games.

All models use Pydantic :class:`~pydantic.BaseModel` for automatic
validation, serialisation, and JSON-schema generation.  Keeping them as
Pydantic models eliminates the manual ``to_dict`` / ``from_dict``
boilerplate and enables first-class OpenAPI schema export.

Compatibility aliases (:meth:`to_dict`, :meth:`from_dict`) are preserved so
existing call sites continue to work without modification.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ------------------------------------------------------------------
# Quiz
# ------------------------------------------------------------------


class QuizQuestion(BaseModel):
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

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Ensure the question text is not empty."""
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("options")
    @classmethod
    def options_min_two(cls, v: list[str]) -> list[str]:
        """Require at least two answer choices."""
        if len(v) < 2:
            raise ValueError("options must contain at least 2 choices")
        return v

    @model_validator(mode="after")
    def correct_index_in_range(self) -> "QuizQuestion":
        """Validate that correct_index refers to an existing option."""
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError(
                f"correct_index {self.correct_index} out of range "
                f"for {len(self.options)} options"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuizQuestion":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


class QuizGame(BaseModel):
    """A complete quiz game definition.

    Attributes:
        game_type: Always ``"quiz"``; included in serialisation for type
            discrimination.
        title: Human-readable title for the quiz.
        questions: Ordered list of quiz questions.
    """

    game_type: Literal["quiz"] = "quiz"
    title: str
    difficulty_percentage: int = Field(default=50, ge=0, le=100)
    questions: list[QuizQuestion] = Field(default_factory=list, min_length=1)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Ensure the title is not empty."""
        if not v:
            raise ValueError("title must not be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuizGame":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


# ------------------------------------------------------------------
# WordPass
# ------------------------------------------------------------------


class WordPassWord(BaseModel):
    """A single reusable WordPass entry.

    Attributes:
        letter: Representative letter for the entry (usually the initial).
            Normalised to upper-case on input.
        hint: Clue or definition read aloud to the player.
        answer: The correct word (starts with or contains *letter*).
        starts_with: True if the answer starts with *letter*;
            False if the answer merely contains *letter*.
    """

    letter: str
    hint: str
    answer: str
    starts_with: bool = True

    @field_validator("letter")
    @classmethod
    def letter_valid(cls, v: str) -> str:
        """Validate that letter is a single alphabetic character and upper-case it."""
        if len(v) != 1 or not v.isalpha():
            raise ValueError(f"letter must be a single alphabetic character, got {v!r}")
        return v.upper()

    @field_validator("hint")
    @classmethod
    def hint_not_empty(cls, v: str) -> str:
        """Ensure the hint is not empty."""
        if not v:
            raise ValueError("hint must not be empty")
        return v

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        """Ensure the answer is not empty."""
        if not v:
            raise ValueError("answer must not be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordPassWord":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


class WordPassGame(BaseModel):
    """A collection of standalone WordPass entries.

    Attributes:
        game_type: Always ``"word-pass"``; included in serialisation for
            type discrimination.
        title: Human-readable title.
        words: List of independent WordPass entries.
    """

    game_type: Literal["word-pass"] = "word-pass"
    title: str
    difficulty_percentage: int = Field(default=50, ge=0, le=100)
    words: list[WordPassWord] = Field(default_factory=list, min_length=1)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Ensure the title is not empty."""
        if not v:
            raise ValueError("title must not be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordPassGame":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


# ------------------------------------------------------------------
# True / False
# ------------------------------------------------------------------


class TrueFalseStatement(BaseModel):
    """A single true/false statement.

    Attributes:
        statement: The assertion to evaluate.
        is_true: Whether the statement is true.
        explanation: Pedagogical explanation.
    """

    statement: str
    is_true: bool
    explanation: str = ""

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        """Ensure the statement text is not empty."""
        if not v:
            raise ValueError("statement must not be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrueFalseStatement":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


class TrueFalseGame(BaseModel):
    """A complete true/false game.

    Attributes:
        game_type: Always ``"true_false"``; included in serialisation for
            type discrimination.
        title: Human-readable title.
        statements: Ordered list of true/false statements.
    """

    game_type: Literal["true_false"] = "true_false"
    title: str
    difficulty_percentage: int = Field(default=50, ge=0, le=100)
    statements: list[TrueFalseStatement] = Field(default_factory=list, min_length=1)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Ensure the title is not empty."""
        if not v:
            raise ValueError("title must not be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Compatibility alias for :meth:`model_dump`.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrueFalseGame":
        """Deserialise from a plain dictionary.

        Compatibility alias for :meth:`model_validate`.
        """
        return cls.model_validate(data)


# ------------------------------------------------------------------
# Envelope (wraps any game type)
# ------------------------------------------------------------------

GAME_TYPE_REGISTRY: dict[str, type[Union[QuizGame, WordPassGame, TrueFalseGame]]] = {
    "quiz": QuizGame,
    "word-pass": WordPassGame,
    "true_false": TrueFalseGame,
}

GAME_TYPE_ALIASES: dict[str, str] = {
    "educational-game": "quiz",
    "educational_game": "quiz",
    "wordpass": "word-pass",
    "word_pass": "word-pass",
    "true-false": "true_false",
    "truefalse": "true_false",
}


def _normalize_game_type(value: Any) -> str:
    raw = str(value or "quiz").strip().lower()
    return GAME_TYPE_ALIASES.get(raw, raw)


class GameEnvelope(BaseModel):
    """Generic wrapper that holds any supported game type.

    Attributes:
        game_type: One of ``"quiz"``, ``"word-pass"``, ``"true_false"``.
        game: The concrete game instance.
    """

    game_type: str
    game: Union[QuizGame, WordPassGame, TrueFalseGame]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the wrapped game to a plain dictionary, including ``game_type``."""
        return self.game.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameEnvelope":
        """Deserialise a game envelope from a dictionary.

        The ``game_type`` key selects the concrete game class via
        :data:`GAME_TYPE_REGISTRY`.

        Args:
            data: Dictionary with at least a ``game_type`` key.

        Returns:
            A fully validated :class:`GameEnvelope` instance.

        Raises:
            ValueError: If ``game_type`` is not in the registry.
        """
        game_type = _normalize_game_type(data.get("game_type", "quiz"))
        game_cls = GAME_TYPE_REGISTRY.get(game_type)
        if game_cls is None:
            raise ValueError(
                f"Unknown game_type {game_type!r}. "
                f"Supported: {list(GAME_TYPE_REGISTRY)}"
            )
        normalized_data = dict(data)
        normalized_data["game_type"] = game_type
        game = game_cls.model_validate(normalized_data)
        return cls(game_type=game_type, game=game)
