"""SDK data models for generated game objects.

These models are designed for clients consuming ``POST /generate`` responses.
They provide a stable and typed contract independent from internal API schemas.
"""

from __future__ import annotations

from typing import Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from ai_engine.sdk.languages import LanguageCode, get_language_info


def _normalize_language(language: LanguageCode | str) -> LanguageCode:
    """Normalize user-provided language values to LanguageCode."""
    return language if isinstance(language, LanguageCode) else LanguageCode(language)


class GenerationMetadata(BaseModel):
    """Shared metadata attached to generated SDK objects.

    Attributes:
        generation_id: Stable identifier for the generated artifact.
        language: ISO 639-1 language code.
        language_id: Stable language identifier (e.g. ``lang-es``).
    """

    generation_id: str = Field(default_factory=lambda: f"gen-{uuid4().hex}")
    language: LanguageCode = LanguageCode.ES
    language_id: str | None = None
    difficulty_percentage: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="after")
    def ensure_language_id(self) -> "GenerationMetadata":
        """Populate language_id from the language catalog when omitted."""
        if self.language_id is None:
            self.language_id = get_language_info(self.language).language_id
        return self


class MultipleChoiceQuestion(BaseModel):
    """Quiz question with up to four answer choices."""

    question_type: Literal["multiple_choice"] = "multiple_choice"
    question: str
    options: list[str] = Field(min_length=2, max_length=4)
    correct_index: int
    explanation: str = ""

    @model_validator(mode="after")
    def validate_correct_index(self) -> "MultipleChoiceQuestion":
        """Validate that correct_index points to a valid option."""
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError("correct_index out of range for options")
        return self


class TrueFalseQuestion(BaseModel):
    """Quiz question represented as a true/false statement."""

    question_type: Literal["true_false"] = "true_false"
    statement: str
    is_true: bool
    explanation: str = ""


class BestAnswerQuestion(BaseModel):
    """Quiz question where one option is the most correct answer."""

    question_type: Literal["best_answer"] = "best_answer"
    question: str
    options: list[str] = Field(min_length=2, max_length=4)
    best_index: int
    explanation: str = ""

    @model_validator(mode="after")
    def validate_best_index(self) -> "BestAnswerQuestion":
        """Validate that best_index points to a valid option."""
        if not 0 <= self.best_index < len(self.options):
            raise ValueError("best_index out of range for options")
        return self


QuizQuestion = Union[MultipleChoiceQuestion, TrueFalseQuestion, BestAnswerQuestion]


class GeneratedQuiz(BaseModel):
    """SDK model for quiz-like generated games.

    This model supports multiple-choice, true/false, and best-answer question
    variants under a single quiz contract.
    """

    model_type: Literal["quiz"] = "quiz"
    metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    title: str
    topic: str
    questions: list[QuizQuestion] = Field(default_factory=list)

    @classmethod
    def from_generate_payload(
        cls,
        payload: dict[str, Any],
        language: LanguageCode | str = LanguageCode.ES,
    ) -> "GeneratedQuiz":
        """Create a quiz model from ``POST /generate`` API payload.

        Accepts both quiz and true_false API outputs and normalizes them into
        the SDK quiz model.
        """

        game_type = str(payload.get("game_type", "quiz"))
        game = payload.get("game") if isinstance(payload.get("game"), dict) else payload

        if not isinstance(game, dict):
            raise ValueError(
                "Invalid generate payload: game object must be a dictionary"
            )

        metadata = GenerationMetadata(language=_normalize_language(language))
        if isinstance(payload.get("metadata"), dict):
            incoming_meta = payload["metadata"]
            if "difficulty_percentage" in incoming_meta:
                metadata.difficulty_percentage = int(
                    incoming_meta["difficulty_percentage"]
                )

        if "difficulty_percentage" in game:
            metadata.difficulty_percentage = int(game["difficulty_percentage"])

        if game_type == "true_false":
            statements = game.get("statements", [])
            tf_questions: list[QuizQuestion] = [
                TrueFalseQuestion(
                    statement=str(item.get("statement", "")),
                    is_true=bool(item.get("is_true", False)),
                    explanation=str(item.get("explanation", "")),
                )
                for item in statements
                if isinstance(item, dict)
            ]
            return cls(
                metadata=metadata,
                title=str(game.get("title", "Untitled Quiz")),
                topic=str(game.get("topic", "General")),
                questions=tf_questions,
            )

        if game_type != "quiz":
            raise ValueError(
                f"Payload game_type {game_type!r} cannot be mapped to GeneratedQuiz"
            )

        raw_questions = game.get("questions", [])
        questions: list[QuizQuestion] = []
        for item in raw_questions:
            if not isinstance(item, dict):
                continue

            question_type = str(item.get("question_type", "multiple_choice"))
            if question_type == "true_false":
                questions.append(
                    TrueFalseQuestion(
                        statement=str(item.get("statement", item.get("question", ""))),
                        is_true=bool(item.get("is_true", False)),
                        explanation=str(item.get("explanation", "")),
                    )
                )
                continue

            if question_type == "best_answer":
                raw_best_index = item.get("best_index", item.get("correct_index", 0))
                best_index = int(raw_best_index) if raw_best_index is not None else 0
                questions.append(
                    BestAnswerQuestion(
                        question=str(item.get("question", "")),
                        options=[str(opt) for opt in item.get("options", [])],
                        best_index=best_index,
                        explanation=str(item.get("explanation", "")),
                    )
                )
                continue

            raw_correct_index = item.get("correct_index", 0)
            correct_index = (
                int(raw_correct_index) if raw_correct_index is not None else 0
            )
            questions.append(
                MultipleChoiceQuestion(
                    question=str(item.get("question", "")),
                    options=[str(opt) for opt in item.get("options", [])],
                    correct_index=correct_index,
                    explanation=str(item.get("explanation", "")),
                )
            )

        return cls(
            metadata=metadata,
            title=str(game.get("title", "Untitled Quiz")),
            topic=str(game.get("topic", "General")),
            questions=questions,
        )


class PasapalabraEntry(BaseModel):
    """Single Pasapalabra item with definition and letter relation."""

    letter: str
    relation: Literal["starts_with", "contains"] = "starts_with"
    word: str
    definition: str

    @field_validator("letter")
    @classmethod
    def validate_letter(cls, value: str) -> str:
        """Normalize and validate the Pasapalabra letter."""
        normalized = value.strip().upper()
        if len(normalized) != 1 or not normalized.isalpha():
            raise ValueError("letter must be a single alphabetic character")
        return normalized


class GeneratedPasapalabra(BaseModel):
    """SDK model for generated Pasapalabra games."""

    model_type: Literal["pasapalabra"] = "pasapalabra"
    metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    title: str
    topic: str
    entries: list[PasapalabraEntry] = Field(default_factory=list)

    @classmethod
    def from_generate_payload(
        cls,
        payload: dict[str, Any],
        language: LanguageCode | str = LanguageCode.ES,
    ) -> "GeneratedPasapalabra":
        """Create a pasapalabra model from ``POST /generate`` API payload."""

        game_type = str(payload.get("game_type", "pasapalabra"))
        if game_type != "pasapalabra":
            raise ValueError(
                f"Payload game_type {game_type!r} cannot be mapped to GeneratedPasapalabra"
            )

        game = payload.get("game") if isinstance(payload.get("game"), dict) else payload
        if not isinstance(game, dict):
            raise ValueError(
                "Invalid generate payload: game object must be a dictionary"
            )

        metadata = GenerationMetadata(language=_normalize_language(language))
        if isinstance(payload.get("metadata"), dict):
            incoming_meta = payload["metadata"]
            if "difficulty_percentage" in incoming_meta:
                metadata.difficulty_percentage = int(
                    incoming_meta["difficulty_percentage"]
                )

        if "difficulty_percentage" in game:
            metadata.difficulty_percentage = int(game["difficulty_percentage"])
        entries: list[PasapalabraEntry] = []

        for item in game.get("words", []):
            if not isinstance(item, dict):
                continue
            starts_with = bool(item.get("starts_with", True))
            entries.append(
                PasapalabraEntry(
                    letter=str(item.get("letter", "")),
                    relation="starts_with" if starts_with else "contains",
                    word=str(item.get("answer", "")),
                    definition=str(item.get("hint", "")),
                )
            )

        return cls(
            metadata=metadata,
            title=str(game.get("title", "Untitled Pasapalabra")),
            topic=str(game.get("topic", "General")),
            entries=entries,
        )


GeneratedGameModel = Union[GeneratedQuiz, GeneratedPasapalabra]


def parse_generate_response(
    payload: dict[str, Any],
    language: LanguageCode | str = LanguageCode.ES,
) -> GeneratedGameModel:
    """Parse ``POST /generate`` response into a typed SDK model."""

    game_type = str(payload.get("game_type", "quiz"))
    if game_type in {"quiz", "true_false"}:
        return GeneratedQuiz.from_generate_payload(payload, language=language)
    if game_type == "pasapalabra":
        return GeneratedPasapalabra.from_generate_payload(payload, language=language)
    raise ValueError(f"Unsupported game_type in generate response: {game_type!r}")
