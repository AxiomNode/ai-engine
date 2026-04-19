"""High-level generator that combines RAG context with an LLM to produce
structured educational game definitions.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any

from ai_engine.games.catalog import get_game_type_profile
from ai_engine.games.prompts import get_prompt
from ai_engine.games.schemas import GameEnvelope
from ai_engine.llm.llama_client import LlamaClient
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.utils import extract_json_from_text

logger = logging.getLogger(__name__)

_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Return only one valid JSON object that strictly matches the "
    "requested schema. Do not include markdown, comments, or extra text."
)

_SEMANTIC_RETRY_SUFFIX = (
    "\n\nIMPORTANT CORRECTION: Your previous JSON was syntactically valid but "
    "failed schema validation because: {validation_error}. Regenerate the FULL "
    "JSON object from scratch. The title field is mandatory and must be a short, "
    "non-empty string relevant to the topic. Every required array and field must "
    "be present and non-empty. Return ONLY the corrected JSON object."
)

# Keep retry bounded to avoid very long second attempts under constrained CPU inference.
_JSON_RETRY_TOKEN_EXTRA_MIN = 128
_JSON_RETRY_TOKEN_EXTRA_MAX = 256
_JSON_RETRY_TOKEN_MULTIPLIER = 1.25
_SKIP_JSON_RETRY_IF_FIRST_LLM_MS = 90_000.0

_INLINE_OPTIONS_RE = re.compile(
    r"\s+[A-D][\)\.:\-]\s*[^\n]+(?:\s+[A-D][\)\.:\-]\s*[^\n]+)+",
    re.IGNORECASE,
)
_LEADING_OPTION_LABEL_RE = re.compile(r"^[A-Z][\)\.:\-]\s*")
_WORD_PASS_TITLE_RE = re.compile(r'"title"\s*:\s*"(?P<title>[^"]+)"', re.IGNORECASE)
_WORD_PASS_LOOSE_ENTRY_RE = re.compile(
    r'"letter"\s*:\s*"(?P<letter>[^"]+)"\s*,\s*'
    r'"hint"\s*:\s*"(?P<hint>[^"]+)"\s*,\s*'
    r'"answer"\s*:\s*"(?P<answer>[^"]+)"\s*,\s*'
    r'"starts_with"\s*:\s*(?P<starts_with>true|false)',
    re.IGNORECASE | re.DOTALL,
)
_QUIZ_TITLE_RE = re.compile(r'"title"\s*:\s*"(?P<title>[^"]+)"', re.IGNORECASE)
_QUIZ_LOOSE_ENTRY_RE = re.compile(
    r'"question"\s*:\s*"(?P<question>[^"]+)"\s*,\s*'
    r'"options"\s*:\s*\[(?P<options>[^\]]+)\]\s*,\s*'
    r'"correct_index"\s*:\s*(?P<correct_index>\d+)\s*,\s*'
    r'"explanation"\s*:\s*"(?P<explanation>[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
_QUIZ_LOOSE_OPTION_RE = re.compile(r'"([^"]+)"')

_FALLBACK_TITLES = {
    "quiz": {
        "es": "Quiz educativo",
        "en": "Educational Quiz",
    },
    "word-pass": {
        "es": "Rosco educativo",
        "en": "Educational Word Pass",
    },
    "true_false": {
        "es": "Verdadero o falso educativo",
        "en": "Educational True or False",
    },
}

_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "ambiguas",
    "comparativa",
    "con",
    "conceptos",
    "contexto",
    "de",
    "del",
    "el",
    "en",
    "esenciales",
    "evitar",
    "falso",
    "for",
    "how",
    "historica",
    "la",
    "las",
    "los",
    "multiple",
    "opcion",
    "pass",
    "por",
    "preguntas",
    "que",
    "respuestas",
    "sentido",
    "sobre",
    "the",
    "trivia",
    "un",
    "una",
    "verdadero",
    "what",
    "with",
    "word",
    "y",
}

_BROAD_TOPIC_KEYWORDS = {
    "animals",
    "anime",
    "animations",
    "art",
    "board",
    "books",
    "cartoon",
    "cartoons",
    "comics",
    "computers",
    "entertainment",
    "film",
    "gadgets",
    "games",
    "general",
    "geography",
    "history",
    "japanese",
    "knowledge",
    "manga",
    "mathematics",
    "music",
    "musicals",
    "nature",
    "science",
    "sports",
    "television",
    "theatres",
    "video",
}

_TOPIC_INSTRUCTION_MARKERS = {
    "ambiguas",
    "comparativa",
    "conceptos",
    "contexto",
    "esenciales",
    "evitar",
    "historica",
    "multiple",
    "opcion",
    "preguntas",
    "respuestas",
    "sentido",
    "trivia",
    "verdadero",
    "word",
}

_TOPIC_KEYWORD_MIN_LENGTH = 4
_TOPIC_ROOT_MIN_LENGTH = 5
_TOPIC_ROOT_PREFIX_LENGTH = 6


class GameGenerator:
    """Orchestrates RAG retrieval + LLM generation for educational games.

    Args:
        rag_pipeline: A configured :class:`RAGPipeline` used to build
            context from ingested documents.
        llm_client: A :class:`LlamaClient` (or any object with a
            ``generate(prompt, max_tokens)`` method).
        default_language: Default output language (ISO 639-1).
        default_max_tokens: Default token budget for generation.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm_client: LlamaClient,
        default_language: str = "es",
        default_max_tokens: int = 1024,
    ) -> None:
        if rag_pipeline is None:
            raise ValueError("rag_pipeline is required")
        if llm_client is None:
            raise ValueError("llm_client is required")

        self.rag_pipeline = rag_pipeline
        self.llm_client = llm_client
        self.default_language = default_language
        self.default_max_tokens = default_max_tokens
        self.last_run_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
        difficulty_percentage: int = 50,
        num_questions: int = 5,
        letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> GameEnvelope:
        """Generate a structured educational game.

        1. Retrieves relevant context from the RAG pipeline.
        2. Builds a prompt tailored to *game_type*.
        3. Sends it to the LLM.
        4. Extracts and validates the JSON response.

        Args:
            query: Search query for RAG retrieval.
            game_type: ``"quiz"``, ``"word-pass"``, or ``"true_false"``.
            language: Output language (defaults to ``self.default_language``).
            num_questions: Number of questions/statements.
            letters: Comma-separated letters (used only by word-pass).
            max_tokens: Override default token budget.
            top_k: Override RAG retrieval count.

        Returns:
            A :class:`GameEnvelope` wrapping the concrete game.

        Raises:
            ValueError: If JSON extraction or parsing fails.
        """
        # 1. Retrieve context once, then delegate to context-based generation.
        lang = language or self.default_language
        profile = get_game_type_profile(game_type)
        context = self.rag_pipeline.build_context(
            query,
            top_k=top_k if top_k is not None else profile.retrieval_top_k,
            max_chars=profile.context_char_limit,
            metadata_preferences={
                "language": lang,
                "game_type": game_type,
            },
        )
        logger.debug("RAG context length: %d chars", len(context))

        return await self.generate_from_context(
            context=context,
            game_type=game_type,
            topic=query,
            language=lang,
            difficulty_percentage=difficulty_percentage,
            num_questions=num_questions,
            letters=letters,
            max_tokens=max_tokens,
        )

    async def generate_from_context(
        self,
        context: str,
        game_type: str = "quiz",
        *,
        topic: str | None = None,
        language: str | None = None,
        difficulty_percentage: int = 50,
        num_questions: int = 5,
        letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
        max_tokens: int | None = None,
    ) -> GameEnvelope:
        """Generate a game from prebuilt context.

        This avoids an additional retrieval pass when the caller already
        computed the RAG context.
        """
        lang = language or self.default_language
        tokens = max_tokens or self.default_max_tokens

        prompt = get_prompt(
            game_type=game_type,
            context=context,
            topic=topic,
            language=lang,
            difficulty_percentage=difficulty_percentage,
            num_questions=num_questions,
            letters=letters,
        )
        logger.debug("Prompt length: %d chars", len(prompt))

        try:
            json_text, raw_output, run_metrics = await self._generate_json_with_retry(
                prompt, tokens
            )
            data = json.loads(json_text)
            data, run_metrics = await self._normalize_generated_payload_with_retry(
                data=data,
                prompt=prompt,
                raw_output=raw_output,
                run_metrics=run_metrics,
                game_type=game_type,
                language=lang,
                topic=topic,
                difficulty_percentage=difficulty_percentage,
                max_tokens=tokens,
            )
        except ValueError:
            if game_type != "word-pass":
                raise
            data = self._build_fallback_word_pass_payload(
                topic=topic,
                language=lang,
                difficulty_percentage=difficulty_percentage,
                num_questions=num_questions,
            )
            run_metrics = locals().get("run_metrics", {})
            run_metrics = dict(run_metrics)
            run_metrics["word_pass_fallback_used"] = True
        self.last_run_metrics = run_metrics
        if "game_type" not in data:
            data["game_type"] = game_type
        return GameEnvelope.from_dict(data)

    async def generate_raw(
        self,
        query: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
        difficulty_percentage: int = 50,
        num_questions: int = 5,
        letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Same as :meth:`generate` but returns the raw parsed dict.

        Useful when you want the JSON directly without validation
        through the dataclass models.
        """
        lang = language or self.default_language
        tokens = max_tokens or self.default_max_tokens
        profile = get_game_type_profile(game_type)

        context = self.rag_pipeline.build_context(
            query,
            top_k=top_k if top_k is not None else profile.retrieval_top_k,
            max_chars=profile.context_char_limit,
            metadata_preferences={
                "language": lang,
                "game_type": game_type,
            },
        )
        prompt = get_prompt(
            game_type=game_type,
            context=context,
            topic=query,
            language=lang,
            difficulty_percentage=difficulty_percentage,
            num_questions=num_questions,
            letters=letters,
        )
        json_text, raw_output, run_metrics = await self._generate_json_with_retry(
            prompt, tokens
        )
        data = json.loads(json_text)
        data, run_metrics = await self._normalize_generated_payload_with_retry(
            data=data,
            prompt=prompt,
            raw_output=raw_output,
            run_metrics=run_metrics,
            game_type=game_type,
            language=lang,
            topic=query,
            difficulty_percentage=difficulty_percentage,
            max_tokens=tokens,
        )
        self.last_run_metrics = run_metrics
        return data

    def _normalize_generated_payload(
        self,
        *,
        data: Any,
        game_type: str,
        language: str,
        topic: str | None = None,
        difficulty_percentage: int,
    ) -> dict[str, Any]:
        """Normalize generated payload fields for downstream contracts."""
        normalized = self._flatten_generated_payload(data, game_type)
        normalized_game_type = (
            str(normalized.get("game_type") or game_type or "quiz").strip().lower()
        )
        if normalized_game_type in {"educational-game", "educational_game"}:
            normalized_game_type = "quiz"
        elif normalized_game_type in {"wordpass", "word_pass"}:
            normalized_game_type = "word-pass"
        elif normalized_game_type in {"true-false", "truefalse"}:
            normalized_game_type = "true_false"
        normalized["game_type"] = normalized_game_type
        normalized["title"] = self._normalize_title(
            normalized.get("title"), normalized_game_type, language
        )
        normalized["difficulty_percentage"] = max(0, min(100, difficulty_percentage))

        if normalized_game_type == "quiz":
            normalized["questions"] = self._normalize_quiz_questions(
                self._resolve_payload_items(normalized, "questions")
            )
        elif normalized_game_type == "word-pass":
            normalized["words"] = self._normalize_word_pass_words(
                self._resolve_payload_items(normalized, "words")
            )
            normalized["words"] = self._ensure_word_pass_topic_signal(
                normalized["words"],
                topic=topic,
                language=language,
            )
        elif normalized_game_type == "true_false":
            normalized["statements"] = self._normalize_true_false_statements(
                self._resolve_payload_items(normalized, "statements")
            )
        self._validate_topic_alignment(
            normalized=normalized,
            game_type=normalized_game_type,
            topic=topic,
        )
        return normalized

    def _normalize_title(self, title: Any, game_type: str, language: str) -> str:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            language_key = (language or "en").strip().lower()
            if language_key not in {"es", "en"}:
                language_key = "en"
            return _FALLBACK_TITLES.get(game_type, {}).get(language_key, "Educational Game")
        return normalized_title

    def _normalize_quiz_questions(self, raw_questions: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("Generated quiz has no questions")

        cleaned_questions: list[dict[str, Any]] = []
        for index, item in enumerate(raw_questions):
            if not isinstance(item, dict):
                raise ValueError(f"Quiz question {index} is not an object")

            question = self._clean_quiz_question_text(str(item.get("question", "")))
            if not question:
                raise ValueError(f"Quiz question {index} is missing text")

            raw_options = item.get("options", item.get("answers"))
            options = self._normalize_quiz_options(raw_options)
            if not options:
                raise ValueError(f"Quiz question {index} is missing options")
            if len(options) < 2:
                raise ValueError(f"Quiz question {index} must contain at least 2 options")

            correct_index = self._resolve_correct_index(item, options, index)
            if correct_index < 0 or correct_index >= len(options):
                raise ValueError(f"Quiz question {index} has out-of-range correct_index")

            cleaned_questions.append(
                {
                    "question": question,
                    "options": options,
                    "correct_index": correct_index,
                    "explanation": str(item.get("explanation", "") or "").strip(),
                }
            )

        return cleaned_questions

    def _normalize_word_pass_words(self, raw_words: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_words, list) or not raw_words:
            raise ValueError("Generated word-pass has no words")

        cleaned_words: list[dict[str, Any]] = []
        for index, item in enumerate(raw_words):
            if not isinstance(item, dict):
                raise ValueError(f"Word-pass entry {index} is not an object")

            letter = self._extract_word_pass_letter(
                item.get("letter", item.get("initial", item.get("key_letter", "")))
            )
            answer = str(item.get("answer", item.get("word", item.get("solution", "")))).strip()
            if not letter and answer:
                letter = self._extract_word_pass_letter(answer)
            if len(letter) != 1 or not letter.isalpha():
                raise ValueError(f"Word-pass entry {index} has invalid letter")

            hint = str(item.get("hint", item.get("definition", item.get("clue", "")))).strip()
            if not hint:
                raise ValueError(f"Word-pass entry {index} is missing hint")

            if not answer:
                raise ValueError(f"Word-pass entry {index} is missing answer")

            starts_with = self._resolve_word_pass_relation(item, letter, answer)

            cleaned_words.append(
                {
                    "letter": letter,
                    "hint": hint,
                    "answer": answer,
                    "starts_with": starts_with,
                }
            )

        return cleaned_words

    def _normalize_true_false_statements(
        self, raw_statements: Any
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_statements, list) or not raw_statements:
            raise ValueError("Generated true/false game has no statements")

        cleaned_statements: list[dict[str, Any]] = []
        for index, item in enumerate(raw_statements):
            if not isinstance(item, dict):
                raise ValueError(f"True/false statement {index} is not an object")

            statement = str(item.get("statement", "")).strip()
            if not statement:
                raise ValueError(f"True/false statement {index} is missing text")

            is_true = self._coerce_bool(item.get("is_true"))
            if is_true is None:
                raise ValueError(f"True/false statement {index} has invalid is_true")

            cleaned_statements.append(
                {
                    "statement": statement,
                    "is_true": is_true,
                    "explanation": str(item.get("explanation", "") or "").strip(),
                }
            )

        return cleaned_statements

    def _clean_quiz_question_text(self, question: str) -> str:
        """Strip embedded options from question title and keep only prompt text."""
        text = (question or "").strip()
        if not text:
            return text

        # Keep first logical line if options were appended as multiline content.
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), text
        )
        sanitized = _INLINE_OPTIONS_RE.sub("", first_line).strip()

        # Fallback: when parser removed all text, keep first line untouched.
        return sanitized or first_line

    def _flatten_generated_payload(
        self,
        data: Any,
        fallback_game_type: str,
    ) -> dict[str, Any]:
        if isinstance(data, list):
            return {
                "game_type": fallback_game_type,
                self._collection_key_for_game_type(fallback_game_type): data,
            }

        if not isinstance(data, dict):
            raise ValueError(f"Generated {fallback_game_type} payload is not an object")

        normalized = dict(data)
        nested_game = normalized.get("game")
        if isinstance(nested_game, dict):
            flattened = dict(nested_game)
            flattened.setdefault(
                "game_type",
                normalized.get("game_type") or nested_game.get("game_type") or fallback_game_type,
            )
            if "metadata" in normalized:
                flattened["metadata"] = normalized["metadata"]
            return flattened
        if isinstance(nested_game, list):
            return {
                "game_type": normalized.get("game_type") or fallback_game_type,
                self._collection_key_for_game_type(fallback_game_type): nested_game,
            }
        return normalized

    def _resolve_payload_items(
        self,
        normalized: dict[str, Any],
        primary_key: str,
    ) -> Any:
        candidates = {
            "questions": ("questions", "items", "entries", "data", "payload"),
            "words": ("words", "entries", "items", "questions", "data", "payload"),
            "statements": ("statements", "items", "questions", "data", "payload"),
        }
        for key in candidates.get(primary_key, (primary_key,)):
            value = normalized.get(key)
            if isinstance(value, list) and value:
                return value
            if isinstance(value, dict) and self._looks_like_single_item(value, primary_key):
                return [value]
        if self._looks_like_single_item(normalized, primary_key):
            return [normalized]
        return normalized.get(primary_key)

    def _collection_key_for_game_type(self, game_type: str) -> str:
        normalized_game_type = str(game_type or "quiz").strip().lower()
        if normalized_game_type == "word-pass":
            return "words"
        if normalized_game_type == "true_false":
            return "statements"
        return "questions"

    def _looks_like_single_item(self, value: dict[str, Any], primary_key: str) -> bool:
        if primary_key == "questions":
            return bool(value.get("question")) and isinstance(
                value.get("options", value.get("answers")),
                list,
            )
        if primary_key == "words":
            return bool(
                value.get("answer", value.get("word", value.get("solution")))
            ) and bool(
                value.get("letter")
                or value.get("initial")
                or value.get("key_letter")
                or value.get("hint")
                or value.get("definition")
                or value.get("clue")
            )
        if primary_key == "statements":
            return bool(value.get("statement", value.get("question"))) and (
                "is_true" in value
            )
        return False

    def _extract_word_pass_letter(self, value: Any) -> str:
        text = str(value or "").strip().upper()
        return next((char for char in text if char.isalpha()), "")

    def _normalize_quiz_options(self, raw_options: Any) -> list[str]:
        if not isinstance(raw_options, list):
            return []

        options: list[str] = []
        for raw_option in raw_options:
            if isinstance(raw_option, dict):
                text = raw_option.get("text", raw_option.get("label", raw_option.get("option", "")))
            else:
                text = raw_option
            normalized = self._normalize_option_text(str(text or ""))
            if normalized:
                options.append(normalized)
        return options

    def _normalize_option_text(self, text: str) -> str:
        return _LEADING_OPTION_LABEL_RE.sub("", text.strip()).strip()

    def _resolve_correct_index(
        self,
        item: dict[str, Any],
        options: list[str],
        index: int,
    ) -> int:
        raw_correct = item.get(
            "correct_index",
            item.get(
                "correct_answer_index",
                item.get(
                    "answer_index",
                    item.get(
                        "correct_option_index",
                        item.get("best_index", item.get("correct_answer", item.get("answer"))),
                    ),
                ),
            ),
        )
        if raw_correct is None:
            raise ValueError(f"Quiz question {index} has invalid correct_index")

        if isinstance(raw_correct, int):
            return raw_correct

        text_value = str(raw_correct).strip()
        if not text_value:
            raise ValueError(f"Quiz question {index} has invalid correct_index")

        normalized_text = self._normalize_option_text(text_value)
        if len(normalized_text) == 1 and normalized_text.upper() in {"A", "B", "C", "D"}:
            return ord(normalized_text.upper()) - ord("A")

        for option_index, option in enumerate(options):
            if self._normalize_match_text(option) == self._normalize_match_text(normalized_text):
                return option_index

        if text_value.isdigit():
            return int(text_value)

        raise ValueError(f"Quiz question {index} has invalid correct_index")

    def _resolve_word_pass_relation(
        self,
        item: dict[str, Any],
        letter: str,
        answer: str,
    ) -> bool:
        starts_with = self._coerce_bool(item.get("starts_with"))
        if starts_with is not None:
            return starts_with

        relation = str(item.get("relation", item.get("match_type", ""))).strip().lower()
        if relation in {"starts_with", "starts-with", "starts with", "prefix"}:
            return True
        if relation in {"contains", "contain", "includes", "substring"}:
            return False

        normalized_letter = self._normalize_match_text(letter)
        normalized_answer = self._normalize_match_text(answer)
        if normalized_letter and normalized_answer.startswith(normalized_letter):
            return True
        return False

    def _coerce_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "si", "sí"}:
                return True
            if normalized in {"false", "no", "n", "0"}:
                return False
        return None

    def _validate_topic_alignment(
        self,
        *,
        normalized: dict[str, Any],
        game_type: str,
        topic: str | None,
    ) -> None:
        topic_keywords = self._extract_topic_keywords(topic)
        if not topic_keywords:
            return
        if self._should_skip_topic_alignment(topic, topic_keywords):
            return
        topic_roots = self._extract_topic_roots(topic_keywords)

        if game_type == "quiz":
            for index, item in enumerate(normalized.get("questions", [])):
                bundle = " ".join(
                    [
                        str(item.get("question", "")),
                        str(item.get("explanation", "")),
                        " ".join(str(option) for option in item.get("options", [])),
                    ]
                )
                if not self._contains_topic_keyword(
                    bundle, topic_keywords, topic_roots
                ):
                    raise ValueError(
                        f"Quiz question {index} is off-topic for requested topic: {topic}"
                    )
            return

        if game_type == "true_false":
            for index, item in enumerate(normalized.get("statements", [])):
                bundle = " ".join(
                    [
                        str(item.get("statement", "")),
                        str(item.get("explanation", "")),
                    ]
                )
                if not self._contains_topic_keyword(
                    bundle, topic_keywords, topic_roots
                ):
                    raise ValueError(
                        f"True/false statement {index} is off-topic for requested topic: {topic}"
                    )

    def _prune_topic_aligned_items(
        self,
        *,
        normalized: dict[str, Any],
        game_type: str,
        topic: str | None,
    ) -> tuple[dict[str, Any], int]:
        topic_keywords = self._extract_topic_keywords(topic)
        if not topic_keywords:
            return normalized, 0
        if self._should_skip_topic_alignment(topic, topic_keywords):
            return normalized, 0
        topic_roots = self._extract_topic_roots(topic_keywords)

        pruned = dict(normalized)
        removed = 0

        if game_type == "quiz":
            kept_questions: list[dict[str, Any]] = []
            for item in normalized.get("questions", []):
                bundle = " ".join(
                    [
                        str(item.get("question", "")),
                        str(item.get("explanation", "")),
                        " ".join(str(option) for option in item.get("options", [])),
                    ]
                )
                if self._contains_topic_keyword(bundle, topic_keywords, topic_roots):
                    kept_questions.append(item)
                else:
                    removed += 1
            pruned["questions"] = kept_questions
            return pruned, removed

        if game_type == "true_false":
            kept_statements: list[dict[str, Any]] = []
            for item in normalized.get("statements", []):
                bundle = " ".join(
                    [
                        str(item.get("statement", "")),
                        str(item.get("explanation", "")),
                    ]
                )
                if self._contains_topic_keyword(bundle, topic_keywords, topic_roots):
                    kept_statements.append(item)
                else:
                    removed += 1
            pruned["statements"] = kept_statements
            return pruned, removed

        return pruned, 0

    def _extract_topic_keywords(self, topic: str | None) -> list[str]:
        if not isinstance(topic, str) or not topic.strip():
            return []

        normalized_topic = self._normalize_match_text(topic)
        keywords: list[str] = []
        for token in normalized_topic.split():
            if len(token) < _TOPIC_KEYWORD_MIN_LENGTH:
                continue
            if token in _TOPIC_STOPWORDS:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords

    def _extract_topic_roots(self, keywords: list[str]) -> list[str]:
        roots: list[str] = []
        for keyword in keywords:
            if len(keyword) < _TOPIC_ROOT_MIN_LENGTH:
                continue
            root = keyword[: min(len(keyword), _TOPIC_ROOT_PREFIX_LENGTH)]
            if root not in roots:
                roots.append(root)
        return roots

    def _should_skip_topic_alignment(
        self,
        topic: str | None,
        keywords: list[str],
    ) -> bool:
        if not isinstance(topic, str) or not topic.strip() or not keywords:
            return False

        normalized_topic = self._normalize_match_text(topic)
        topic_tokens = set(normalized_topic.split())
        if not topic_tokens.intersection(_TOPIC_INSTRUCTION_MARKERS):
            return False

        return all(keyword in _BROAD_TOPIC_KEYWORDS for keyword in keywords)

    def _contains_topic_keyword(
        self,
        text: str,
        keywords: list[str],
        roots: list[str] | None = None,
    ) -> bool:
        return bool(
            self._matched_topic_signals(text, keywords, roots or self._extract_topic_roots(keywords))
        )

    def _matched_topic_signals(
        self,
        text: str,
        keywords: list[str],
        roots: list[str],
    ) -> set[str]:
        normalized_text = self._normalize_match_text(text)
        tokens = normalized_text.split()
        matches: set[str] = set()

        for keyword in keywords:
            if keyword in normalized_text:
                matches.add(keyword)

        for root in roots:
            if any(token.startswith(root) for token in tokens):
                matches.add(root)

        return matches

    def _normalize_match_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or ""))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()

    async def _normalize_generated_payload_with_retry(
        self,
        *,
        data: Any,
        prompt: str,
        raw_output: str,
        run_metrics: dict[str, Any],
        game_type: str,
        language: str,
        topic: str | None,
        difficulty_percentage: int,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics = self._with_semantic_retry_metrics(run_metrics)
        initial_salvaged: tuple[dict[str, Any], int] | None = None
        try:
            normalized = self._normalize_generated_payload(
                data=data,
                game_type=game_type,
                language=language,
                topic=topic,
                difficulty_percentage=difficulty_percentage,
            )
            return normalized, metrics
        except ValueError as exc:
            initial_salvaged = self._try_salvage_topic_filtered_payload(
                data=data,
                game_type=game_type,
                language=language,
                topic=topic,
                difficulty_percentage=difficulty_percentage,
            )
            llm_total_ms = float(metrics.get("llm_total_ms", 0.0))
            if llm_total_ms >= _SKIP_JSON_RETRY_IF_FIRST_LLM_MS:
                if initial_salvaged is not None:
                    normalized, removed_items = initial_salvaged
                    metrics.update(
                        {
                            "topic_pruning_used": True,
                            "topic_pruning_removed_items": removed_items,
                        }
                    )
                    return normalized, metrics
                raise

            retry_tokens = self._compute_retry_tokens(max_tokens)
            retry_prompt = self._build_semantic_retry_prompt(
                prompt=prompt,
                game_type=game_type,
                validation_error=str(exc),
                raw_output=raw_output,
            )
            logger.warning(
                "Generated %s failed validation (%s); retrying with corrective prompt and max_tokens=%d.",
                game_type,
                exc,
                retry_tokens,
            )
            llm_retry_start = time.perf_counter()
            retry_output = await self.llm_client.generate(
                retry_prompt,
                max_tokens=retry_tokens,
            )
            llm_retry_ms = (time.perf_counter() - llm_retry_start) * 1000

            parse_retry_start = time.perf_counter()
            retry_json = extract_json_from_text(retry_output)
            parse_retry_ms = (time.perf_counter() - parse_retry_start) * 1000
            if not retry_json:
                salvaged_retry = self._try_salvage_loose_retry_payload(
                    retry_output,
                    game_type=game_type,
                )
                if salvaged_retry is not None:
                    normalized = self._normalize_generated_payload(
                        data=salvaged_retry,
                        game_type=game_type,
                        language=language,
                        topic=topic,
                        difficulty_percentage=difficulty_percentage,
                    )
                    metrics.update(
                        {
                            "semantic_retry_used": True,
                            "semantic_retry_error": str(exc),
                            "llm_semantic_retry_ms": round(llm_retry_ms, 2),
                            "parse_semantic_retry_ms": round(parse_retry_ms, 2),
                            "llm_total_ms": round(llm_total_ms + llm_retry_ms, 2),
                            "parse_total_ms": round(
                                float(metrics.get("parse_total_ms", 0.0))
                                + parse_retry_ms,
                                2,
                            ),
                            "semantic_retry_fallback_to_initial_payload": False,
                        }
                    )
                    return normalized, metrics
                if initial_salvaged is not None:
                    normalized, removed_items = initial_salvaged
                    metrics.update(
                        {
                            "semantic_retry_used": True,
                            "semantic_retry_error": str(exc),
                            "llm_semantic_retry_ms": round(llm_retry_ms, 2),
                            "parse_semantic_retry_ms": round(parse_retry_ms, 2),
                            "llm_total_ms": round(llm_total_ms + llm_retry_ms, 2),
                            "parse_total_ms": round(
                                float(metrics.get("parse_total_ms", 0.0))
                                + parse_retry_ms,
                                2,
                            ),
                            "topic_pruning_used": True,
                            "topic_pruning_removed_items": removed_items,
                            "semantic_retry_fallback_to_initial_payload": True,
                        }
                    )
                    return normalized, metrics
                raise ValueError(
                    "Failed to extract JSON from semantic retry after validation error: "
                    f"{exc}. Retry output (first 500 chars): {retry_output[:500]}"
                ) from exc

            retry_data = json.loads(retry_json)
            try:
                normalized = self._normalize_generated_payload(
                    data=retry_data,
                    game_type=game_type,
                    language=language,
                    topic=topic,
                    difficulty_percentage=difficulty_percentage,
                )
            except ValueError as retry_exc:
                salvaged = self._try_salvage_topic_filtered_payload(
                    data=retry_data,
                    game_type=game_type,
                    language=language,
                    topic=topic,
                    difficulty_percentage=difficulty_percentage,
                )
                if salvaged is not None:
                    normalized, removed_items = salvaged
                    metrics.update(
                        {
                            "semantic_retry_used": True,
                            "semantic_retry_error": str(exc),
                            "llm_semantic_retry_ms": round(llm_retry_ms, 2),
                            "parse_semantic_retry_ms": round(parse_retry_ms, 2),
                            "llm_total_ms": round(llm_total_ms + llm_retry_ms, 2),
                            "parse_total_ms": round(
                                float(metrics.get("parse_total_ms", 0.0))
                                + parse_retry_ms,
                                2,
                            ),
                            "topic_pruning_used": True,
                            "topic_pruning_removed_items": removed_items,
                        }
                    )
                    return normalized, metrics
                if initial_salvaged is not None:
                    normalized, removed_items = initial_salvaged
                    metrics.update(
                        {
                            "semantic_retry_used": True,
                            "semantic_retry_error": str(exc),
                            "llm_semantic_retry_ms": round(llm_retry_ms, 2),
                            "parse_semantic_retry_ms": round(parse_retry_ms, 2),
                            "llm_total_ms": round(llm_total_ms + llm_retry_ms, 2),
                            "parse_total_ms": round(
                                float(metrics.get("parse_total_ms", 0.0))
                                + parse_retry_ms,
                                2,
                            ),
                            "topic_pruning_used": True,
                            "topic_pruning_removed_items": removed_items,
                            "semantic_retry_fallback_to_initial_payload": True,
                        }
                    )
                    return normalized, metrics
                raise ValueError(
                    f"{exc}; retry also failed validation: {retry_exc}"
                ) from retry_exc

            metrics.update(
                {
                    "semantic_retry_used": True,
                    "semantic_retry_error": str(exc),
                    "llm_semantic_retry_ms": round(llm_retry_ms, 2),
                    "parse_semantic_retry_ms": round(parse_retry_ms, 2),
                    "llm_total_ms": round(llm_total_ms + llm_retry_ms, 2),
                    "parse_total_ms": round(
                        float(metrics.get("parse_total_ms", 0.0)) + parse_retry_ms, 2
                    ),
                }
            )
            return normalized, metrics

    def _with_semantic_retry_metrics(
        self, run_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        metrics = dict(run_metrics)
        metrics.setdefault("semantic_retry_used", False)
        metrics.setdefault("semantic_retry_error", "")
        metrics.setdefault("llm_semantic_retry_ms", 0.0)
        metrics.setdefault("parse_semantic_retry_ms", 0.0)
        metrics.setdefault("semantic_retry_fallback_to_initial_payload", False)
        metrics.setdefault("topic_pruning_used", False)
        metrics.setdefault("topic_pruning_removed_items", 0)
        metrics.setdefault("word_pass_fallback_used", False)
        return metrics

    def _maybe_salvage_topic_filtered_payload(
        self,
        *,
        data: dict[str, Any],
        game_type: str,
        language: str,
        topic: str | None,
        difficulty_percentage: int,
    ) -> tuple[dict[str, Any], int] | None:
        if not isinstance(topic, str) or not topic.strip():
            return None

        structurally_normalized = self._normalize_generated_payload(
            data=data,
            game_type=game_type,
            language=language,
            topic=None,
            difficulty_percentage=difficulty_percentage,
        )
        pruned, removed_items = self._prune_topic_aligned_items(
            normalized=structurally_normalized,
            game_type=game_type,
            topic=topic,
        )
        if removed_items == 0:
            return None

        if game_type == "quiz" and pruned.get("questions"):
            return pruned, removed_items
        if game_type == "true_false" and pruned.get("statements"):
            return pruned, removed_items
        return None

    def _try_salvage_topic_filtered_payload(
        self,
        *,
        data: Any,
        game_type: str,
        language: str,
        topic: str | None,
        difficulty_percentage: int,
    ) -> tuple[dict[str, Any], int] | None:
        try:
            return self._maybe_salvage_topic_filtered_payload(
                data=data,
                game_type=game_type,
                language=language,
                topic=topic,
                difficulty_percentage=difficulty_percentage,
            )
        except ValueError:
            return None

    def _try_salvage_loose_retry_payload(
        self,
        raw_output: str,
        *,
        game_type: str,
    ) -> dict[str, Any] | None:
        if game_type == "word-pass":
            entries = [
                {
                    "letter": match.group("letter"),
                    "hint": match.group("hint"),
                    "answer": match.group("answer"),
                    "starts_with": match.group("starts_with").lower() == "true",
                }
                for match in _WORD_PASS_LOOSE_ENTRY_RE.finditer(raw_output)
            ]
            if not entries:
                return None

            title_match = _WORD_PASS_TITLE_RE.search(raw_output)
            title = title_match.group("title") if title_match else "WordPass"
            return {
                "game_type": "word-pass",
                "title": title,
                "words": entries,
            }

        if game_type != "quiz":
            return None

        questions: list[dict[str, Any]] = []
        for match in _QUIZ_LOOSE_ENTRY_RE.finditer(raw_output):
            options = [option.strip() for option in _QUIZ_LOOSE_OPTION_RE.findall(match.group("options")) if option.strip()]
            if len(options) < 2:
                continue
            questions.append(
                {
                    "question": match.group("question"),
                    "options": options,
                    "correct_index": int(match.group("correct_index")),
                    "explanation": match.group("explanation"),
                }
            )
        if not questions:
            return None

        title_match = _QUIZ_TITLE_RE.search(raw_output)
        title = title_match.group("title") if title_match else "Quiz"
        return {
            "game_type": "quiz",
            "title": title,
            "questions": questions,
        }

    def _build_fallback_word_pass_payload(
        self,
        *,
        topic: str | None,
        language: str,
        difficulty_percentage: int,
        num_questions: int,
    ) -> dict[str, Any]:
        normalized_topic = str(topic or "WordPass").strip() or "WordPass"
        tokens = self._extract_topic_keywords(normalized_topic) or [normalized_topic]
        selected = tokens[: max(1, min(2, num_questions))]
        words: list[dict[str, Any]] = []
        for token in selected:
            answer = " ".join(part.capitalize() for part in token.split()) or normalized_topic
            letter = self._extract_word_pass_letter(answer) or "W"
            if (language or "").lower() == "es":
                hint = f"En {normalized_topic}, termino clave: {answer}."
            else:
                hint = f"In {normalized_topic}, key term: {answer}."
            words.append(
                {
                    "letter": letter,
                    "hint": hint,
                    "answer": answer,
                    "starts_with": True,
                }
            )

        title = normalized_topic[:1].upper() + normalized_topic[1:]
        return {
            "game_type": "word-pass",
            "title": title,
            "difficulty_percentage": max(0, min(100, difficulty_percentage)),
            "words": words,
        }

    def _ensure_word_pass_topic_signal(
        self,
        words: list[dict[str, Any]],
        *,
        topic: str | None,
        language: str,
    ) -> list[dict[str, Any]]:
        topic_keywords = self._extract_topic_keywords(topic)
        if not topic_keywords:
            return words

        topic_roots = self._extract_topic_roots(topic_keywords)
        normalized_topic = str(topic or "").strip()
        if not normalized_topic:
            return words

        enriched: list[dict[str, Any]] = []
        prefix = f"En {normalized_topic}, " if (language or "").lower() == "es" else f"In {normalized_topic}, "
        for item in words:
            bundle = " ".join(
                [
                    str(item.get("letter", "")),
                    str(item.get("hint", "")),
                    str(item.get("answer", "")),
                ]
            )
            if self._contains_topic_keyword(bundle, topic_keywords, topic_roots):
                enriched.append(item)
                continue
            updated = dict(item)
            updated["hint"] = prefix + str(item.get("hint", "")).strip()
            enriched.append(updated)
        return enriched

    def _compute_retry_tokens(self, max_tokens: int) -> int:
        return min(
            max(
                max_tokens + _JSON_RETRY_TOKEN_EXTRA_MIN,
                int(max_tokens * _JSON_RETRY_TOKEN_MULTIPLIER),
            ),
            max_tokens + _JSON_RETRY_TOKEN_EXTRA_MAX,
        )

    def _build_semantic_retry_prompt(
        self,
        *,
        prompt: str,
        game_type: str,
        validation_error: str,
        raw_output: str,
    ) -> str:
        return (
            prompt
            + _SEMANTIC_RETRY_SUFFIX.format(validation_error=validation_error)
            + "\nValidation target game type: "
            + game_type
            + "\nPrevious invalid JSON excerpt:\n"
            + raw_output[:500]
        )

    async def _generate_json_with_retry(
        self,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        """Generate JSON with one corrective retry when extraction fails.

        The second attempt uses a stricter JSON-only instruction and a larger
        token budget, which helps when the first answer is truncated.
        """
        llm_start = time.perf_counter()
        raw_output = await self.llm_client.generate(prompt, max_tokens=max_tokens)
        llm_first_ms = (time.perf_counter() - llm_start) * 1000
        logger.debug("Raw LLM output length: %d chars", len(raw_output))

        parse_start = time.perf_counter()
        json_text = extract_json_from_text(raw_output)
        parse_first_ms = (time.perf_counter() - parse_start) * 1000
        if json_text:
            return (
                json_text,
                raw_output,
                {
                    "retry_used": False,
                    "llm_first_ms": round(llm_first_ms, 2),
                    "llm_retry_ms": 0.0,
                    "llm_total_ms": round(llm_first_ms, 2),
                    "parse_first_ms": round(parse_first_ms, 2),
                    "parse_retry_ms": 0.0,
                    "parse_total_ms": round(parse_first_ms, 2),
                },
            )

        # Avoid triggering a second long-running upstream call when the first
        # attempt already consumed most of the end-to-end timeout budget.
        if llm_first_ms >= _SKIP_JSON_RETRY_IF_FIRST_LLM_MS:
            raise ValueError(
                "Failed to extract JSON from LLM output: first attempt exceeded retry "
                f"latency budget ({round(llm_first_ms, 2)}ms)."
            )

        retry_tokens = self._compute_retry_tokens(max_tokens)
        retry_prompt = prompt + _JSON_RETRY_SUFFIX
        logger.warning(
            "Could not extract JSON from first LLM output; retrying with stricter prompt "
            "and max_tokens=%d.",
            retry_tokens,
        )
        llm_retry_start = time.perf_counter()
        retry_output = await self.llm_client.generate(
            retry_prompt, max_tokens=retry_tokens
        )
        llm_retry_ms = (time.perf_counter() - llm_retry_start) * 1000
        logger.debug("Retry LLM output length: %d chars", len(retry_output))

        parse_retry_start = time.perf_counter()
        retry_json = extract_json_from_text(retry_output)
        parse_retry_ms = (time.perf_counter() - parse_retry_start) * 1000
        if retry_json:
            return (
                retry_json,
                retry_output,
                {
                    "retry_used": True,
                    "llm_first_ms": round(llm_first_ms, 2),
                    "llm_retry_ms": round(llm_retry_ms, 2),
                    "llm_total_ms": round(llm_first_ms + llm_retry_ms, 2),
                    "parse_first_ms": round(parse_first_ms, 2),
                    "parse_retry_ms": round(parse_retry_ms, 2),
                    "parse_total_ms": round(parse_first_ms + parse_retry_ms, 2),
                },
            )

        raise ValueError(
            "Failed to extract JSON from LLM output after retry.\n"
            f"First output (first 500 chars): {raw_output[:500]}\n"
            f"Retry output (first 500 chars): {retry_output[:500]}"
        )
