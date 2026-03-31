"""High-level generator that combines RAG context with an LLM to produce
structured educational game definitions.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

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

# Keep retry bounded to avoid very long second attempts under constrained CPU inference.
_JSON_RETRY_TOKEN_EXTRA_MIN = 128
_JSON_RETRY_TOKEN_EXTRA_MAX = 256
_JSON_RETRY_TOKEN_MULTIPLIER = 1.25
_SKIP_JSON_RETRY_IF_FIRST_LLM_MS = 90_000.0

_INLINE_OPTIONS_RE = re.compile(
    r"\s+[A-D][\)\.:\-]\s*[^\n]+(?:\s+[A-D][\)\.:\-]\s*[^\n]+)+",
    re.IGNORECASE,
)


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
        context = self.rag_pipeline.build_context(query, top_k=top_k)
        logger.debug("RAG context length: %d chars", len(context))

        return await self.generate_from_context(
            context=context,
            game_type=game_type,
            language=language,
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
            language=lang,
            difficulty_percentage=difficulty_percentage,
            num_questions=num_questions,
            letters=letters,
        )
        logger.debug("Prompt length: %d chars", len(prompt))

        json_text, _, run_metrics = await self._generate_json_with_retry(prompt, tokens)
        self.last_run_metrics = run_metrics

        data = json.loads(json_text)
        data = self._normalize_generated_payload(
            data=data,
            game_type=game_type,
            difficulty_percentage=difficulty_percentage,
        )
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

        context = self.rag_pipeline.build_context(query, top_k=top_k)
        prompt = get_prompt(
            game_type=game_type,
            context=context,
            language=lang,
            difficulty_percentage=difficulty_percentage,
            num_questions=num_questions,
            letters=letters,
        )
        json_text, _, run_metrics = await self._generate_json_with_retry(prompt, tokens)
        self.last_run_metrics = run_metrics
        data = json.loads(json_text)
        return self._normalize_generated_payload(
            data=data,
            game_type=game_type,
            difficulty_percentage=difficulty_percentage,
        )

    def _normalize_generated_payload(
        self,
        *,
        data: dict[str, Any],
        game_type: str,
        difficulty_percentage: int,
    ) -> dict[str, Any]:
        """Normalize generated payload fields for downstream contracts."""
        normalized = dict(data)
        normalized_game_type = str(normalized.get("game_type") or game_type or "quiz").strip().lower()
        if normalized_game_type in {"educational-game", "educational_game"}:
            normalized_game_type = "quiz"
        elif normalized_game_type in {"wordpass", "word_pass"}:
            normalized_game_type = "word-pass"
        elif normalized_game_type in {"true-false", "truefalse"}:
            normalized_game_type = "true_false"
        normalized["game_type"] = normalized_game_type
        normalized.setdefault("title", "Quiz")
        normalized["difficulty_percentage"] = max(0, min(100, difficulty_percentage))

        if normalized_game_type == "quiz":
            raw_questions = normalized.get("questions", [])
            if isinstance(raw_questions, list):
                cleaned_questions: list[dict[str, Any]] = []
                for item in raw_questions:
                    if not isinstance(item, dict):
                        continue
                    question = str(item.get("question", "")).strip()
                    item = dict(item)
                    item["question"] = self._clean_quiz_question_text(question)

                    raw_options = item.get("options", [])
                    options: list[str] = []
                    if isinstance(raw_options, list):
                        options = [str(option).strip() for option in raw_options if str(option).strip()]

                    # Ensure minimum choices required by QuizQuestion schema.
                    if len(options) < 2:
                        fallback_a = "Option A"
                        fallback_b = "Option B"
                        if options:
                            fallback_a = options[0]
                        options = [fallback_a, fallback_b]
                    item["options"] = options

                    raw_correct_index = item.get("correct_index", 0)
                    try:
                        correct_index = int(raw_correct_index)
                    except (TypeError, ValueError):
                        correct_index = 0
                    if correct_index < 0 or correct_index >= len(options):
                        correct_index = 0
                    item["correct_index"] = correct_index

                    item["explanation"] = str(item.get("explanation", "") or "")

                    if not item["question"]:
                        item["question"] = "Multiple choice question"
                    cleaned_questions.append(item)
                normalized["questions"] = cleaned_questions

            normalized["title"] = str(normalized.get("title") or "Quiz").strip() or "Quiz"
        return normalized

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

        retry_tokens = min(
            max(max_tokens + _JSON_RETRY_TOKEN_EXTRA_MIN, int(max_tokens * _JSON_RETRY_TOKEN_MULTIPLIER)),
            max_tokens + _JSON_RETRY_TOKEN_EXTRA_MAX,
        )
        retry_prompt = prompt + _JSON_RETRY_SUFFIX
        logger.warning(
            "Could not extract JSON from first LLM output; retrying with stricter prompt "
            "and max_tokens=%d.",
            retry_tokens,
        )
        llm_retry_start = time.perf_counter()
        retry_output = await self.llm_client.generate(retry_prompt, max_tokens=retry_tokens)
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
