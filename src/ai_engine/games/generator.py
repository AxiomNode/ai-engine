"""High-level generator that combines RAG context with an LLM to produce
structured educational game definitions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ai_engine.games.prompts import get_prompt
from ai_engine.games.schemas import GameEnvelope
from ai_engine.llm.llama_client import LlamaClient
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.utils import extract_json_from_text

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        topic: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
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
            topic: Educational topic for the game.
            game_type: ``"quiz"``, ``"pasapalabra"``, or ``"true_false"``.
            language: Output language (defaults to ``self.default_language``).
            num_questions: Number of questions/statements.
            letters: Comma-separated letters (used only by pasapalabra).
            max_tokens: Override default token budget.
            top_k: Override RAG retrieval count.

        Returns:
            A :class:`GameEnvelope` wrapping the concrete game.

        Raises:
            ValueError: If JSON extraction or parsing fails.
        """
        lang = language or self.default_language
        tokens = max_tokens or self.default_max_tokens

        # 1. Retrieve context
        context = self.rag_pipeline.build_context(query, top_k=top_k)
        logger.debug("RAG context length: %d chars", len(context))

        # 2. Build prompt
        prompt = get_prompt(
            game_type=game_type,
            context=context,
            topic=topic,
            language=lang,
            num_questions=num_questions,
            letters=letters,
        )
        logger.debug("Prompt length: %d chars", len(prompt))

        # 3. Call LLM
        raw_output = self.llm_client.generate(prompt, max_tokens=tokens)
        logger.debug("Raw LLM output length: %d chars", len(raw_output))

        # 4. Extract JSON
        json_text = extract_json_from_text(raw_output)
        if not json_text:
            raise ValueError(
                f"Failed to extract JSON from LLM output.\n"
                f"Raw output (first 500 chars): {raw_output[:500]}"
            )

        data = json.loads(json_text)

        # Ensure game_type is present in the parsed data
        if "game_type" not in data:
            data["game_type"] = game_type

        return GameEnvelope.from_dict(data)

    def generate_raw(
        self,
        query: str,
        topic: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
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
            topic=topic,
            language=lang,
            num_questions=num_questions,
            letters=letters,
        )
        raw_output = self.llm_client.generate(prompt, max_tokens=tokens)
        json_text = extract_json_from_text(raw_output)
        if not json_text:
            raise ValueError(
                f"Failed to extract JSON from LLM output.\n"
                f"Raw output (first 500 chars): {raw_output[:500]}"
            )
        return json.loads(json_text)
