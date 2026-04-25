"""Prompt templates for educational game generation.

Each template instructs the LLM to produce strict JSON matching the
schemas defined in :mod:`ai_engine.games.schemas`.  Templates use Python
:meth:`str.format` placeholders: ``{context}`` and ``{num_questions}``.

Prompt versioning
-----------------

Prompts are versioned per ``game_type`` via :data:`PROMPT_VERSIONS`.
Any meaningful change to a template (wording, schema, instruction policy)
MUST bump the corresponding entry. The version is consumed by the LLMOps
eval harness (see ``tests/eval/``) and is exported as part of
:func:`get_prompt_version` so callers can log it for traceability.
"""

from __future__ import annotations

from ai_engine.games.catalog import get_game_type_profile, get_supported_game_types

# ------------------------------------------------------------------
# Prompt versioning (LLMOps - ADR 0009)
# ------------------------------------------------------------------

#: Active version of each game-type prompt template. Bump on any change.
PROMPT_VERSIONS: dict[str, str] = {
    "quiz": "v1",
    "word-pass": "v1",
    "true_false": "v1",
}


def get_prompt_version(game_type: str) -> str:
    """Return the active prompt template version for *game_type*."""
    profile = get_game_type_profile(game_type)
    version = PROMPT_VERSIONS.get(profile.game_type)
    if version is None:
        raise ValueError(
            f"No prompt version registered for {game_type!r}. "
            f"Supported: {get_supported_game_types()}"
        )
    return version


# ------------------------------------------------------------------
# System-level instruction shared by every game type
# ------------------------------------------------------------------

_SYSTEM = (
    "You are an expert pedagogue specialising in educational game design. "
    "You MUST reply ONLY with valid, strict JSON — no markdown fences, no "
    "commentary, no explanation before or after the JSON block. "
    "CRITICAL: Base ALL questions, answers, and explanations EXCLUSIVELY on "
    "the provided context. Do NOT fabricate facts, names, dates, or data "
    "that are not present in the context. If the context is insufficient "
    "to generate the requested number of items, generate fewer items "
    "rather than inventing content. "
    "LANGUAGE RULE: You MUST write ALL text (title, questions, options, "
    "explanations, hints, answers) in English only. Do NOT mix languages."
)

_WORD_PASS_SYSTEM = (
    "Return ONLY valid JSON. Use ONLY the provided context. "
    "If the context is insufficient, return fewer entries instead of inventing facts. "
    "Keep every hint and answer directly tied to the requested topic. "
    "Write all title, hint, and answer text in English only."
)

# ------------------------------------------------------------------
# Quiz
# ------------------------------------------------------------------

QUIZ_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a multiple-choice quiz.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "{topic_clause}"
    "- Language for all content: English\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Number of questions: {num_questions}\n"
    "- Title must be a short, non-empty string relevant to the topic.\n"
    "- Every question, option set, and explanation must stay directly related to the requested topic.\n"
    "- Do not include unrelated general-knowledge filler, even if it appears in the context.\n"
    "- Each question has exactly 4 options, only one correct.\n"
    "- Include a short pedagogical explanation for each answer.\n\n"
    "### Output JSON schema\n"
    "{{\n"
    '  "game_type": "quiz",\n'
    '  "title": "<string>",\n'
    '  "questions": [\n'
    "    {{\n"
    '      "question": "<string>",\n'
    '      "options": ["<A>", "<B>", "<C>", "<D>"],\n'
    '      "correct_index": <0-3>,\n'
    '      "explanation": "<string>"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "IMPORTANT: Generate exactly {num_questions} questions. "
    "Write ALL text in English. "
    "Generate the quiz now:"
)

# ------------------------------------------------------------------
# WordPass
# ------------------------------------------------------------------

WORD_PASS_TEMPLATE = (
    "{system}\n\n"
    "Create standalone WordPass entries from this context.\n\n"
    "Context:\n{context}\n\n"
    "Requirements:\n"
    "{topic_clause}"
    "- Language: English\n"
    "- Difficulty: {difficulty_percentage}%\n"
    "- Return up to {num_questions} standalone entries.\n"
    "- Use short, topic-specific hints.\n"
    "- Do not build an alphabet rosco; entries must be independently reusable.\n"
    '- Set "starts_with" to true only when the answer begins with the letter.\n\n'
    "JSON schema:\n"
    "{{\n"
    '  "game_type": "word-pass",\n'
    '  "title": "<string>",\n'
    '  "words": [\n'
    "    {{\n"
    '      "letter": "<A-Z>",\n'
    '      "hint": "<string>",\n'
    '      "answer": "<string>",\n'
    '      "starts_with": <true|false>\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Return the word-pass JSON now:"
)

# ------------------------------------------------------------------
# True / False
# ------------------------------------------------------------------

TRUE_FALSE_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a true/false game.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "{topic_clause}"
    "- Language for all content: English\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Number of statements: {num_questions}\n"
    "- Title must be a short, non-empty string relevant to the topic.\n"
    "- Every statement and explanation must stay directly related to the requested topic.\n"
    "- Do not include unrelated general-knowledge filler, even if it appears in the context.\n"
    "- Mix of true and false statements (roughly balanced).\n"
    "- Include a short pedagogical explanation for each.\n\n"
    "### Output JSON schema\n"
    "{{\n"
    '  "game_type": "true_false",\n'
    '  "title": "<string>",\n'
    '  "statements": [\n'
    "    {{\n"
    '      "statement": "<string>",\n'
    '      "is_true": <true|false>,\n'
    '      "explanation": "<string>"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Generate the game now:"
)

# ------------------------------------------------------------------
# Helper to get template by game type
# ------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "quiz": QUIZ_TEMPLATE,
    "word-pass": WORD_PASS_TEMPLATE,
    "true_false": TRUE_FALSE_TEMPLATE,
}


def get_prompt(
    game_type: str,
    context: str,
    topic: str | None = None,
    language: str = "en",
    difficulty_percentage: int = 50,
    num_questions: int = 5,
    letters: str | None = None,
) -> str:
    """Build a complete prompt for the given game type.

    Args:
        game_type: One of ``"quiz"``, ``"word-pass"``, ``"true_false"``.
        context: RAG-retrieved context text.
        language: Deprecated legacy argument. Prompts always generate English.
        num_questions: Number of questions/statements to generate.
        letters: Deprecated legacy rosco letters; ignored by new prompts.

    Returns:
        Ready-to-send prompt string.

    Raises:
        ValueError: If *game_type* is not recognised.
    """
    profile = get_game_type_profile(game_type)
    template = TEMPLATES.get(profile.game_type)
    if template is None:
        raise ValueError(
            f"Unknown game_type {game_type!r}. Supported: {get_supported_game_types()}"
        )
    return template.format(
        system=_WORD_PASS_SYSTEM if profile.game_type == "word-pass" else _SYSTEM,
        context=context,
        topic_clause=(
            f"- Topic: {topic.strip()}\\n"
            if isinstance(topic, str) and topic.strip()
            else ""
        ),
        difficulty_percentage=difficulty_percentage,
        num_questions=num_questions,
        letters=letters or "",
    )
