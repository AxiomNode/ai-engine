"""Prompt templates for educational game generation.

Each template instructs the LLM to produce strict JSON matching the
schemas defined in :mod:`ai_engine.games.schemas`.  Templates use Python
:meth:`str.format` placeholders: ``{context}``,
``{num_questions}``, ``{language}``, and ``{letters}``.
"""

from __future__ import annotations

from ai_engine.games.catalog import get_supported_game_types, get_game_type_profile

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
    "explanations, hints, answers) in the language specified in the "
    "requirements. Do NOT mix languages. If the requested language is "
    "'es', write everything in Spanish. If 'en', write everything in English."
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
    "- Language for all content: {language}\n"
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
    "Write ALL text in {language}. "
    "Generate the quiz now:"
)

# ------------------------------------------------------------------
# WordPass
# ------------------------------------------------------------------

WORD_PASS_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a WordPass (rosco) game.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "{topic_clause}"
    "- Language for all content: {language}\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Cover these letters: {letters}\n"
    "- Title must be a short, non-empty string relevant to the topic.\n"
    "- Every hint and answer must stay directly related to the requested topic.\n"
    "- Do not include unrelated general-knowledge filler, even if it appears in the context.\n"
    "- For each letter provide a hint (definition/clue) and the answer word.\n"
    '- Set "starts_with" to true if the answer starts with the letter, false if it only contains it.\n\n'
    "### Output JSON schema\n"
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
    "IMPORTANT: Generate one word for EACH letter in the list. "
    "Write ALL text in {language}. "
    "Generate the word-pass now:"
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
    "- Language for all content: {language}\n"
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
    language: str = "es",
    difficulty_percentage: int = 50,
    num_questions: int = 5,
    letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
) -> str:
    """Build a complete prompt for the given game type.

    Args:
        game_type: One of ``"quiz"``, ``"word-pass"``, ``"true_false"``.
        context: RAG-retrieved context text.
        language: Target language code (default ``"es"`` = Spanish).
        num_questions: Number of questions/statements to generate.
        letters: Comma-separated letters for word-pass.

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
        system=_SYSTEM,
        context=context,
        topic_clause=(
            f"- Focus specifically on this topic/request: {topic.strip()}\\n"
            if isinstance(topic, str) and topic.strip()
            else ""
        ),
        language=language,
        difficulty_percentage=difficulty_percentage,
        num_questions=num_questions,
        letters=letters,
    )
