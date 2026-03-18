"""Prompt templates for educational game generation.

Each template instructs the LLM to produce strict JSON matching the
schemas defined in :mod:`ai_engine.games.schemas`.  Templates use Python
:meth:`str.format` placeholders: ``{context}``, ``{topic}``,
``{num_questions}``, ``{language}``, and ``{letters}``.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# System-level instruction shared by every game type
# ------------------------------------------------------------------

_SYSTEM = (
    "You are an expert pedagogue specialising in educational game design. "
    "You MUST reply ONLY with valid, strict JSON — no markdown fences, no "
    "commentary, no explanation before or after the JSON block."
)

# ------------------------------------------------------------------
# Quiz
# ------------------------------------------------------------------

QUIZ_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a multiple-choice quiz.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "- Topic: {topic}\n"
    "- Language for all content: {language}\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Number of questions: {num_questions}\n"
    "- Each question has exactly 4 options, only one correct.\n"
    "- Include a short pedagogical explanation for each answer.\n\n"
    "### Output JSON schema\n"
    "{{\n"
    '  "game_type": "quiz",\n'
    '  "title": "<string>",\n'
    '  "topic": "<string>",\n'
    '  "questions": [\n'
    "    {{\n"
    '      "question": "<string>",\n'
    '      "options": ["<A>", "<B>", "<C>", "<D>"],\n'
    '      "correct_index": <0-3>,\n'
    '      "explanation": "<string>"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Generate the quiz now:"
)

# ------------------------------------------------------------------
# Pasapalabra
# ------------------------------------------------------------------

PASAPALABRA_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a Pasapalabra (rosco) game.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "- Topic: {topic}\n"
    "- Language for all content: {language}\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Cover these letters: {letters}\n"
    "- For each letter provide a hint (definition/clue) and the answer word.\n"
    '- Set "starts_with" to true if the answer starts with the letter, false if it only contains it.\n\n'
    "### Output JSON schema\n"
    "{{\n"
    '  "game_type": "pasapalabra",\n'
    '  "title": "<string>",\n'
    '  "topic": "<string>",\n'
    '  "words": [\n'
    "    {{\n"
    '      "letter": "<A-Z>",\n'
    '      "hint": "<string>",\n'
    '      "answer": "<string>",\n'
    '      "starts_with": <true|false>\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Generate the pasapalabra now:"
)

# ------------------------------------------------------------------
# True / False
# ------------------------------------------------------------------

TRUE_FALSE_TEMPLATE = (
    "{system}\n\n"
    "Using the following educational context, create a true/false game.\n\n"
    "### Context\n{context}\n\n"
    "### Requirements\n"
    "- Topic: {topic}\n"
    "- Language for all content: {language}\n"
    "- Difficulty percentage: {difficulty_percentage}% (0 easy, 100 hard)\n"
    "- Number of statements: {num_questions}\n"
    "- Mix of true and false statements (roughly balanced).\n"
    "- Include a short pedagogical explanation for each.\n\n"
    "### Output JSON schema\n"
    "{{\n"
    '  "game_type": "true_false",\n'
    '  "title": "<string>",\n'
    '  "topic": "<string>",\n'
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
    "pasapalabra": PASAPALABRA_TEMPLATE,
    "true_false": TRUE_FALSE_TEMPLATE,
}


def get_prompt(
    game_type: str,
    context: str,
    topic: str,
    language: str = "es",
    difficulty_percentage: int = 50,
    num_questions: int = 5,
    letters: str = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
) -> str:
    """Build a complete prompt for the given game type.

    Args:
        game_type: One of ``"quiz"``, ``"pasapalabra"``, ``"true_false"``.
        context: RAG-retrieved context text.
        topic: Educational topic.
        language: Target language code (default ``"es"`` = Spanish).
        num_questions: Number of questions/statements to generate.
        letters: Comma-separated letters for pasapalabra.

    Returns:
        Ready-to-send prompt string.

    Raises:
        ValueError: If *game_type* is not recognised.
    """
    template = TEMPLATES.get(game_type)
    if template is None:
        raise ValueError(
            f"Unknown game_type {game_type!r}. Supported: {list(TEMPLATES)}"
        )
    return template.format(
        system=_SYSTEM,
        context=context,
        topic=topic,
        language=language,
        difficulty_percentage=difficulty_percentage,
        num_questions=num_questions,
        letters=letters,
    )
