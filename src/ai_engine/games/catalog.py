"""Central game-type profiles for generation and retrieval tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameTypeProfile:
    """Operational profile for one supported educational game type."""

    game_type: str
    retrieval_top_k: int
    context_char_limit: int
    min_tokens: int
    base_tokens: int
    per_item_tokens: int


GAME_TYPE_ALIASES: dict[str, str] = {
    "educational-game": "quiz",
    "educational_game": "quiz",
    "wordpass": "word-pass",
    "word_pass": "word-pass",
    "true-false": "true_false",
    "truefalse": "true_false",
}


GAME_TYPE_PROFILES: dict[str, GameTypeProfile] = {
    "quiz": GameTypeProfile(
        game_type="quiz",
        retrieval_top_k=6,
        context_char_limit=4200,
        min_tokens=320,
        base_tokens=160,
        per_item_tokens=72,
    ),
    "word-pass": GameTypeProfile(
        game_type="word-pass",
        retrieval_top_k=6,
        context_char_limit=2600,
        min_tokens=256,
        base_tokens=96,
        per_item_tokens=16,
    ),
    "true_false": GameTypeProfile(
        game_type="true_false",
        retrieval_top_k=8,
        context_char_limit=3800,
        min_tokens=256,
        base_tokens=112,
        per_item_tokens=48,
    ),
}


def normalize_game_type(game_type: str) -> str:
    """Normalize known aliases to their canonical game type."""
    normalized = str(game_type or "quiz").strip().lower()
    return GAME_TYPE_ALIASES.get(normalized, normalized)


def get_game_type_profile(game_type: str) -> GameTypeProfile:
    """Return the operational profile for a canonical or aliased game type."""
    normalized = normalize_game_type(game_type)
    profile = GAME_TYPE_PROFILES.get(normalized)
    if profile is None:
        raise ValueError(
            f"Unknown game_type {game_type!r}. Supported: {list(GAME_TYPE_PROFILES)}"
        )
    return profile


def count_requested_items(
    game_type: str,
    *,
    item_count: int | None = None,
    num_questions: int = 5,
    letters: str | None = None,
) -> int:
    """Estimate the number of generated items requested by the caller."""
    if item_count is not None:
        return max(1, int(item_count))
    normalized = normalize_game_type(game_type)
    if normalized == "word-pass" and isinstance(letters, str) and letters.strip():
        parsed_letters = [entry.strip() for entry in letters.split(",") if entry.strip()]
        if parsed_letters:
            return max(1, len(parsed_letters))
    return max(1, int(num_questions))


def estimate_effective_max_tokens(
    game_type: str,
    requested_max_tokens: int,
    *,
    item_count: int | None = None,
    num_questions: int = 5,
    letters: str | None = None,
) -> int:
    """Clamp token budgets using per-game operational profiles."""
    profile = get_game_type_profile(game_type)
    requested = max(64, int(requested_max_tokens))
    requested_items = count_requested_items(
        game_type,
        item_count=item_count,
        num_questions=num_questions,
        letters=letters,
    )
    budget = max(profile.min_tokens, profile.base_tokens + (profile.per_item_tokens * requested_items))
    return min(requested, budget)


def get_supported_game_types() -> list[str]:
    """Return canonical game types in registration order."""
    return list(GAME_TYPE_PROFILES)