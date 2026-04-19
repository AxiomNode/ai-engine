"""Tests for central game-type tuning profiles."""

from ai_engine.games.catalog import (
    count_requested_items,
    estimate_effective_max_tokens,
    get_game_type_profile,
    normalize_game_type,
)


def test_normalize_game_type_aliases() -> None:
    assert normalize_game_type("word_pass") == "word-pass"
    assert normalize_game_type("true-false") == "true_false"


def test_count_requested_items_prefers_item_count_for_word_pass() -> None:
    assert count_requested_items("word-pass", item_count=4) == 4
    assert count_requested_items("word-pass", num_questions=3) == 3


def test_estimate_effective_max_tokens_uses_game_profile() -> None:
    assert estimate_effective_max_tokens("quiz", 512, num_questions=5) == 512
    assert estimate_effective_max_tokens("word-pass", 1024, item_count=5) == 256
    assert estimate_effective_max_tokens("word-pass", 1024, item_count=20) == 416


def test_get_game_type_profile_returns_retrieval_settings() -> None:
    profile = get_game_type_profile("quiz")
    assert profile.retrieval_top_k >= 5
    assert profile.context_char_limit >= 3500