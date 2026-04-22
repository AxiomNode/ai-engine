"""Tests for central game-type tuning profiles."""

import pytest

from ai_engine.games.catalog import (
    count_requested_items,
    estimate_effective_max_tokens,
    get_game_type_profile,
    get_supported_game_types,
    normalize_game_type,
)


def test_normalize_game_type_aliases() -> None:
    assert normalize_game_type("word_pass") == "word-pass"
    assert normalize_game_type("true-false") == "true_false"
    assert normalize_game_type(" Educational-Game ") == "quiz"


def test_get_game_type_profile_rejects_unknown_game_type() -> None:
    with pytest.raises(ValueError, match="Unknown game_type"):
        get_game_type_profile("arcade")


def test_count_requested_items_prefers_item_count_for_word_pass() -> None:
    assert count_requested_items("word-pass", item_count=4) == 4
    assert count_requested_items("word-pass", num_questions=3) == 3


def test_count_requested_items_uses_letters_when_present() -> None:
    assert count_requested_items("word-pass", letters="A, B, , C ") == 3
    assert count_requested_items("word-pass", letters=" , ", num_questions=2) == 2


def test_estimate_effective_max_tokens_uses_game_profile() -> None:
    assert estimate_effective_max_tokens("quiz", 512, num_questions=5) == 512
    assert estimate_effective_max_tokens("word-pass", 1024, item_count=5) == 256
    assert estimate_effective_max_tokens("word-pass", 1024, item_count=20) == 416


def test_estimate_effective_max_tokens_clamps_low_requests_and_letters_budget() -> None:
    assert estimate_effective_max_tokens("quiz", 32, num_questions=1) == 64
    assert estimate_effective_max_tokens("word-pass", 512, letters="A,B,C,D,E,F") == 256


def test_get_game_type_profile_returns_retrieval_settings() -> None:
    profile = get_game_type_profile("quiz")
    assert profile.retrieval_top_k >= 5
    assert profile.context_char_limit >= 3500


def test_get_supported_game_types_returns_canonical_values() -> None:
    assert get_supported_game_types() == ["quiz", "word-pass", "true_false"]
