"""Tests for ai_engine.games.prompts – prompt template builder."""

import pytest

from ai_engine.games.prompts import TEMPLATES, get_prompt


class TestGetPrompt:

    def test_quiz_prompt_contains_required_fields(self):
        prompt = get_prompt(
            game_type="quiz",
            context="Python is a programming language.",
        )
        assert "quiz" in prompt.lower()
        assert "Python is a programming language" in prompt
        assert "correct_index" in prompt

    def test_word_pass_prompt_contains_letters(self):
        prompt = get_prompt(
            game_type="word-pass",
            context="Geography facts.",
            letters="A,B,C",
        )
        assert "A,B,C" in prompt
        assert "word-pass" in prompt.lower()

    def test_true_false_prompt(self):
        prompt = get_prompt(
            game_type="true_false",
            context="Science facts.",
            num_questions=10,
        )
        assert "true_false" in prompt or "true/false" in prompt.lower()
        assert "10" in prompt

    def test_unknown_game_type_raises(self):
        with pytest.raises(ValueError, match="Unknown game_type"):
            get_prompt(game_type="bogus", context="")

    def test_default_language_is_spanish(self):
        prompt = get_prompt(game_type="quiz", context="C")
        assert "es" in prompt

    def test_all_templates_registered(self):
        assert "quiz" in TEMPLATES
        assert "word-pass" in TEMPLATES
        assert "true_false" in TEMPLATES
