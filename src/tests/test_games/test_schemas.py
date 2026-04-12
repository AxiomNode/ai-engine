"""Tests for ai_engine.games.schemas – educational game data models."""

import pytest
from pydantic import ValidationError

from ai_engine.games.schemas import (
    GameEnvelope,
    QuizGame,
    QuizQuestion,
    TrueFalseGame,
    TrueFalseStatement,
    WordPassGame,
    WordPassWord,
)

# ------------------------------------------------------------------
# QuizQuestion
# ------------------------------------------------------------------


class TestQuizQuestion:

    def test_valid_creation(self):
        q = QuizQuestion(
            question="What is 2+2?",
            options=["3", "4", "5", "6"],
            correct_index=1,
        )
        assert q.question == "What is 2+2?"
        assert q.correct_index == 1

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError, match="question must not be empty"):
            QuizQuestion(question="", options=["a", "b"], correct_index=0)

    def test_too_few_options_raises(self):
        with pytest.raises(ValidationError, match="at least 2 choices"):
            QuizQuestion(question="Q?", options=["only one"], correct_index=0)

    def test_correct_index_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="out of range"):
            QuizQuestion(question="Q?", options=["a", "b"], correct_index=5)

    def test_round_trip(self):
        q = QuizQuestion(
            question="Q?",
            options=["a", "b", "c", "d"],
            correct_index=2,
            explanation="Because c.",
        )
        d = q.to_dict()
        q2 = QuizQuestion.from_dict(d)
        assert q2.question == q.question
        assert q2.options == q.options
        assert q2.correct_index == q.correct_index
        assert q2.explanation == q.explanation


# ------------------------------------------------------------------
# QuizGame
# ------------------------------------------------------------------


class TestQuizGame:

    def test_valid_creation(self):
        game = QuizGame(title="My Quiz")
        assert game.title == "My Quiz"
        assert game.questions == []

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="title must not be empty"):
            QuizGame(title="")

    def test_to_dict_includes_game_type(self):
        game = QuizGame(
            title="T",
            questions=[
                QuizQuestion(question="Q?", options=["a", "b"], correct_index=0),
            ],
        )
        d = game.to_dict()
        assert d["game_type"] == "quiz"
        assert len(d["questions"]) == 1

    def test_from_dict_round_trip(self):
        game = QuizGame(
            title="T",
            questions=[
                QuizQuestion(
                    question="Q?",
                    options=["a", "b", "c", "d"],
                    correct_index=1,
                    explanation="explanation",
                ),
            ],
        )
        d = game.to_dict()
        game2 = QuizGame.from_dict(d)
        assert game2.title == game.title
        assert len(game2.questions) == 1
        assert game2.questions[0].correct_index == 1


# ------------------------------------------------------------------
# WordPassWord
# ------------------------------------------------------------------


class TestWordPassWord:

    def test_valid_creation(self):
        w = WordPassWord(letter="A", hint="First letter", answer="Alpha")
        assert w.letter == "A"
        assert w.starts_with is True

    def test_letter_uppercased(self):
        w = WordPassWord(letter="b", hint="H", answer="Beta")
        assert w.letter == "B"

    def test_invalid_letter_raises(self):
        with pytest.raises(ValidationError, match="single alphabetic character"):
            WordPassWord(letter="AB", hint="H", answer="A")

    def test_empty_hint_raises(self):
        with pytest.raises(ValidationError, match="hint must not be empty"):
            WordPassWord(letter="A", hint="", answer="A")

    def test_empty_answer_raises(self):
        with pytest.raises(ValidationError, match="answer must not be empty"):
            WordPassWord(letter="A", hint="H", answer="")

    def test_round_trip(self):
        w = WordPassWord(letter="z", hint="Last", answer="Zebra", starts_with=True)
        d = w.to_dict()
        w2 = WordPassWord.from_dict(d)
        assert w2.letter == "Z"
        assert w2.answer == "Zebra"


# ------------------------------------------------------------------
# WordPassGame
# ------------------------------------------------------------------


class TestWordPassGame:

    def test_valid_creation(self):
        game = WordPassGame(title="Rosco")
        assert game.words == []

    def test_to_dict_game_type(self):
        game = WordPassGame(
            title="R",
            words=[
                WordPassWord(
                    letter="A", hint="Capital of France \u2013 not!", answer="Amsterdam"
                ),
            ],
        )
        d = game.to_dict()
        assert d["game_type"] == "word-pass"
        assert len(d["words"]) == 1


# ------------------------------------------------------------------
# TrueFalseStatement
# ------------------------------------------------------------------


class TestTrueFalseStatement:

    def test_valid_creation(self):
        s = TrueFalseStatement(statement="Water is wet", is_true=True)
        assert s.is_true is True

    def test_empty_statement_raises(self):
        with pytest.raises(ValidationError, match="statement must not be empty"):
            TrueFalseStatement(statement="", is_true=True)

    def test_round_trip(self):
        s = TrueFalseStatement(statement="S", is_true=False, explanation="E")
        d = s.to_dict()
        s2 = TrueFalseStatement.from_dict(d)
        assert s2.is_true is False


# ------------------------------------------------------------------
# TrueFalseGame
# ------------------------------------------------------------------


class TestTrueFalseGame:

    def test_valid_creation(self):
        game = TrueFalseGame(title="T/F")
        assert game.statements == []

    def test_to_dict_game_type(self):
        d = TrueFalseGame(
            title="T",
            statements=[
                TrueFalseStatement(statement="The sun is a star", is_true=True),
            ],
        ).to_dict()
        assert d["game_type"] == "true_false"


# ------------------------------------------------------------------
# GameEnvelope
# ------------------------------------------------------------------


class TestGameEnvelope:

    def test_from_dict_quiz(self):
        data = {
            "game_type": "quiz",
            "title": "Q",
            "questions": [
                {"question": "Q?", "options": ["a", "b"], "correct_index": 0},
            ],
        }
        env = GameEnvelope.from_dict(data)
        assert env.game_type == "quiz"
        assert isinstance(env.game, QuizGame)

    def test_from_dict_word_pass(self):
        data = {
            "game_type": "word-pass",
            "title": "P",
            "words": [
                {"letter": "A", "hint": "H", "answer": "Ans"},
            ],
        }
        env = GameEnvelope.from_dict(data)
        assert env.game_type == "word-pass"
        assert isinstance(env.game, WordPassGame)

    def test_from_dict_true_false(self):
        data = {
            "game_type": "true_false",
            "title": "TF",
            "statements": [
                {"statement": "S", "is_true": True},
            ],
        }
        env = GameEnvelope.from_dict(data)
        assert env.game_type == "true_false"
        assert isinstance(env.game, TrueFalseGame)

    def test_unknown_game_type_raises(self):
        with pytest.raises(ValueError, match="Unknown game_type"):
            GameEnvelope.from_dict({"game_type": "bogus"})

    def test_game_type_alias_is_normalized(self):
        data = {
            "game_type": "educational-game",
            "title": "Q",
            "questions": [
                {"question": "Q?", "options": ["a", "b", "c", "d"], "correct_index": 1},
            ],
        }
        env = GameEnvelope.from_dict(data)
        assert env.game_type == "quiz"
        assert isinstance(env.game, QuizGame)

    def test_to_dict_round_trip(self):
        data = {
            "game_type": "quiz",
            "title": "Q",
            "questions": [
                {"question": "Q?", "options": ["a", "b", "c", "d"], "correct_index": 2},
            ],
        }
        env = GameEnvelope.from_dict(data)
        d = env.to_dict()
        assert d["game_type"] == "quiz"
        assert d["questions"][0]["correct_index"] == 2
