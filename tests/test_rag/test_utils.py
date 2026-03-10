"""Tests for ai_engine.rag.utils – JSON extraction helpers."""


from ai_engine.rag.utils import extract_json_from_text


class TestExtractJsonFromText:
    """Tests for extract_json_from_text."""

    def test_returns_json_from_plain_json_string(self):
        raw = '{"title": "Quiz", "questions": []}'
        result = extract_json_from_text(raw)
        assert result == '{"title": "Quiz", "questions": []}'

    def test_extracts_json_after_chain_of_thought(self):
        raw = (
            "Thinking step 1...\nThinking step 2...\n\n"
            '{"title": "TestGame", "questions": []}'
        )
        result = extract_json_from_text(raw)
        assert result is not None
        assert '"title"' in result
        assert '"TestGame"' in result

    def test_extracts_nested_json(self):
        raw = 'Some preamble {"a": {"b": [1, 2, 3]}} trailing text'
        result = extract_json_from_text(raw)
        assert result is not None
        assert '"a"' in result
        assert '"b"' in result

    def test_returns_none_when_no_json(self):
        assert extract_json_from_text("No JSON here at all") is None

    def test_returns_none_for_empty_string(self):
        assert extract_json_from_text("") is None

    def test_extracts_json_array(self):
        raw = 'Preamble [{"item": 1}, {"item": 2}] done'
        result = extract_json_from_text(raw)
        assert result is not None
        assert '"item"' in result

    def test_handles_json_with_braces_in_strings(self):
        raw = 'text {"msg": "use {x} here"} end'
        result = extract_json_from_text(raw)
        assert result is not None

    def test_returns_first_valid_json_block(self):
        raw = 'noise {"first": true} more {"second": true} end'
        result = extract_json_from_text(raw)
        assert result is not None
        assert '"first"' in result
