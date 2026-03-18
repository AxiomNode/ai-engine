"""Utility helpers for the RAG pipeline."""

from __future__ import annotations

import json


def extract_json_from_text(text: str) -> str | None:
    """Extract the first valid JSON object or array from *text*.

    Useful when an LLM returns chain-of-thought reasoning followed by a
    JSON block.  The function scans for the first ``{`` or ``[`` and
    attempts to parse progressively longer substrings until it finds
    valid JSON.

    Args:
        text: Raw text that may contain embedded JSON.

    Returns:
        The extracted JSON substring, or ``None`` if no valid JSON is
        found.

    Examples:
        >>> extract_json_from_text('thinking... {"a": 1} done')
        '{"a": 1}'
        >>> extract_json_from_text('no json') is None
        True
    """
    if not text:
        return None

    # Try object ({...}) first, then array ([...])
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        if start == -1:
            continue

        # Walk from the end backwards to find the matching close bracket
        for end in range(len(text), start, -1):
            candidate = text[start:end]
            if candidate.rstrip()[-1:] != close_char:
                continue
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, IndexError):
                continue

    return None
