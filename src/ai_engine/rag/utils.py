"""Helper utilities for RAG pipeline."""

from __future__ import annotations

import json
from typing import Optional


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract the first JSON object found in *text* and return it as a string.

    Scans for the first '{' and attempts to find the matching closing '}'.
    Returns None if no valid JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    # validate JSON
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return None
    return None
