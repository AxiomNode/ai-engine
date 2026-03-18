"""Backward-compatible entrypoint for demo suite script.

Prefer:
    python scripts/demos/demo_suite.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "demos" / "demo_suite.py"
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
