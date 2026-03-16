"""CLI wrapper to run the AI-Engine demo suite.

Usage:
    python scripts/demo_suite.py list
    python scripts/demo_suite.py run kbd
    python scripts/demo_suite.py run all
"""

from __future__ import annotations

from ai_engine.cli.demo_suite import main

if __name__ == "__main__":
    raise SystemExit(main())
