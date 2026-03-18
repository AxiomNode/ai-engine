"""Backward-compatible entrypoint for benchmark scripts.

Prefer:
    python scripts/benchmarks/benchmark_generation_paths.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = (
        Path(__file__).resolve().parent / "benchmarks" / "benchmark_generation_paths.py"
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
