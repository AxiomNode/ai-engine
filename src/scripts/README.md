# Scripts Layout

This folder is organized by purpose:

- `scripts/install/`: installation and deployment scripts by environment.
- `scripts/demos/`: demo entrypoints.
- `scripts/benchmarks/`: benchmark and baseline generation utilities.

Backward-compatible wrappers are kept at the top level (`scripts/demo_suite.py`,
`scripts/demo_rag_kbd.py`, and `scripts/benchmark_generation_paths.py`).

From repository root, invoke scripts using the `src/` prefix (for example,
`./src/scripts/install/deploy.sh dev windows`).
