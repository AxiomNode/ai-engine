# Scripts Layout

This folder is organized by purpose:

- `scripts/install/`: installation and deployment scripts by environment.
- `scripts/demos/`: demo entrypoints.
- `scripts/benchmarks/`: benchmark and baseline generation utilities.

Generation-quality benchmarking lives under `scripts/benchmarks/benchmark_generation_quality.py`
and runs a curated matrix of real topic queries against a live ai-engine endpoint.

Backward-compatible wrappers are kept at the top level (`scripts/demo_suite.py`,
`scripts/demo_rag_kbd.py`, and `scripts/benchmark_generation_paths.py`).

From repository root, invoke scripts using the `src/` prefix (for example,
`./src/scripts/install/deploy.sh dev windows`).

For the staging workstation topology, `scripts/install/deploy.ps1` can also
bootstrap the VPS relay path that exposes local `ai-engine` services to the
Kubernetes cluster through a reverse SSH tunnel.
