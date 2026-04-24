# ai-engine package

This directory contains the Python package and runtime resources used by the project.

## Diagnostics test runner

The diagnostics runner behind `POST /diagnostics/tests/run` executes in a background thread and now includes bounded real-performance checks:

- retrieval latency samples against the active RAG pipeline
- end-to-end generation latency samples against the active generator/LLM path

Progress is exposed via `GET /diagnostics/tests/status` with incremental fields (`progress`, `suites`, `summary`) and operator recommendations under `recommendations`.

For full project documentation, see the repository root `README.md`.
