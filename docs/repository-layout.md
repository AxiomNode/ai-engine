# Repository Layout

This document describes the current folder distribution and conventions.

## Root

- `.github/workflows/`: CI pipelines.
- `docs/`: architecture, deployment, runbooks, ADRs, and checklists.
- `models/`: local GGUF models (runtime artifacts, not versioned by default).
- `notebooks/`: exploratory and demo notebooks.
- `scripts/`: operational scripts grouped by purpose.
- `src/`: source package (`ai_engine`).
- `tests/`: unit/integration test suites.
- `data/`: runtime data (cache and generated artifacts).

## Scripts

- `scripts/install/`: environment-specific installers.
- `scripts/demos/`: demo runners.
- `scripts/benchmarks/`: benchmark scripts.

Top-level wrappers in `scripts/` are preserved for backward compatibility.

## Source Package (`src/ai_engine`)

- `api/`: generation API, schemas, middleware, and optimization orchestration.
- `cli/`: reusable CLI implementations.
- `games/`: game schemas, prompts, and generation logic.
- `kbd/`: knowledge base interfaces and TinyDB backend.
- `llm/`: llama client and model management.
- `observability/`: collector, metrics API, and middleware wrappers.
- `rag/`: retrieval pipeline, vector interfaces, embedders, and vector stores.
- `sdk/`: typed SDK models and language helpers.

## RAG Vector Store Naming

For consistency, the preferred import path for persistent stores is now:

- `ai_engine.rag.vectorstores`

The legacy path remains supported for compatibility:

- `ai_engine.rag.vectorstore`
