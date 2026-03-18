# Repository Layout

This document describes the current folder distribution and conventions.

## Root

- `.github/workflows/`: CI pipelines.
- `docs/`: architecture, deployment, runbooks, ADRs, and checklists.
- `src/`: project implementation and operational assets.
- `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE`: repository-level documents.

## Project Root (`src/`)

- `ai_engine/`: source package.
- `tests/`: unit/integration test suites.
- `scripts/`: operational scripts grouped by purpose.
- `distributions/`: deployment matrix (`dev|stg|pro` x `windows|vps-cpu|vps-gpu`).
- `docker-compose.yml`, `Dockerfile`, `Dockerfile.llama`: container orchestration/build files.
- `models/`: local GGUF models (runtime artifacts, not versioned by default).
- `data/`: runtime data (cache and generated artifacts).
- `notebooks/`: exploratory and demo notebooks.
- `pyproject.toml`, `requirements.txt`: packaging and dependency metadata.

## Scripts

- `src/scripts/install/`: environment-specific installers.
- `src/scripts/demos/`: demo runners.
- `src/scripts/benchmarks/`: benchmark scripts.

Top-level wrappers in `src/scripts/` are preserved for backward compatibility.

## Docs

- `docs/operations/`: operational index and links to deployment/runbooks.

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
