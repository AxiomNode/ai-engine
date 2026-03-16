# Technologies

This document lists every technology, library, and tool used in ai-engine,
with its purpose and the minimum required version.

---

## Runtime

### Core

| Library | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.10 | Language runtime. Uses `match`, PEP 604 union types (`X \| Y`), and `from __future__ import annotations`. |
| **requests** | ≥ 2.30 | HTTP client used by `LlamaClient` (API backend) and `model_manager` (model download). |

### RAG extras (`pip install -e ".[rag]"`)

| Library | Version | Purpose |
|---|---|---|
| **sentence-transformers** | ≥ 2.2 | Produces dense vector embeddings from text via `SentenceTransformersEmbedder`. Default model: `all-MiniLM-L6-v2` (384-dim, ~80 MB). |
| **chromadb** | ≥ 0.3 | Declared as an optional dependency for future persistent `VectorStore` backends. Not yet wired up — use `InMemoryVectorStore` for development. |

### LLM extras (`pip install -e ".[llm]"`)

| Library | Version | Purpose |
|---|---|---|
| **llama-cpp-python** | ≥ 0.2.0 | Python bindings for `llama.cpp`. Loads GGUF model files and runs inference on CPU / GPU via `LlamaClient._generate_local()`. |
| **PyYAML** | ≥ 6.0 | Declared as an optional LLM dependency for potential config-file loading. |

### Games extras (`pip install -e ".[games]"`)

Pulls `llama-cpp-python` + `sentence-transformers` together — everything
needed to run `GameGenerator` end-to-end locally.

### KBD extras (`pip install -e ".[kbd]"`)

| Library | Version | Purpose |
|---|---|---|
| **tinydb** | ≥ 4.7 | Declared as an optional persistence backend for `KnowledgeBase`. Not yet wired up — the default implementation is in-memory. |

### API extras (`pip install -e ".[api]"`)

| Library | Version | Purpose |
|---|---|---|
| **FastAPI** | ≥ 0.110 | Web framework for the observability REST API (`/health`, `/stats`, `/stats/history`, `/stats/reset`). |
| **uvicorn[standard]** | ≥ 0.29 | ASGI server used to run the FastAPI application. |
| **httpx** | ≥ 0.27 | Async HTTP client used by FastAPI's test client in tests. |

### Redis extras (`pip install -e ".[redis]"`)

| Library | Version | Purpose |
|---|---|---|
| **redis** | ≥ 5.0 | Optional persistent cache backend client used by generation optimization service. |

---

## Development & Tooling

| Tool | Version | Purpose |
|---|---|---|
| **pytest** | ≥ 7.4 | Test runner for unit and integration tests. |
| **pytest-cov** | ≥ 4.1 | Coverage plugin — reports line coverage per module. |
| **ruff** | ≥ 0.4 | Fast Python linter (replaces flake8 + many plugins). Configured in `pyproject.toml`. |
| **black** | ≥ 23.9.1 | Opinionated code formatter. Line length: 88. |
| **isort** | ≥ 5.12 | Import sorter, configured with `profile = "black"`. |
| **mypy** | ≥ 1.5 | Static type checker. `python_version = "3.10"`, `ignore_missing_imports = true`. |
| **types-requests** | ≥ 2.31 | Type stubs for the `requests` library (mypy). |
| **pre-commit** | ≥ 3.4 | Git hook runner. Executes `black`, `isort`, `ruff`, and common file checks on each commit. |

---

## CI / CD

| Service | Configuration | What it does |
|---|---|---|
| **GitHub Actions** | `.github/workflows/ci.yml` | Runs on every push to `main`/`develop` and on every PR. Main matrix: Python 3.10, 3.11, 3.12. Optional-extras profile matrix: `core_api`, `rag_kbd`, `redis` to catch dependency-profile regressions. |

---

## Models

### LLM — Game Content Generation

The **designated model** for AxiomNode game content generation is
**Qwen2.5-7B-Instruct Q4_K_M**. See [project-context.md](project-context.md)
for the full selection rationale.

| Tier | Model | Format | Quant | Size | License | When to use |
|---|---|---|---|---|---|---|
| **Recommended** | **Qwen2.5-7B-Instruct** | GGUF | Q4_K_M | ~4.8 GB | Apache-2.0 | Production — best balance of Spanish quality, JSON fidelity, and local inference speed. |
| Lightweight | Phi-3.5-mini-instruct | GGUF | Q4_K_M | ~2.4 GB | MIT | Resource-constrained environments (≤ 8 GB RAM with OS overhead). |
| Lightweight | Qwen2.5-3B-Instruct | GGUF | Q4_K_M | ~2 GB | Apache-2.0 | Ultra-low-resource fallback. Acceptable JSON output. |

Model files are downloaded via `python -m ai_engine.llm.model_manager download <url>` and
stored in the directory set by the `AI_ENGINE_MODELS_DIR` environment variable
(default: `models/`).

### Embeddings — RAG Context Retrieval

| Model | Dimensions | Size | Source | Purpose |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | ~80 MB | Hugging Face (sentence-transformers) | Default dense embedder for the RAG pipeline. Fast CPU inference. Apache-2.0 licence. |

---

## Standards & Conventions

| Standard | Where applied |
|---|---|
| **Conventional Commits** | All commit messages (`feat:`, `fix:`, `chore:`, …) |
| **Google-style docstrings** | All public modules, classes, methods, functions |
| **PEP 517/518** (pyproject.toml) | Build system configuration |
| **GBNF grammar** | JSON-constrained sampling in `LlamaClient` (`JSON_GRAMMAR` constant) |
| **Dataclasses** | All data models (`Document`, `KnowledgeEntry`, `GenerationEvent`, game schemas) |
