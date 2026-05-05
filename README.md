# ai-engine

Last updated: 2026-05-03.

[![codecov](https://codecov.io/gh/AxiomNode/ai-engine/branch/main/graph/badge.svg)](https://codecov.io/gh/AxiomNode/ai-engine)

AI Engine — RAG, local LLM inference and structured educational game generation.

## Responsibility

`ai-engine` provides the shared AI runtime capability for AxiomNode: retrieval-augmented generation, structured educational game generation, local or split llama execution, and observability of AI activity.

It owns AI-specific generation, ingest, cache-aware behavior, and model-target selection. It does not own the domain validity or persistence rules enforced later by quiz and word-pass services.

## Runtime role

### Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Guide | Description |
|---|---|
| [Architecture Index](docs/architecture/README.md) | Context, system design, technologies, ADRs, and repository layout |
| [Distributions Matrix](src/distributions/README.md) | Stage/environment deployment matrix (dev/stg/pro x windows/windows-gpu/vps-cpu/vps-gpu) |
| [Guides Index](docs/guides/README.md) | Getting started, RAG, KBD, SDK, and metrics usage guides |
| [Operations Index](docs/operations/README.md) | Deployment, incident runbook, and continuous improvement logs |

## Documentation

- `docs/README.md`
- `docs/architecture/README.md`
- `docs/guides/README.md`
- `docs/operations/README.md`
- `src/distributions/README.md`

### Description

`ai-engine` is both a Python library surface and a deployable runtime split into API, stats, and llama-serving concerns.

### Integration in the current platform architecture

`ai-engine` runs as an internal AI capability service for domain microservices.

- It is not intended as a public endpoint for end users.
- Expected consumers: `microservice-quizz`, `microservice-wordpass`, and other authorized internal services.
- Recommended exposure: private network with service-to-service access controls.

Initial internal contract:

- `contracts-and-schemas/schemas/openapi/internal-ai-engine.v1.yaml`

## Runtime surface

### Deployment shape

The repository produces multiple runtime components that can be deployed together or in a split topology:

- `ai-engine-api`: generation and ingest surface
- `ai-engine-stats`: AI observability and cache/statistics surface
- `llama-server`: model-serving runtime used by `ai-engine-api`
- Redis cache backend in local or split-runtime environments

### Core runtime components

- `ai-engine-api`: primary AI generation and ingestion API, with cache and RAG integration.
- `ai-engine-stats`: AI observability API for events and aggregated runtime metrics.
- `llama-server`: local LLM inference runtime used by the engine.

Detailed runtime topology, module boundaries, and distribution rules live under `docs/architecture/README.md`, `docs/operations/README.md`, and `src/distributions/README.md`.

### Runtime state and target resolution

`ai-engine-api` can persist runtime model-target information so that the active llama destination survives process restart or pod recreation. Effective model routing can therefore differ from environment defaults during controlled operations.

## Local setup

### Installation

### Development

```bash
cd src
pip install -e ".[dev]"
```

### With local LLM support (llama.cpp)

```bash
cd src
pip install -e ".[llm]"
```

### Full install (RAG + LLM + games)

```bash
cd src
pip install -e ".[games]"
```

### With observability API (FastAPI)

```bash
cd src
pip install -e ".[api]"
```

### Download an LLM model

The recommended model is **Phi-3.5-mini-instruct** (Q4_K_M, ~2.4 GB),
excellent for structured JSON generation:

```bash
python -m ai_engine.llm.model_manager download
```

To list all available models:

```bash
python -m ai_engine.llm.model_manager list
```

### Tests

```bash
cd src
pytest
```

For module layout, usage guides, and distribution-specific install flows, use the local documentation indexes instead of duplicating that detail here.

## Dependencies and contracts

### Dependency model

Core dependencies vary by runtime slice:

- `ai-engine-api`: llama runtime, embeddings/model assets, cache backend, stats API
- `ai-engine-stats`: internal collector state and optional bridge to API runtime information
- `llama-server`: GGUF models and machine-level CPU/GPU resources

Primary consumers:

- `microservice-quizz`
- `microservice-wordpass`
- `bff-backoffice` diagnostics and stats flows
- `api-gateway` internal AI proxy routes

## Deployment and operations notes

- CI, distribution-specific rollout, and incident recovery are documented in `docs/operations/README.md`.
- The split runtime model and deployment matrix are documented in `src/distributions/README.md`.
- Cross-repo behavior is documented in `../docs/guides/capabilities/ai/ai-content-generation.md` and `../docs/guides/capabilities/ai/ai-observability-and-diagnostics.md`.

## References

- `docs/guides/README.md` for usage guides.
- `docs/operations/README.md` for deployment and recovery.
- `docs/architecture/README.md` for module boundaries and ADRs.
- `src/distributions/README.md` for the deployment matrix.
- `../docs/guides/capabilities/ai/ai-content-generation.md` for the cross-repo capability view.
