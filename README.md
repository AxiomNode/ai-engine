# ai-engine

AI Engine — RAG, local LLM inference and structured educational game generation.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Guide | Description |
|---|---|
| [Architecture Index](docs/architecture/README.md) | Context, system design, technologies, ADRs, and repository layout |
| [Distributions Matrix](src/distributions/README.md) | Stage/environment deployment matrix (dev/stg/pro x windows/vps-cpu/vps-gpu) |
| [Guides Index](docs/guides/README.md) | Getting started, RAG, KBD, SDK, and metrics usage guides |
| [Operations Index](docs/operations/README.md) | Deployment, incident runbook, and continuous improvement logs |

## Description

**ai-engine** provides the fundamental building blocks for creating
**Retrieval-Augmented Generation (RAG)** systems, a **Knowledge Base (KBD)**,
integration with **local language models** (llama.cpp / GGUF), a
**structured educational game generator** (quiz, word-pass, true/false),
and an **observability / stats API** (FastAPI) for monitoring model usage — all in Python.

## Integration in the target architecture

`ai-engine` runs as an internal AI capability service for domain microservices.

- It is not intended as a public endpoint for end users.
- Expected consumers: `microservice-quizz`, `microservice-wordpass`, and other authorized internal services.
- Recommended exposure: private network with service-to-service access controls.

Initial internal contract:

- `contracts-and-schemas/schemas/openapi/internal-ai-engine.v1.yaml`

## Core runtime components

- `ai-engine-api`: primary AI generation and ingestion API, with cache and RAG integration.
- `ai-engine-stats`: AI observability API for events and aggregated runtime metrics.
- `llama-server`: local LLM inference runtime used by the engine.

## Project Structure

```
src/ai_engine/
├── rag/                          # RAG module
│   ├── document.py               # Document model
│   ├── chunker.py                # Text splitting into chunks
│   ├── embedder.py               # Embedding interface (abstract)
│   ├── vector_store.py           # Vector store (abstract + InMemory)
│   ├── retriever.py              # Relevant document retrieval
│   ├── pipeline.py               # End-to-end orchestrator
│   ├── utils.py                  # Helpers (JSON extraction, etc.)
│   └── embedders/
│       └── sentence_transformers.py  # SentenceTransformers embedder
├── llm/                          # LLM module
│   ├── llama_client.py           # llama.cpp client (HTTP API + local GGUF)
│   └── model_manager.py          # GGUF model download and management
├── games/                        # Educational games module
│   ├── schemas.py                # Data models (Quiz, WordPass, T/F)
│   ├── prompts.py                # Prompt templates per game type
│   └── generator.py              # Orchestrator: RAG + LLM → structured game
├── observability/                # Stats & monitoring module
│   ├── collector.py              # Thread-safe event collector
│   ├── middleware.py             # LLM/GameGenerator instrumentation
│   └── api.py                    # FastAPI endpoints (/stats, /health, …)
└── kbd/                          # KBD module
    ├── entry.py                  # Knowledge entry model
    └── knowledge_base.py         # Knowledge base management
```

## Installation

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

## Download an LLM Model

The recommended model is **Phi-3.5-mini-instruct** (Q4_K_M, ~2.4 GB),
excellent for structured JSON generation:

```bash
python -m ai_engine.llm.model_manager download
```

To list all available models:

```bash
python -m ai_engine.llm.model_manager list
```

## Tests

```bash
cd src
pytest
```

## CI/CD workflow behavior

- `.github/workflows/ci.yml`
    - Trigger: push (`main`, `develop`) and pull request.
    - `test` job: lint, formatting, import order, required type check, and coverage-gated tests.
    - `optional-extras-matrix` job: profile-based validation for `core_api`, `rag_kbd`, and `redis` extras.
    - `trigger-platform-infra-build` job:
        - Runs on push to `main`.
        - Dispatches `platform-infra/.github/workflows/build-push.yaml` twice:
            - `service=ai-engine-api`
            - `service=ai-engine-stats`
        - Requires `PLATFORM_INFRA_DISPATCH_TOKEN` in this repository.

## Deployment automation chain

Push to `main` triggers image rebuilds in `platform-infra`, followed by automatic deployment to `dev`.

## Quick Start

### RAG

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
# Implement Embedder with your preferred model (OpenAI, HuggingFace, etc.)

pipeline = RAGPipeline(embedder=my_embedder, vector_store=InMemoryVectorStore())
pipeline.ingest([Document(content="Python is a programming language...", doc_id="1")])
context = pipeline.build_context("What is Python?")
```

### Generate an Educational Game

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
from ai_engine.llm import LlamaClient, model_path
from ai_engine.games import GameGenerator

# 1. Set up RAG
embedder = SentenceTransformersEmbedder()
pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())
pipeline.ingest([Document(content="The water cycle consists of...", doc_id="1")])

# 2. Set up local LLM
llm = LlamaClient(model_path=str(model_path()), json_mode=True)

# 3. Generate a quiz
gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)
game = gen.generate(
    query="water cycle",
    topic="Natural Sciences",
    game_type="quiz",        # "quiz" | "word-pass" | "true_false"
    num_questions=5,
    language="en",
)
print(game.game.to_dict())
```

### Observability API

```python
from ai_engine.observability import StatsCollector, create_app

collector = StatsCollector()
app = create_app(collector)
# Run with: uvicorn ai_engine.observability.api:app
```

### KBD

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()
kb.add(KnowledgeEntry("1", "Python", "Python is a high-level language", tags=["python"]))
results = kb.search_by_tag("python")
```

## CLI Demo Suite

Run simple, deterministic demos for each module and for cross-module integration:

```bash
python src/scripts/demo_suite.py list
python src/scripts/demo_suite.py run kbd
python src/scripts/demo_suite.py run integration
python src/scripts/demo_suite.py run all
```

The suite prints stylized output with concrete pass/fail results and a final summary table.

## SDK Models For Generated Games

The project ships a small SDK under `ai_engine.sdk` to parse `/generate` responses
into typed objects.

```python
from ai_engine.sdk import LanguageCode, parse_generate_response

payload = {
    "game_type": "quiz",
    "game": {
        "game_type": "quiz",
        "title": "Science Quiz",
        "topic": "Science",
        "questions": [
            {
                "question": "What is H2O?",
                "options": ["Water", "Fire", "Air", "Earth"],
                "correct_index": 0,
                "explanation": "H2O is water.",
            }
        ],
    },
}

game = parse_generate_response(payload, language=LanguageCode.EN)
print(game.metadata.language_id)  # lang-en
```
