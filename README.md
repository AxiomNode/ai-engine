# ai-engine

AI Engine — RAG, local LLM inference and structured educational game generation.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Guide | Description |
|---|---|
| [Project Context](docs/project-context.md) | AxiomNode vision, high-level architecture, and ai-engine scope |
| [Architecture](docs/architecture.md) | System design, layers, data flows, and extension points |
| [Technologies](docs/technologies.md) | Libraries, tools, models, and standards |
| [Getting Started](docs/getting-started.md) | Local setup, TDD workflow, code quality |
| [Deployment](docs/deployment.md) | API server, Docker, llama.cpp server |
| [RAG Usage](docs/rag-usage.md) | Ingestion, retrieval, custom embedders and vector stores |
| [KBD Usage](docs/kbd-usage.md) | Knowledge Base CRUD, tag/keyword search |
| [SDK Usage](docs/sdk-usage.md) | Typed client-side models for generate endpoint responses |
| [Cache Strategy ADR](docs/adr-0001-cache-strategy.md) | Architecture decision for in-memory, TinyDB, and Redis cache layers |
| [Incident Runbook](docs/incident-runbook.md) | Operational response and rollback procedures for runtime incidents |

## Description

**ai-engine** provides the fundamental building blocks for creating
**Retrieval-Augmented Generation (RAG)** systems, a **Knowledge Base (KBD)**,
integration with **local language models** (llama.cpp / GGUF), a
**structured educational game generator** (quiz, pasapalabra, true/false),
and an **observability / stats API** (FastAPI) for monitoring model usage — all in Python.

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
│   ├── schemas.py                # Data models (Quiz, Pasapalabra, T/F)
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
pip install -e ".[dev]"
```

### With local LLM support (llama.cpp)

```bash
pip install -e ".[llm]"
```

### Full install (RAG + LLM + games)

```bash
pip install -e ".[games]"
```

### With observability API (FastAPI)

```bash
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
pytest
```

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
    game_type="quiz",        # "quiz" | "pasapalabra" | "true_false"
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
python scripts/demo_suite.py list
python scripts/demo_suite.py run kbd
python scripts/demo_suite.py run integration
python scripts/demo_suite.py run all
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
