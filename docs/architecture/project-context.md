# AxiomNode — Project Context

## What is AxiomNode?

**AxiomNode** is a **social and educational platform** where learning happens
through play. Users engage with each other via structured educational games —
trivia quizzes, pasapalabra roscos, true/false challenges, and more — with all
game content generated dynamically by local AI models.

The result is a platform that is:

- **Personalised** — content is generated on demand for any topic or difficulty level.
- **Dynamic** — no static question banks; each session can produce fresh content.
- **Private** — inference runs on local models (no external API calls required).
- **Scalable** — each game type is an independent microservice behind a common orchestrator.

---

## Core Objective

> Deploy and operate **local AI models** (GGUF / llama.cpp) that generate
> structured educational game content **dynamically, efficiently, and in a
> consistent format** — ready to be consumed by any game microservice.

The AI layer (`ai-engine`) is the engine that powers content generation.
It is not application logic — it is the infrastructure that any game
microservice calls to receive a validated, structured game definition
(JSON) without knowing anything about the underlying model.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AxiomNode App                            │
│              (Mobile / Web frontend — React Native, etc.)       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ REST / WebSocket
┌──────────────────────▼──────────────────────────────────────────┐
│                       Orchestrator                              │
│   • Routes requests to the correct game microservice            │
│   • Handles authentication, rate limiting, session state        │
│   • Aggregates results and leaderboard data                     │
└────┬──────────────────────┬────────────────────────┬────────────┘
     │                      │                        │
┌────▼────────┐   ┌─────────▼──────────┐   ┌────────▼───────────┐
│   Trivia    │   │    Pasapalabra     │   │   True / False     │
│ Microservice│   │   Microservice     │   │   Microservice     │
│             │   │                    │   │                    │
│ (game rules,│   │ (rosco logic,      │   │ (statement eval,   │
│  scoring)   │   │  letter tracking)  │   │  scoring)          │
└────┬────────┘   └─────────┬──────────┘   └────────┬───────────┘
     │                      │                        │
     └──────────────────────┼────────────────────────┘
                            │  Internal HTTP / gRPC
┌───────────────────────────▼─────────────────────────────────────┐
│                        AI Layer  (ai-engine)                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RAGPipeline                                            │   │
│  │  (ingestion → chunking → embedding → vector store)      │   │
│  │  (retrieval → context building)                         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │ context string                      │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  GameGenerator                                          │   │
│  │  (prompt template → LlamaClient → JSON extraction       │   │
│  │   → GameEnvelope validation)                            │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │ GameEnvelope (validated JSON)       │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  LlamaClient                                            │   │
│  │  (local GGUF model via llama-cpp-python                 │   │
│  │   OR llama.cpp HTTP server)                             │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐  │
│  │  KnowledgeBase│   │  StatsCollector│   │  Observability   │  │
│  │  (KBD layer) │   │  (metrics)     │   │  API (FastAPI)   │  │
│  └──────────────┘   └────────────────┘   └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Game Generation Request

```
User starts a game session in the App
           │
           ▼
   Orchestrator receives: { game_type, topic, language, num_questions }
           │
           ▼
   Routes to the matching Game Microservice
           │
           ▼
   Microservice calls ai-engine:
     GameGenerator.generate(
         query   = "<topic keywords>",
         topic   = "<educational topic>",
         game_type = "quiz" | "pasapalabra" | "true_false",
         language  = "es" | "en" | ...,
         num_questions = N,
     )
           │
           ├── RAGPipeline.build_context(query)
           │         → embed query → vector search → top-K chunks → context string
           │
           ├── get_prompt(game_type, context, topic, language, ...)
           │         → fills prompt template with JSON schema embedded
           │
           ├── LlamaClient.generate(prompt, json_mode=True)
           │         → local GGUF inference with JSON grammar constraint
           │
           └── GameEnvelope.from_dict(parsed_json)
                     → validated dataclass (QuizGame / PasapalabraGame / TrueFalseGame)
           │
           ▼
   Microservice applies game rules (scoring, timing, letter tracking, etc.)
   and returns a structured game session to the Orchestrator
           │
           ▼
   Orchestrator sends the session to the App
           │
           ▼
   User plays
```

---

## Responsibilities by Layer

### App (Frontend)
- User interface, social features (friends, challenges, leaderboard).
- Sends game requests with topic, language, and difficulty.
- Renders the game based on the structured session received.
- Does **not** know about AI or content generation.

### Orchestrator
- Single entry point for the App.
- Routes game requests to the correct microservice by `game_type`.
- Handles cross-cutting concerns: auth, rate limiting, session persistence,
  leaderboard aggregation.
- Composes responses from multiple microservices if needed (e.g. mixed game sessions).

### Game Microservices
- One service per game type (trivia, pasapalabra, true/false, …).
- Owns the **game rules and scoring logic** for that game type.
- Requests content from the AI layer (`ai-engine`) via an internal API.
- Returns a structured, ready-to-play session to the Orchestrator.
- Stateless with respect to content — content is always generated or cached,
  never hardcoded.

### AI Layer (ai-engine — this repository)
- Generates structured game content as validated JSON (`GameEnvelope`).
- Retrieves relevant educational context via the RAG pipeline.
- Runs local LLM inference (no external API dependency).
- Exposes an observability API for monitoring latency, error rates, and usage
  by game type.
- Has no knowledge of the App, user sessions, or business rules.

---

## Design Principles

| Principle | Application |
|---|---|
| **Loose coupling** | ai-engine and each game microservice are independent deployables. They communicate via a contract (JSON schema), not shared code. |
| **Local-first AI** | All inference runs on GGUF models via llama.cpp. No external API keys required. Models are swappable without changing any service. |
| **Structured output** | JSON grammar (GBNF) constrains LLM output at the sampler level. Downstream microservices receive clean, validated data — no fragile parsing. |
| **Observability** | Every AI generation call is recorded (`StatsCollector`). The FastAPI observability API exposes latency, success rates, and per-game-type metrics. |
| **Extensibility** | New game types require only: a new prompt template, a new schema dataclass, and a new game microservice. The AI layer and orchestrator need no changes. |
| **RAG for quality** | Game content is grounded in an ingested knowledge corpus, reducing hallucinations and ensuring educational relevance. |

---

## Adding a New Game Type

The architecture is open for extension:

1. **Add schema** — create a new dataclass in `ai_engine/games/schemas.py` and
   register it in `GAME_TYPE_REGISTRY`.
2. **Add prompt** — add a template to `ai_engine/games/prompts.py`.
3. **Wire `GameGenerator`** — `get_prompt()` already routes by `game_type` string.
4. **Write tests** — follow TDD: schema tests, prompt tests, generator mock tests.
5. **New microservice** — implement the game rules service; call `GameGenerator`
   with the new `game_type` string.

No changes to the Orchestrator or App are required until the new microservice
is ready to be registered.

---

## Recommended LLM for Game Content Generation

### Designated Model: Qwen2.5-7B-Instruct (Q4_K_M GGUF)

```
Model  : Qwen2.5-7B-Instruct
Format : GGUF  (llama.cpp compatible)
Quant  : Q4_K_M  (~4.8 GB on disk, ~5.5 GB peak RAM)
License: Apache-2.0
Source : https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF
```

### Why Qwen2.5-7B-Instruct?

#### 1 — Spanish language quality
AxiomNode targets primarily Spanish-speaking users. Qwen2.5 is trained on a
large multilingual corpus with strong Spanish representation, producing
naturally phrased questions, definitions, and statements that feel idiomatic
rather than machine-translated. Smaller ~3B models show measurable quality
drop in Spanish educational phrasing.

#### 2 — JSON instruction following
Game generation depends entirely on the model producing valid, schema-compliant
JSON (constrained at the sampler level with GBNF grammar via llama.cpp).
Qwen2.5-7B has been extensively benchmarked on structured output tasks and
consistently fills nested JSON schemas without degenerating — critical for
`QuizGame`, `PasapalabraGame`, and `TrueFalseGame` envelopes that the
microservices consume directly.

#### 3 — Factual accuracy for educational content
At 7B parameters, the model has significantly higher factual recall than 3B
variants. For a platform where wrong answers would undermine learning, factual
reliability is not cosmetic — it is a product requirement.

#### 4 — Local deployment fit
Q4_K_M quantization brings the model to ~4.8 GB on disk and ~5.5 GB peak RAM
during inference. A server with 8 GB of dedicated RAM (e.g., a single
consumer GPU or a small VPS) runs it comfortably. CPU-only inference is viable
for low-traffic scenarios; a CUDA-capable GPU reduces generation latency
from ~30 s to ~2–4 s per game set.

#### 5 — License
Apache-2.0 — fully permissive for commercial use, consistent with the
ai-engine license and the AxiomNode project goals.

### Model Selection Decision Matrix

| Criterion | Qwen2.5-7B | Phi-3.5-mini (3.8B) | Qwen2.5-3B | Llama-3.1-8B |
|---|---|---|---|---|
| Spanish quality | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ |
| JSON schema fidelity | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| Factual accuracy | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| RAM footprint | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★☆☆ |
| License | Apache-2.0 | MIT | Apache-2.0 | Llama 3.1 |
| **Overall fit** | **Recommended** | Lightweight fallback | Ultra-light fallback | Alternative |

### How to Download and Configure

```bash
# Set the models directory (defaults to models/ if not set)
export AI_ENGINE_MODELS_DIR=/path/to/models

# Download the recommended model
python -m ai_engine.llm.model_manager download \
  https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

```python
from ai_engine.llm.llama_client import LlamaClient

# Local GGUF inference (recommended)
client = LlamaClient(
    model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,          # context window
    n_gpu_layers=-1,     # -1 = offload all layers to GPU if available
)

# Or connect to a running llama.cpp HTTP server
client = LlamaClient(api_url="http://localhost:8080")
```

### Performance Expectations

| Hardware | Backend | ~Latency per game set (10 questions) |
|---|---|---|
| CPU only (8-core, no GPU) | llama-cpp-python | 25–45 s |
| NVIDIA GPU 6 GB VRAM (partial offload) | llama-cpp-python + CUDA | 4–8 s |
| NVIDIA GPU 8 GB VRAM (full offload) | llama-cpp-python + CUDA | 2–4 s |
| llama.cpp HTTP server (dedicated) | LlamaClient API backend | depends on server HW |

Latency is for a single synchronous call. The Orchestrator can parallelise
requests across multiple game microservices if the server has sufficient
memory to hold the model once and serve concurrent requests.

---

## Repository Scope

This repository (`ai-engine`) covers **only the AI layer**:

| In scope | Out of scope |
|---|---|
| RAG pipeline (ingestion, retrieval) | Orchestrator implementation |
| LLM client (local GGUF / HTTP) | Game microservices business logic |
| Game content generation + validation | User authentication |
| Knowledge Base (KBD) | Leaderboard and social features |
| Observability API | Frontend / mobile app |
| Model download tooling | Database persistence beyond in-memory |
