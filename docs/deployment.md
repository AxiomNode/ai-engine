# Deployment

This guide covers how to run ai-engine's components in different environments:
the observability API as a standalone service, the RAG pipeline as a library,
and the game generator from the command line or a web service.

---

## Option A — Library (embedded use)

The most common deployment: import ai-engine into your own application or
notebook. No server required.

```bash
pip install -e ".[games]"   # or install from PyPI when published
```

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
from ai_engine.llm import LlamaClient, model_path
from ai_engine.games import GameGenerator

embedder = SentenceTransformersEmbedder()
pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())
pipeline.ingest([Document(content="...", doc_id="1")])

llm = LlamaClient(model_path=str(model_path()), json_mode=True)
gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)
game = gen.generate(query="water cycle", topic="Science", game_type="quiz")
```

---

## Option B — Observability API (standalone FastAPI server)

The observability module ships a ready-to-run FastAPI application.

### Development server

```bash
pip install -e ".[api]"
uvicorn ai_engine.observability.api:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Production server (Gunicorn + Uvicorn workers)

```bash
pip install gunicorn
gunicorn ai_engine.observability.api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Mounting as a sub-application

Embed the stats API inside an existing FastAPI app:

```python
from fastapi import FastAPI
from ai_engine.observability import StatsCollector, create_app

main_app = FastAPI(title="My Application")
collector = StatsCollector()
stats_app = create_app(collector)

main_app.mount("/observability", stats_app)
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns status, uptime, and total event count |
| `GET` | `/stats` | Aggregate statistics (latency percentiles, success rate, etc.) |
| `GET` | `/stats/history?last_n=100` | Raw event log, optionally capped to last N events |
| `POST` | `/stats/reset` | Clear all recorded events |

Example response from `/stats`:

```json
{
  "total_calls": 42,
  "successful_calls": 40,
  "failed_calls": 2,
  "success_rate": 0.952,
  "avg_latency_ms": 1834.5,
  "p50_latency_ms": 1721.0,
  "p95_latency_ms": 3210.0,
  "p99_latency_ms": 4050.0,
  "json_mode_calls": 38,
  "game_type_counts": {"quiz": 30, "true_false": 8, "pasapalabra": 4}
}
```

---

## Option C — llama.cpp HTTP server + LlamaClient

Run `llama.cpp` as a dedicated inference server and point `LlamaClient` at it.
This decouples the model from the Python process, which is useful for shared
deployments or GPU servers.

### Start the llama.cpp server

```bash
# Build llama.cpp (or use a prebuilt binary / Docker image)
./server -m /path/to/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096
```

### Connect with LlamaClient

```python
from ai_engine.llm import LlamaClient

llm = LlamaClient(
    api_url="http://localhost:8080/completion",
    json_mode=True,
    temperature=0.2,
)
response = llm.generate("Explain photosynthesis.", max_tokens=256)
```

---

## Option D — Docker (recommended for production)

The repository ships a production-ready multi-stage `Dockerfile` and a
`docker-compose.yml` that orchestrates the full stack.

For CPU-only VPS deployments, the Docker setup is optimized to avoid GPU
runtime dependencies:

- `ai-api` uses a dedicated image target with `api + rag` dependencies and
  CPU-only PyTorch wheels.
- `ai-stats` uses a dedicated lightweight image target with only `api`
  dependencies (no RAG stack).

### Prerequisites

1. Download at least one model before building (models are **not** baked into
   the image — they are mounted as a read-only volume at runtime):

   ```bash
   python -m ai_engine.llm.model_manager download qwen2.5-3b   # ~2 GB
   # or the recommended production model (~4.8 GB):
   python -m ai_engine.llm.model_manager download
   ```

2. Copy the environment template and adjust if needed:

   ```bash
   cp .env.example .env
   ```

### Single image – observability API only

```bash
# Build
docker build -t ai-engine-stats .

# Run (mount the local models directory)
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  ai-engine-stats
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Full stack – docker compose

```bash
docker compose up --build
```

If you want the stack to consume more host resources, raise the limits in
`.env` before starting Compose. Example:

```bash
LLAMA_CPUS=8.0
LLAMA_MEMORY_LIMIT=16g
LLAMA_MEMORY_RESERVATION=12g
LLAMA_SHM_SIZE=3g
API_CPUS=6.0
API_MEMORY_LIMIT=10g
API_MEMORY_RESERVATION=6g
API_SHM_SIZE=2g
API_WORKERS=4
```

On Windows, these values are still capped by Docker Desktop's global limits.
If Compose does not use the requested memory/CPU, increase the Docker Desktop
or WSL 2 allocation first, then restart Docker Desktop.

This starts three services:

| Service | Port | Description |
|---|---|---|
| `llama-server` | 8080 | llama.cpp HTTP inference server |
| `ai-stats` | 8000 | ai-engine observability / statistics FastAPI |
| `ai-api` | 8001 | ai-engine generation FastAPI |

> **Tip:** Override the model file or port without editing `docker-compose.yml`
> by setting variables in `.env` (see `.env.example` for all options).

### Useful compose commands

```bash
# Start in background
docker compose up -d --build

# Follow logs for a specific service
docker compose logs -f ai-stats

# Stop and remove containers (volumes are preserved)
docker compose down

# Rebuild after code changes
docker compose up --build --force-recreate ai-stats
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `AI_ENGINE_MODELS_DIR` | `<project_root>/models/` | Path where GGUF model files are stored. Override to point to a shared network drive or Docker volume. |
| `LLAMA_MODEL_FILE` | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | GGUF filename loaded by the `llama-server` compose service. |
| `LLAMA_CTX_SIZE` | `4096` | Context window size (tokens) for the llama.cpp server. |
| `LLAMA_PORT` | `8080` | Host port exposed by the `llama-server` service. |
| `LLAMA_CPUS` | `6.0` | CPU quota assigned to `llama-server` by Docker Compose. |
| `LLAMA_MEMORY_LIMIT` | `12g` | Hard memory limit for `llama-server`. |
| `LLAMA_MEMORY_RESERVATION` | `8g` | Soft memory reservation for `llama-server`. |
| `LLAMA_SHM_SIZE` | `2g` | Shared memory available to `llama-server`. |
| `STATS_PORT` | `8000` | Host port exposed by the `ai-stats` service. |
| `STATS_CPUS` | `1.0` | CPU quota assigned to `ai-stats`. |
| `STATS_MEMORY_LIMIT` | `1g` | Hard memory limit for `ai-stats`. |
| `STATS_MEMORY_RESERVATION` | `512m` | Soft memory reservation for `ai-stats`. |
| `API_PORT` | `8001` | Host port exposed by the `ai-api` service. |
| `API_CPUS` | `4.0` | CPU quota assigned to `ai-api`. |
| `API_MEMORY_LIMIT` | `8g` | Hard memory limit for `ai-api`. |
| `API_MEMORY_RESERVATION` | `4g` | Soft memory reservation for `ai-api`. |
| `API_SHM_SIZE` | `1g` | Shared memory available to `ai-api`. |
| `API_WORKERS` | `2` | Number of Uvicorn workers used by `ai-api`. |
| `STATS_WORKERS` | `1` | Number of Uvicorn workers used by `ai-stats`. |
| `AI_ENGINE_API_KEY` | *(unset)* | Shared secret for the future generation API (`X-API-Key` header). |

### Windows note: Docker Desktop resource ceiling

If you use Docker Desktop on Windows, Compose cannot exceed the global limits
assigned to the Linux backend.

1. Open Docker Desktop.
2. Go to Settings.
3. If you use the Hyper-V backend, open Resources and raise CPUs, Memory, and Swap.
4. If you use WSL 2, raise the limits for the `docker-desktop` WSL environment or remove restrictive values from your `.wslconfig`.
5. Restart Docker Desktop.

After that, run:

```bash
docker compose down
docker compose up -d --build
```

---

## Health check for orchestrators (Kubernetes / ECS)

Use the `/health` endpoint as a liveness probe:

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Running the demo notebooks

```bash
pip install -e ".[games,api]"
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/demo_rag.ipynb
```

The notebook demonstrates an end-to-end RAG + game generation flow and can
be used as a smoke test for a freshly configured environment.
