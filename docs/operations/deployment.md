# Deployment

This guide covers how to run ai-engine's components in different environments:
the observability API as a standalone service, the RAG pipeline as a library,
and the game generator from the command line or a web service.

---

## Option A — Library (embedded use)

The most common deployment: import ai-engine into your own application or
notebook. No server required.

```bash
cd src
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
cd src
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
  "game_type_counts": {"quiz": 30, "true_false": 8, "word-pass": 4}
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
`src/docker-compose.yml` that orchestrates the full stack.

### Distribution Matrix

The repository now defines deployment distributions by stage and environment:

- stages: `dev`, `stg`, `pro`
- environments: `windows`, `vps-cpu`, `vps-gpu`

Files live in:

- `src/distributions/<stage>/<environment>.env`

Use the unified installers:

```bash
# Linux/macOS shells
./src/scripts/install/deploy.sh dev vps-cpu
./src/scripts/install/deploy.sh stg vps-gpu
./src/scripts/install/deploy.sh pro vps-cpu
```

```powershell
# Windows PowerShell
./src/scripts/install/deploy.ps1 -Stage dev -Environment windows
./src/scripts/install/deploy.ps1 -Stage stg -Environment vps-cpu
./src/scripts/install/deploy.ps1 -Stage pro -Environment vps-gpu
```

All distributions now propagate deployment tags through runtime channels:

- response header: `X-Distribution-Version`
- monitoring payloads (`/health`, `/stats`, `/metrics`): `distribution_version`
- Prometheus metrics label: `distribution_version` via
  `ai_engine_distribution_version_info` and
  `ai_engine_distribution_version_calls`

The label value is computed from env vars:

- `AI_ENGINE_DISTRIBUTION`
- `AI_ENGINE_RELEASE_VERSION`

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
  python -m ai_engine.llm.model_manager download               # recommended 7B (~4.8 GB)
  # or use the ultra-light fallback only if you are resource-constrained:
  python -m ai_engine.llm.model_manager download qwen2.5-3b   # ~2 GB
   ```

2. Copy the environment template and adjust if needed:

   ```bash
  cp src/distributions/examples/.env.example src/.env
   ```

### Single image – observability API only

```bash
# Build
docker build \
  --build-arg APP_NAME=AxiomNode \
  --build-arg SERVICE_NAME=ai_stats \
  --build-arg DISTRIBUTION=dev \
  --build-arg RELEASE_VERSION=v1 \
  --target runtime-stats \
  -t AxiomNode_ai_stats_dev_v1 \
  -f src/Dockerfile src

# Run (mount the local models directory)
docker run -p 8000:8000 \
  -v "$(pwd)/src/models:/app/models:ro" \
  AxiomNode_ai_stats_dev_v1
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Full stack – docker compose

```bash
docker compose -f src/docker-compose.yml --profile cpu up --build
```

If port `8080` is already used on your machine, start with a different host
port for llama-server:

```bash
docker compose \
  --env-file src/distributions/dev/windows.env \
  --profile cpu \
  -f src/docker-compose.yml \
  up -d --build
```

`src/docker-compose.yml` publishes host ports from the distribution env file.
Default mappings remain `7002 -> 8080`, `7001 -> 8001`, and `7000 -> 8000`.
On Windows workstation deployments, `src/scripts/install/deploy.ps1` can also
configure stable public TCP listeners through firewall rules plus
`netsh interface portproxy` when `AUTO_EXPOSE_PUBLIC_PORTS=true` is set in the
distribution env file.

For the staging topology where ai-engine runs on a workstation and the rest of
the platform runs on the VPS Kubernetes cluster, prefer `AUTO_EXPOSE_VPS_RELAY`
instead. That mode opens a reverse SSH tunnel from the workstation to the VPS
and starts a tiny TCP relay on the VPS, so STG reaches ai-engine through the
VPS relay host instead of requiring inbound access directly to the workstation.

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

With `--profile cpu` this starts four services:

| Service | Port | Description |
|---|---|---|
| `llama-server` | `${LLAMA_PORT:-7002}` -> 8080 | llama.cpp HTTP inference server |
| `ai-stats` | `${STATS_PORT:-7000}` -> 8000 | Monitoring/backoffice API (`/health`, `/stats*`, `/cache/*`, `/metrics`) |
| `ai-api` | `${API_PORT:-7001}` -> 8001 | Game API for microservices (`/generate*`, `/ingest*`, `/health`) |
| `ai-cache` | (internal) | Redis cache backend with AOF persistence volume (`ai_cache_data`) |

Without any profile, only `ai-api` and `ai-stats` are started (llama services
are profile-gated).

### Service Integration Contract (Backoffice vs Microservices)

| Consumer | Service | Endpoint family | Purpose | Required key/header |
|---|---|---|---|---|
| Backoffice | `ai-stats` | `/health`, `/stats`, `/stats/history`, `/metrics` | Platform monitoring and operational dashboards | `X-API-Key` = `AI_ENGINE_BRIDGE_API_KEY` (or `AI_ENGINE_API_KEY` fallback); `/health` can remain public |
| Backoffice | `ai-stats` | `/cache/stats`, `/cache/reset` | Cache runtime visibility and selective/global invalidation | `X-API-Key` = `AI_ENGINE_BRIDGE_API_KEY` (or `AI_ENGINE_API_KEY` fallback) |
| Game microservices | `ai-api` | `/generate*` | Structured game generation for quiz/word-pass/true-false | `X-API-Key` = `AI_ENGINE_GAMES_API_KEY` (or `AI_ENGINE_API_KEY` fallback) |
| Bridge/integration service | `ai-api` | `/ingest*` | RAG corpus ingestion from external systems | `X-API-Key` = `AI_ENGINE_BRIDGE_API_KEY` (or `AI_ENGINE_API_KEY` fallback) |
| ai-api (internal publisher) | `ai-stats` | `/events` | Push generation/ingest telemetry events | `X-API-Key` = `AI_ENGINE_STATS_API_KEY` (or `AI_ENGINE_BRIDGE_API_KEY` / `AI_ENGINE_API_KEY` fallback) |

Notes:

- `ai-stats` is the only public monitoring surface for backoffice consumption.
- `ai-api` exposes only game generation and RAG ingestion as public business routes.
- Cache endpoints are bridged by `ai-stats` to `ai-api` internal routes (`/internal/cache/*`) and are intentionally hidden from `ai-api` OpenAPI.

Compose image/container names follow:

- `<APP_NAME>_<SERVICE_NAME>_<AI_ENGINE_DISTRIBUTION>_<AI_ENGINE_RELEASE_VERSION>`

Example default values:

- `AxiomNode_ai_api_dev_v1`
- `AxiomNode_ai_stats_dev_v1`
- `AxiomNode_llama_server_cpu_dev_v1`

Responsibility split:

- `ai-stats` is the public monitoring surface for backoffice systems.
- `ai-api` is focused on generation and ingestion workflows.
- Cache monitoring/reset is exposed by `ai-stats` and internally bridged to
  `ai-api` via `/internal/cache/*`.
> **Tip:** Override the model file or port without editing `src/docker-compose.yml`
> by setting variables in `src/.env` (see `src/distributions/examples/.env.example` for all options).

### Useful compose commands

```bash
# Start in background
docker compose -f src/docker-compose.yml --profile cpu up -d --build

# Follow logs for a specific service
docker compose -f src/docker-compose.yml logs -f ai-stats

# Stop and remove containers (volumes are preserved)
docker compose -f src/docker-compose.yml --profile cpu down

# Rebuild after code changes
docker compose -f src/docker-compose.yml --profile cpu up --build --force-recreate ai-stats
```

For Windows workstations, prefer the wrapper script so host exposure is kept in
sync with the distribution env file:

```powershell
./src/scripts/install/deploy.ps1 -Stage stg -Environment windows-gpu
```

Relevant Windows exposure variables in the env file:

- `AUTO_EXPOSE_PUBLIC_PORTS=true` enables firewall + portproxy automation.
- `STATS_PUBLIC_PORT=27000` exposes `ai-stats` externally on TCP 27000.
- `API_PUBLIC_PORT=27001` exposes `ai-api` externally on TCP 27001.
- `LLAMA_PUBLIC_PORT` is optional and usually not needed for backoffice flows.

Relevant VPS relay variables in the env file:

- `AUTO_EXPOSE_VPS_RELAY=true` enables reverse-tunnel + VPS relay automation.
- `VPS_RELAY_SSH_HOST` sets the SSH destination of the VPS.
- `VPS_RELAY_PUBLIC_HOST` is the host that STG should use as ai-engine target.
- `VPS_RELAY_STATS_PUBLIC_PORT=27000` publishes stats on the VPS.
- `VPS_RELAY_API_PUBLIC_PORT=27001` publishes api on the VPS.
- `VPS_RELAY_STATS_TUNNEL_PORT=27100` and `VPS_RELAY_API_TUNNEL_PORT=27101`
  are loopback-only ports on the VPS used by the reverse tunnel.

Those settings configure the Windows host only. If the workstation is behind a
router or carrier NAT, you still need upstream forwarding or a tunnel so the
internet can reach the machine.

In the current staging topology, the relay mode is the preferred tunnel.

### Linux VPS (CPU-only) optimized profile

The repository uses a centralized single `src/docker-compose.yml` with
runtime profiles (`cpu`, `gpu`).

1. Start stack using a distribution env file + CPU profile:

```bash
docker compose \
  --env-file src/distributions/stg/vps-cpu.env \
  --profile cpu \
  -f src/docker-compose.yml \
  up -d --build
```

3. Validate deployment:

```bash
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/stats
curl -sS http://localhost:8001/health
curl -sS http://localhost:8001/generate -X POST -H 'Content-Type: application/json' -d '{"query":"test","topic":"demo"}'
```

Notes:

- `ai-api` image now includes `kbd` dependencies, so TinyDB cache works in
  containers.
- Persistent cache file is stored under `/app/data/generation_cache.json`
  through the `./data:/app/data` volume.
- Start with Qwen 3B quantized model on CPU-only hosts, then scale resources
  and context size conservatively.

### Linux VPS (GPU) optimized profile

GPU deployments use the same centralized compose file with `gpu` profile.

1. Start stack using a distribution env file + GPU profile:

```bash
docker compose \
  --env-file src/distributions/pro/vps-gpu.env \
  --profile gpu \
  -f src/docker-compose.yml \
  up -d --build
```

3. Validate deployment:

```bash
curl -sS http://localhost:8000/health
curl -sS http://localhost:8001/health
```

The GPU profile uses `LLAMA_GPU_IMAGE` for `llama-server` and enables
`gpus: all`. The default pinned value is currently
`ghcr.io/ggerganov/llama.cpp:server-cuda-b4719`, because the registry no longer
publishes a stable `server-cuda` floating tag.

### Installation scripts by environment

You can run one command per target environment:

- Linux VPS (CPU): `src/scripts/install/install_vps_linux_cpu.sh`
- Linux VPS (GPU): `src/scripts/install/install_vps_linux_gpu.sh`
- Windows (Docker Desktop CPU): `src/scripts/install/install_windows.ps1 -Stage <stage> -Environment windows`
- Windows (Docker Desktop GPU): `src/scripts/install/install_windows.ps1 -Stage <stage> -Environment windows-gpu`

These wrappers now map to the distribution matrix system.

Examples with stage:

```bash
./src/scripts/install/install_vps_linux_cpu.sh dev
./src/scripts/install/install_vps_linux_gpu.sh pro
```

```powershell
./src/scripts/install/install_windows.ps1 -Stage stg -Environment windows
./src/scripts/install/install_windows.ps1 -Stage stg -Environment windows-gpu
```

Examples:

```bash
# Linux CPU
./src/scripts/install/install_vps_linux_cpu.sh

# Linux GPU
./src/scripts/install/install_vps_linux_gpu.sh
```

```powershell
# Windows CPU (defaults LLAMA port to 18080)
./src/scripts/install/install_windows.ps1

# Windows GPU workstation
./src/scripts/install/install_windows.ps1 -Stage stg -Environment windows-gpu
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `AI_ENGINE_MODELS_DIR` | `<project_root>/src/models/` | Path where GGUF model files are stored. Override to point to a shared network drive or Docker volume. |
| `LLAMA_MODEL_FILE` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | GGUF filename loaded by the `llama-server` compose service. |
| `LLAMA_CTX_SIZE` | `4096` | Context window size (tokens) for the llama.cpp server. |
| `LLAMA_PORT` | `8080` | Legacy variable kept in env templates; the centralized compose currently publishes `7002:8080`. |
| `LLAMA_CPUS` | `6.0` | CPU quota assigned to `llama-server` by Docker Compose. |
| `LLAMA_MEMORY_LIMIT` | `12g` | Hard memory limit for `llama-server`. |
| `LLAMA_MEMORY_RESERVATION` | `8g` | Soft memory reservation for `llama-server`. |
| `LLAMA_SHM_SIZE` | `2g` | Shared memory available to `llama-server`. |
| `STATS_PORT` | `8000` | Legacy variable kept in env templates; the centralized compose currently publishes `7000:8000`. |
| `STATS_CPUS` | `1.0` | CPU quota assigned to `ai-stats`. |
| `STATS_MEMORY_LIMIT` | `1g` | Hard memory limit for `ai-stats`. |
| `STATS_MEMORY_RESERVATION` | `512m` | Soft memory reservation for `ai-stats`. |
| `API_PORT` | `8001` | Legacy variable kept in env templates; the centralized compose currently publishes `7001:8001`. |
| `API_CPUS` | `4.0` | CPU quota assigned to `ai-api`. |
| `API_MEMORY_LIMIT` | `8g` | Hard memory limit for `ai-api`. |
| `API_MEMORY_RESERVATION` | `4g` | Soft memory reservation for `ai-api`. |
| `API_SHM_SIZE` | `1g` | Shared memory available to `ai-api`. |
| `API_WORKERS` | `2` | Number of Uvicorn workers used by `ai-api`. |
| `STATS_WORKERS` | `1` | Number of Uvicorn workers used by `ai-stats`. |
| `AI_ENGINE_STATS_URL` | `http://ai-stats:8000` | Base URL used by `ai-api` to push observability events into `ai-stats` (`POST /events`). |
| `AI_ENGINE_GAMES_API_KEY` | *(unset)* | API key required on generation routes (`/generate*`) for game microservices. |
| `AI_ENGINE_BRIDGE_API_KEY` | *(unset)* | API key required on bridge-facing routes (`/ingest*`, `/stats*`, `/cache/*`, `/metrics`). |
| `AI_ENGINE_STATS_API_KEY` | *(unset)* | API key used by `ai-api` to publish events to `ai-stats` (`POST /events`). |
| `AI_ENGINE_GENERATION_API_URL` | `http://ai-api:8001` | Internal URL used by `ai-stats` to query cache monitoring/reset endpoints from `ai-api`. |
| `AI_ENGINE_LLAMA_MAX_CONCURRENT_REQUESTS` | `1` | Maximum concurrent llama upstream requests admitted by `ai-api`. |
| `AI_ENGINE_GENERATION_MAX_IN_FLIGHT` | `1` | Maximum active generation requests allowed by `ai-api` admission control. |
| `AI_ENGINE_GENERATION_MAX_QUEUE_SIZE` | `1` | Maximum queued generation requests allowed before `503 busy`. |
| `AI_ENGINE_CACHE_WARMUP_ENABLED` | `false` | Enables or disables startup cache warmup for `ai-api`. |
| `AI_ENGINE_GENERATION_CACHE_BACKEND` | `redis` | Cache backend used by `ai-api` (`redis` recommended for persistent containerized deployments). |
| `AI_ENGINE_GENERATION_CACHE_REDIS_URL` | `redis://ai-cache:6379/0` | Redis connection string used by `ai-api` cache layer. |
| `AI_ENGINE_API_KEY` | *(unset)* | Global fallback API key kept for backward compatibility. |
| `APP_NAME` | `AxiomNode` | Base application name used in image/container naming. |
| `API_SERVICE_NAME` | `ai_api` | Service name segment for `ai-api` image/container naming. |
| `STATS_SERVICE_NAME` | `ai_stats` | Service name segment for `ai-stats` image/container naming. |
| `LLAMA_CPU_SERVICE_NAME` | `llama_server_cpu` | Service name segment for CPU llama service naming. |
| `LLAMA_GPU_SERVICE_NAME` | `llama_server_gpu` | Service name segment for GPU llama service naming. |
| `AI_ENGINE_DISTRIBUTION` | `dev` | Distribution suffix used in observability and image/container naming. |
| `AI_ENGINE_RELEASE_VERSION` | `v1` | Release-version suffix used in observability and image/container naming. |

### Windows note: Docker Desktop resource ceiling

If you use Docker Desktop on Windows, Compose cannot exceed the global limits
assigned to the Linux backend.

1. Open Docker Desktop.
2. Go to Settings.
3. If you use the Hyper-V backend, open Resources and raise CPUs, Memory, and Swap.
4. If you use WSL 2, raise the limits for the `docker-desktop` WSL environment or remove restrictive values from your `.wslconfig`.
5. Restart Docker Desktop.

For the `windows-gpu` workstation profile added for RTX 4080 SUPER / Ryzen 7950X class hosts, start with at least:

- 20 vCPUs available to Docker Desktop
- 40 GB RAM available to Docker Desktop / WSL2
- GPU sharing enabled for the Linux backend

That leaves headroom on a 32-thread / 64 GB workstation while still allowing the stack to use roughly 60% of host capacity.

After that, run:

```bash
docker compose -f src/docker-compose.yml --profile gpu down
docker compose --env-file src/distributions/stg/windows-gpu.env -f src/docker-compose.yml --profile gpu up -d --build
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
