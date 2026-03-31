# Resource Allocation — VPS 8 cores / 32 GB RAM

Baseline resource distribution for the minimum production VPS target.
All values are configurable via the corresponding `.env` file in
`src/distributions/<env>/vps-cpu.env`.

---

## 1. Hardware Target

| Resource | Value |
|----------|-------|
| CPU | 8 physical cores |
| RAM | 32 GB |
| GPU | None (CPU-only inference) |
| Disk | SSD (recommended for model loading) |

This machine hosts **all** AxiomNode services simultaneously:
ai-engine (4 containers), 3 microservices + 3 PostgreSQL DBs,
API gateway, 2 BFF proxies, and a static frontend.

---

## 2. ai-engine Services

| Service | CPU | RAM limit | RAM reserv. | Justification |
|---------|-----|-----------|-------------|---------------|
| **llama-server** | 4.0 | 10 GB | 6 GB | Heaviest workload: matrix multiplications for token generation. 4 threads = half the cores, leaving room for ai-api embedding work. 10 GB covers Qwen2.5-3B Q4_K_M (~2 GB weights) + KV cache (2 slots × 2048 ctx) + OS buffers. |
| **ai-api** | 1.5 | 4 GB | 2 GB | Runs FastAPI + sentence-transformers (all-MiniLM-L6-v2, ~90 MB). 2 Uvicorn workers handle concurrent requests while llama-server does the heavy lifting. 4 GB headroom for RAG vector store in memory. |
| **ai-stats** | 0.5 | 512 MB | 256 MB | Lightweight FastAPI collecting observability events. Minimal CPU — only JSON serialization and Redis writes. |
| **ai-cache** (Redis) | 0.25 | 256 MB | 128 MB | In-memory cache with LRU eviction at 128 MB. Stores serialized game JSON (~2-5 KB each). 128 MB ≈ 25,000+ cached games — more than enough. |

**Subtotal ai-engine: 6.25 CPU / ~15 GB RAM**

---

## 3. Other AxiomNode Services

| Service | CPU | RAM limit | Notes |
|---------|-----|-----------|-------|
| microservice-quiz | 0.5 | 512 MB | Node.js + Prisma ORM. Low CPU — mostly DB queries. |
| quiz-db (Postgres) | 0.25 | 256 MB | Small dataset, mostly reads. Shared buffers 64 MB. |
| microservice-wordpass | 0.5 | 512 MB | Same pattern as quiz. |
| wordpass-db (Postgres) | 0.25 | 256 MB | Same pattern as quiz-db. |
| microservice-users | 0.25 | 256 MB | Auth + profile CRUD. Light traffic. |
| users-db (Postgres) | 0.25 | 256 MB | User records — small dataset. |
| api-gateway | 0.25 | 128 MB | Reverse proxy routing. Negligible CPU. |
| bff-mobile | 0.25 | 128 MB | Thin proxy aggregating backend calls. |
| bff-backoffice | 0.25 | 128 MB | Same as bff-mobile. |

**Subtotal other services: ~2.75 CPU / ~2.4 GB RAM**

---

## 4. Total Budget

| | CPU (cores) | RAM |
|---|---|---|
| ai-engine | 6.25 | ~15.0 GB |
| Other services | ~2.75 | ~2.4 GB |
| **Allocated** | **~9.0** | **~17.4 GB** |
| **OS + buffers** | — | **~14.6 GB** |

> CPU overcommit is ~1 core (9/8). This is acceptable because llama-server
> rarely saturates all 4 cores during token-by-token generation (it bursts
> during prompt prefill only), and microservices are mostly idle between
> requests.

---

## 5. llama.cpp Inference Tuning

These parameters have the biggest impact on generation latency:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `LLAMA_CTX_SIZE` | 2048 | Context window. Game prompts use ~1200 input tokens + 512 output. Reducing from 4096 halves KV-cache memory per slot. |
| `LLAMA_THREADS` | 4 | Threads for matrix ops. Matches CPU allocation. More threads than physical cores hurts performance due to context switching. |
| `LLAMA_BATCH` | 1024 (pro) | Prompt prefill batch size. Higher = fewer iterations to process the input prompt. 1024 is optimal for 4 threads. |
| `LLAMA_PARALLEL` | 2 | Concurrent request slots. Each slot has its own KV cache. 2 slots allow the warm-up task and user requests to run simultaneously. |
| `--cont-batching` | always | Continuous batching lets new requests start while others are generating, improving throughput. |
| `--mlock` | always | Pins model weights in RAM. Prevents the OS from paging model data to disk under memory pressure. Critical for consistent latency. |

### Why native C++ instead of llama-cpp-python?

The official `ghcr.io/ggerganov/llama.cpp:server` image runs the llama.cpp
HTTP server compiled directly from C++. The previous setup used
`llama-cpp-python`, which wraps the same C++ code through Python bindings,
adding ~30% overhead from the GIL, memory copies, and Python event loop.

The native server:
- Exposes the same OpenAI-compatible `/v1/completions` endpoint
- Supports `--parallel` and `--cont-batching` natively
- Has lower memory footprint (no Python runtime)
- Starts faster (~5s vs ~15s)

---

## 6. Scaling Guide

When the VPS gets more resources, adjust these knobs in order:

| Priority | Action | When |
|----------|--------|------|
| 1 | `LLAMA_THREADS` 4 → 6 | 12+ cores available |
| 2 | `LLAMA_PARALLEL` 2 → 4 | High concurrent traffic |
| 3 | `LLAMA_BATCH` 1024 → 2048 | More RAM headroom |
| 4 | `LLAMA_MODEL_FILE` → 7B | 48+ GB RAM, need better quality |
| 5 | `LLAMA_CTX_SIZE` 2048 → 4096 | Longer prompts needed |
| 6 | `API_WORKERS` 2 → 4 | API becomes bottleneck |

### Switching to 7B model

```bash
# In the .env file:
LLAMA_MODEL_FILE=Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLAMA_MEMORY_LIMIT=18g    # 7B Q4 needs ~4.5 GB weights + KV cache
LLAMA_THREADS=6            # more layers = benefits from more threads
```

> The 7B model produces noticeably better content quality but doubles
> inference time on CPU. Only switch when latency is acceptable for
> the use case or when GPU offload is available.
