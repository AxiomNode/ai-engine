# Resource Allocation — Windows GPU Workstation (RTX 4080 SUPER / Ryzen 9 7950X / 64 GB RAM)

Recommended starting profile for running **ai-engine** on a dedicated workstation
through Docker Desktop + WSL2, while keeping enough headroom for the host OS.

---

## 1. Hardware Target

| Resource | Value |
|----------|-------|
| CPU | 16 cores / 32 threads |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM) |
| Runtime | Docker Desktop + WSL2 GPU passthrough |

This profile is intended for **manual, on-demand generation sessions** where the
AI engine can use a substantial part of the workstation, but should not consume
the entire machine.

---

## 2. Recommended Global Docker / WSL Limits

Start with:

| Limit | Recommended value |
|-------|-------------------|
| WSL / Docker CPUs | 20 |
| WSL / Docker Memory | 40 GB |
| WSL Swap | 8 GB |

This keeps roughly 35-40% of host capacity free for Windows, browsers, IDEs,
and other development tools.

---

## 3. ai-engine Container Budget

These values correspond to the tracked `windows-gpu` distribution files.

| Service | CPU | RAM limit | RAM reserv. | Notes |
|---------|-----|-----------|-------------|-------|
| llama-server-gpu | 10-12 | 22-24 GB | 16-18 GB | Main inference workload. Full GPU offload (`LLAMA_N_GPU_LAYERS=99`) with CPU headroom for prompt handling and batching. |
| ai-api | 5-6 | 10-12 GB | 8 GB | FastAPI + embeddings + RAG orchestration. Higher RAM keeps sentence-transformers, vector store, and request bursts stable. |
| ai-stats | 1-1.5 | 1-1.5 GB | 512-768 MB | Lightweight metrics and observability surface. |
| ai-cache | 0.5 | 512 MB | 256 MB | Redis generation cache. |

Approximate subtotal:

- CPU: 16.5 to 20 cores
- RAM: 33.5 to 38 GB

---

## 4. Inference Tuning Rationale

| Parameter | Starting value | Why |
|-----------|----------------|-----|
| `LLAMA_MODEL_FILE` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Better content quality than 3B while still fitting comfortably in 16 GB VRAM. |
| `LLAMA_CTX_SIZE` | `4096` (stg) / `8192` (pro) | 4096 is a safe starting point for structured generation. 8192 is reasonable once stability is confirmed. |
| `LLAMA_THREADS` | `12` | Enough CPU support threads for prefill and batching without monopolizing the full 7950X. |
| `LLAMA_BATCH` | `1536`-`2048` | Higher batch improves prompt prefill throughput on a strong workstation. |
| `LLAMA_PARALLEL` | `2` | Good balance for a single 4080 SUPER. More slots usually increase contention before they improve useful throughput. |
| `AI_ENGINE_LLAMA_MAX_CONCURRENT_REQUESTS` | `2` | Keep ai-api aligned with the llama server parallelism. |
| `AI_ENGINE_GENERATION_MAX_IN_FLIGHT` | `4` | Allows controlled overlap in ai-api without immediately returning busy. |
| `AI_ENGINE_GENERATION_MAX_QUEUE_SIZE` | `8` | Short queue for burst absorption during manual generation sessions. |

---

## 5. Scaling Rules for This Workstation

Increase gradually in this order:

1. `AI_ENGINE_GENERATION_MAX_IN_FLIGHT` only after confirming stable latency.
2. `LLAMA_BATCH` only if VRAM remains comfortable and prompt prefill is the bottleneck.
3. `LLAMA_CTX_SIZE` only when prompts genuinely need more context.
4. `LLAMA_PARALLEL` only after measuring real throughput; do not assume higher is better on a single 16 GB GPU.

Avoid increasing all knobs together. On this class of machine, the most common
failure mode is not raw OOM but **latency inflation due to excessive parallelism**.

---

## 6. Preflight Before First Run

From `ai-engine/src`:

```powershell
./scripts/install/preflight_windows_gpu.ps1
```

Then start the stack:

```powershell
./scripts/install/install_windows.ps1 -Stage stg -Environment windows-gpu
```

The Windows deploy script now supports two exposure modes from the tracked
distribution env file:

- direct Windows host exposure with firewall + `netsh interface portproxy`
- VPS relay exposure for the staging topology where ai-engine runs on this PC
	and the rest of the platform runs on the VPS Kubernetes cluster

For staging, the tracked `windows-gpu` profile now prefers the VPS relay mode.
It opens an SSH reverse tunnel from the workstation to the VPS and publishes
stable TCP relay ports on the VPS, so cluster services can reach ai-engine
without depending on this workstation's router/NAT.

For the tracked STG workstation profile, this means:

- `27000 -> 7000` for `ai-stats`
- `27001 -> 7001` for `ai-api`

The recommended STG target host is the VPS relay host IP, not the public
gateway domain, because the gateway domain may sit behind an HTTP proxy that
does not pass arbitrary TCP ports. In the tracked profile this host is
`195.35.48.40`.

If you deliberately choose direct Windows exposure instead of the VPS relay,
run the script from an elevated PowerShell session so those host-level rules
can be applied.

---

## 7. First Validation Checks

After startup, confirm:

```bash
curl -sS http://localhost:7000/health
curl -sS http://localhost:7001/health
docker ps
docker stats
```

For the VPS relay mode, validate from the VPS relay host:

```bash
curl -sS http://195.35.48.40:27000/health
curl -sS http://195.35.48.40:27001/health
```

During real generation runs, also watch:

```bash
nvidia-smi
docker logs ai_engine_llama_gpu_stg_workstation-gpu
```

If GPU memory is fine but latency remains high, reduce API concurrency before
raising llama parallelism.