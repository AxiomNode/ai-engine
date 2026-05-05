# VPS Autodeploy

This runbook prepares a single VPS to keep `ai-engine` updated with Docker
Compose and systemd. Use it when `ai-engine-api`, `ai-engine-stats`, Redis, and
llama.cpp run directly on the VPS outside the Kubernetes rollout path.

For the platform k3s staging/prod path, image build and rollout are still owned
by `platform-infra`.

## Prerequisites

- Linux VPS with systemd.
- Docker Engine with the Docker Compose plugin.
- Git checkout of `ai-engine` on the VPS.
- Runtime secrets prepared in `src/.env.secrets` or a sibling `secrets` repo
  available at `../secrets`.
- Required GGUF model file under `src/models/`.

Example model download from `src/`:

```bash
python -m ai_engine.llm.model_manager download qwen2.5-7b
```

## Install Timer

From the `ai-engine` repository root on the VPS:

```bash
./src/scripts/install/install-vps-autodeploy.sh pro vps-cpu main 5min
```

Arguments:

- `stage`: `stg` or `pro`
- `environment`: `vps-cpu` or `vps-gpu`
- `branch`: defaults to `main`
- `interval`: systemd interval, defaults to `5min`

The installer creates:

- `/usr/local/bin/axiomnode-ai-engine-autodeploy`
- `/etc/systemd/system/axiomnode-ai-engine-autodeploy.service`
- `/etc/systemd/system/axiomnode-ai-engine-autodeploy.timer`

## Runtime Behavior

Each run:

1. Acquires a lock so deploys cannot overlap.
2. Fetches and fast-forwards the selected branch.
3. Prepares secrets from `../secrets` when that repo exists.
4. Runs `src/scripts/install/deploy.sh <stage> <environment>`.
5. Waits for `ai-stats` and `ai-api` health on the ports defined by the
   distribution env file.
6. Polls `GET /diagnostics/rag/stats` and fails the deploy if the RAG index has
  no chunks or still reports empty coverage.

The git update uses `git pull --ff-only`. Local uncommitted changes or divergent
history stop the rollout instead of being overwritten.

## Operations

Run immediately:

```bash
sudo systemctl start axiomnode-ai-engine-autodeploy.service
```

Inspect timer:

```bash
systemctl list-timers axiomnode-ai-engine-autodeploy.timer
```

Follow logs:

```bash
journalctl -u axiomnode-ai-engine-autodeploy.service -f
```

Disable autodeploy:

```bash
sudo systemctl disable --now axiomnode-ai-engine-autodeploy.timer
```

## VPS Runtime Defaults

The VPS distribution files use:

- Redis generation cache.
- Chroma persistent vector store at `/app/data/chroma` mounted from `src/data`.
- Idempotent Chroma upserts so restart or redeploy can re-prime the curated RAG
  corpus without duplicate-id failures.
- English-only curated RAG corpus packaged with the application image.

## Verify RAG On The VPS

After a manual deploy or timer run, inspect the service logs:

```bash
journalctl -u axiomnode-ai-engine-autodeploy.service -n 100 --no-pager
```

A successful run ends with a line similar to:

```text
RAG ready at http://localhost:7001/diagnostics/rag/stats: chunks=94, coverage=moderate
```

To query the API directly, include the API key when production secrets are
enabled:

```bash
curl -fsS -H "X-API-Key: ${AI_ENGINE_API_KEY}" \
  http://localhost:7001/diagnostics/rag/stats
```