# VPS Autodeploy

This runbook prepares a single VPS to keep `ai-engine` updated with Docker
Compose and systemd. Use it when `ai-engine-api`, `ai-engine-stats`, Redis, and
llama.cpp run directly on the VPS outside the Kubernetes rollout path.

There are two deploy modes:

| Mode | What the timer does | Use |
| --- | --- | --- |
| `image` (default for new installs) | `docker compose pull` + `up -d --no-build` of the pinned GHCR images from `src/.env.images`. No `git pull`, no build. | Recommended. The host also runs llama.cpp, so it should never compile or install torch. |
| `build` | `git pull --ff-only` + `docker compose up -d --build` on the host. | Legacy fallback, see [Fallback: build on the host](#fallback-build-on-the-host). |

Existing hosts keep running in `build` mode until someone re-runs the installer
by hand: merging changes to the scripts or compose files does not switch a host.

For the platform k3s staging/prod path, image build and rollout are still owned
by `platform-infra`.

## Prerequisites

- Linux VPS with systemd.
- Docker Engine with the Docker Compose plugin.
- Git checkout of `ai-engine` on the VPS (needed for the compose files and scripts; in
  `image` mode it is updated by hand, see below).
- `image` mode: read access to the GHCR packages `ghcr.io/axiomnode/ai-engine-api` and
  `ai-engine-stats` (`docker login ghcr.io` with a token that has `read:packages`, if the
  packages are private) and the pin file `src/.env.images` (next section).
- Runtime secrets prepared in `src/.env.secrets` or a sibling `secrets` repo
  available at `../secrets`.
- Required GGUF model file under `src/models/`.

Example model download from `src/`:

```bash
python -m ai_engine.llm.model_manager download qwen2.5-7b
```

## Pin The Images (image mode)

CI (`platform-infra` `build-push.yaml`) publishes both images on every build:

- `ghcr.io/axiomnode/ai-engine-api:<8-char tag>` (also `:dev` and, on `main`, `:stg`)
- `ghcr.io/axiomnode/ai-engine-stats:<8-char tag>` (same tags)

The 8-char tag is the first 8 characters of the commit SHA of the `platform-infra` run that
built the image, not an `ai-engine` commit; look it up in that run, or resolve a digest with
`docker buildx imagetools inspect ghcr.io/axiomnode/ai-engine-api:stg`.

```bash
cp src/distributions/examples/images.env.example src/.env.images
# edit src/.env.images: set AI_ENGINE_API_IMAGE and AI_ENGINE_STATS_IMAGE
```

Prefer a digest (`...@sha256:<digest>`) or a tag over `:stg`: a mutable tag makes every pull
move to whatever CI built last. `src/.env.images` is git-ignored. To roll forward or back,
edit the two lines and either wait for the timer or start the service by hand.

## Install Timer

Installing the timer is a manual step on the VPS; nothing here runs from CI or from a merge.
From the `ai-engine` repository root on the VPS:

```bash
./src/scripts/install/install-vps-autodeploy.sh pro vps-cpu main 15min image
```

Arguments:

- `stage`: `stg` or `pro`
- `environment`: `vps-cpu` or `vps-gpu`
- `branch`: defaults to `main` (only used in `build` mode)
- `interval`: systemd interval, defaults to `15min` in `image` mode and `5min` in `build` mode
- `mode`: `image` (default) or `build`

Re-running the installer replaces the runner script and unit files, so it is also how an
existing host switches modes.

The installer creates:

- `/usr/local/bin/axiomnode-ai-engine-autodeploy`
- `/etc/systemd/system/axiomnode-ai-engine-autodeploy.service`
- `/etc/systemd/system/axiomnode-ai-engine-autodeploy.timer`

## Runtime Behavior

Each run (`image` mode):

1. Acquires a lock so deploys cannot overlap.
2. Prepares secrets from `../secrets` when that repo exists.
3. Runs `src/scripts/install/deploy.sh <stage> <environment>` with
   `AI_ENGINE_DEPLOY_MODE=image`: validates the compose config, pulls the images pinned in
   `src/.env.images`, then `docker compose up -d --no-build`. Compose only recreates a
   container whose image or configuration changed, so with unchanged pins a run is a no-op
   that also restarts anything that died.
4. Waits for `ai-stats` and `ai-api` health on the ports defined by the
   distribution env file.
5. Polls `GET /diagnostics/rag/stats` and fails the deploy if the RAG index has
  no chunks or still reports empty coverage.

In `image` mode the checkout is not touched. When compose files or scripts change
(for example `docker-compose.yml`, `docker-compose.ghcr.yml`, `distributions/`), update it
by hand and redeploy:

```bash
git pull --ff-only && sudo systemctl start axiomnode-ai-engine-autodeploy.service
```

`git pull --ff-only` stops on local uncommitted changes or divergent history instead of
overwriting them.

## Fallback: build on the host

`build` mode is the previous behavior: each run does `git fetch`, `git checkout <branch>`,
`git pull --ff-only` and `docker compose up -d --build`, so a push to the branch is deployed
within the interval, but the image is compiled on the VPS (torch and sentence-transformers
layers are rebuilt there whenever `pyproject.toml` or `ai_engine/` changes), competing with
llama.cpp. To use it:

```bash
./src/scripts/install/install-vps-autodeploy.sh pro vps-cpu main 5min build
```

or one-off, without the timer: `./src/scripts/install/deploy.sh pro vps-cpu` (the default
`AI_ENGINE_DEPLOY_MODE=build`).

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

- One Uvicorn worker for `ai-api` (`API_WORKERS=1`). Each extra worker loads its own
  embedding model (estimate: 0.5-1 GB RAM) and re-embeds the corpus at boot, and the rate
  limiter and capacity gate are per process; see
  [Uvicorn workers](../../src/distributions/README.md#uvicorn-workers).

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