# Deployment Distributions Matrix

This folder defines deployment distributions by:

- stage: `dev`, `stg`, `pro`
- environment: `windows`, `windows-gpu`, `vps-cpu`, `vps-gpu`

Each combination is a dedicated env file (from repository root):

- `src/distributions/<stage>/<environment>.env`

These tracked files must contain only deployment shape and capacity settings.
Real secrets are injected separately into `src/.env.secrets` from the private `secrets` repository.

General local template:

- `src/distributions/examples/.env.example`

Examples:

- `src/distributions/dev/windows.env`
- `src/distributions/stg/vps-cpu.env`
- `src/distributions/pro/vps-gpu.env`

## Usage

Use the unified installers under `src/scripts/install/`:

- Linux: `src/scripts/install/deploy.sh <stage> <environment>`
- Windows: `src/scripts/install/deploy.ps1 -Stage <stage> -Environment <environment>`

Before running installers, inject the matching runtime secrets:

- `node ../secrets/scripts/prepare-runtime-secrets.mjs dev ai-engine`
- `node ../secrets/scripts/prepare-runtime-secrets.mjs stg ai-engine`
- `node ../secrets/scripts/prepare-runtime-secrets.mjs pro ai-engine`

The installers automatically map environment to a centralized compose profile:

- `windows` -> `cpu`
- `windows-gpu` -> `gpu`
- `vps-cpu` -> `cpu`
- `vps-gpu` -> `gpu`

So all deployments run from a single `src/docker-compose.yml`.

Installers now load two env files in order:

1. `src/distributions/<stage>/<environment>.env` for resource sizing and stage metadata
2. `src/.env.secrets` for API keys and other private runtime values

Valid values:

- stage: `dev|stg|pro`
- environment: `windows|windows-gpu|vps-cpu|vps-gpu`

Each env file should define a deployment tag pair used by logs, metrics,
and monitoring endpoints:

- `AI_ENGINE_DISTRIBUTION` (for example: `dev`, `stg`, `pro`)
- `AI_ENGINE_RELEASE_VERSION` (for example: `v1`, `2026.03.16`)

Runtime systems expose the combined label as:

- `distribution-version` = `<AI_ENGINE_DISTRIBUTION>-<AI_ENGINE_RELEASE_VERSION>`

## Resource Tiers by Stage

Distributions now follow a fixed capacity strategy:

- `dev`: minimum resources for fast local validation and functional tests.
- `stg`: medium resources, closer to production behavior without full cost.
- `pro`: recommended resources for stable production workloads.

Primary knobs adjusted per env file:

- model profile (`LLAMA_MODEL_FILE`, `LLAMA_CTX_SIZE`, `LLAMA_BATCH`)
- container capacity (`LLAMA_*`, `API_*`, `STATS_*` CPU/RAM)
- processing headroom (`API_WORKERS`)
- generation budget (`AI_ENGINE_LLAMA_TIMEOUT_SECONDS`)

## Docker Image and Container Naming

Compose uses the naming pattern below for built images and container names:

- `<APP_NAME>_<SERVICE_NAME>_<AI_ENGINE_DISTRIBUTION>_<AI_ENGINE_RELEASE_VERSION>`

Default example:

- `AxiomNode_ai_api_dev_v1`

Required naming variables are now included in each distribution env file:

- `APP_NAME`
- `API_SERVICE_NAME`
- `STATS_SERVICE_NAME`
- `LLAMA_CPU_SERVICE_NAME`
- `LLAMA_GPU_SERVICE_NAME`
