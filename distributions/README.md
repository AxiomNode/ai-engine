# Deployment Distributions Matrix

This folder defines deployment distributions by:

- stage: `dev`, `stg`, `pro`
- environment: `windows`, `vps-cpu`, `vps-gpu`

Each combination is a dedicated env file:

- `distributions/<stage>/<environment>.env`

Examples:

- `distributions/dev/windows.env`
- `distributions/stg/vps-cpu.env`
- `distributions/pro/vps-gpu.env`

## Usage

Use the unified installers under `scripts/install/`:

- Linux: `scripts/install/deploy.sh <stage> <environment>`
- Windows: `scripts/install/deploy.ps1 -Stage <stage> -Environment <environment>`

The installers automatically map environment to a centralized compose profile:

- `windows` -> `cpu`
- `vps-cpu` -> `cpu`
- `vps-gpu` -> `gpu`

So all deployments run from a single `docker-compose.yml`.

Valid values:

- stage: `dev|stg|pro`
- environment: `windows|vps-cpu|vps-gpu`

Each env file should define a deployment tag pair used by logs, metrics,
and monitoring endpoints:

- `AI_ENGINE_DISTRIBUTION` (for example: `dev`, `stg`, `pro`)
- `AI_ENGINE_RELEASE_VERSION` (for example: `v1`, `2026.03.16`)

Runtime systems expose the combined label as:

- `distribution-version` = `<AI_ENGINE_DISTRIBUTION>-<AI_ENGINE_RELEASE_VERSION>`
