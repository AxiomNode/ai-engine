# Operations Guide

This section groups operational runbooks and deployment-oriented references.

## Contents

- [deployment.md](deployment.md) — Build and run the stack with Docker and distribution matrix.
- [resource-allocation.md](resource-allocation.md) — CPU/RAM distribution per service for VPS 8c/32GB with justification.
- [incident-runbook.md](incident-runbook.md) — Incident response and rollback playbooks.
- [improvement-history.md](improvement-history.md) — Chronological history of completed improvements.
- [improvement-checklist.md](improvement-checklist.md) — Next-cycle checklist by phase and priority.
- [../../src/distributions/README.md](../../src/distributions/README.md) — Distribution matrix (`dev|stg|pro` x `windows|vps-cpu|vps-gpu`).

## Quick Start

```bash
cp src/distributions/examples/.env.example src/.env
./src/scripts/install/deploy.sh dev windows
```

```powershell
./src/scripts/install/deploy.ps1 -Stage dev -Environment windows
```

## Notes

- The deployment flow is centralized in `src/docker-compose.yml`.
- Compose profiles are auto-selected by install scripts:
  - `windows` and `vps-cpu` map to `cpu`.
  - `vps-gpu` maps to `gpu`.
- Built image/container names follow:
  - `<APP_NAME>_<SERVICE_NAME>_<AI_ENGINE_DISTRIBUTION>_<AI_ENGINE_RELEASE_VERSION>`.
