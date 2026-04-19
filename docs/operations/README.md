# Operations Guide

This section groups operational runbooks and deployment-oriented references.

## Contents

- [deployment.md](deployment.md) — Build and run the stack with Docker and distribution matrix.
- [resource-allocation.md](resource-allocation.md) — CPU/RAM distribution per service for VPS 8c/32GB with justification.
- [resource-allocation-workstation-gpu.md](resource-allocation-workstation-gpu.md) — Starting CPU/RAM/GPU budget for RTX 4080 SUPER / 7950X class Windows workstations.
- [incident-runbook.md](incident-runbook.md) — Incident response and rollback playbooks.
- [improvement-history.md](improvement-history.md) — Chronological history of completed improvements.
- [improvement-checklist.md](improvement-checklist.md) — Next-cycle checklist by phase and priority.
- [../../src/distributions/README.md](../../src/distributions/README.md) — Distribution matrix (`dev|stg|pro` x `windows|windows-gpu|vps-cpu|vps-gpu`).

## Quick Start

```bash
cp src/distributions/examples/.env.example src/.env
./src/scripts/install/deploy.sh dev windows
```

```powershell
./src/scripts/install/deploy.ps1 -Stage dev -Environment windows
./src/scripts/install/deploy.ps1 -Stage stg -Environment windows-gpu
```

## Notes

- The deployment flow is centralized in `src/docker-compose.yml`.
- Compose profiles are auto-selected by install scripts:
  - `windows` and `vps-cpu` map to `cpu`.
  - `windows-gpu` and `vps-gpu` map to `gpu`.
- Built image/container names follow:
  - `<APP_NAME>_<SERVICE_NAME>_<AI_ENGINE_DISTRIBUTION>_<AI_ENGINE_RELEASE_VERSION>`.
