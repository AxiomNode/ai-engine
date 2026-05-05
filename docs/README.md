# docs/

Last updated: 2026-05-03.

## Purpose

This folder contains the technical documentation for **ai-engine**.

## Navigation

| Document | Description |
|---|---|
| [architecture/README.md](architecture/README.md) | Architecture index: context, design, technologies, ADRs, repository layout |
| [../src/distributions/README.md](../src/distributions/README.md) | Deployment matrix by stage (dev/stg/pro) and environment (windows/windows-gpu/vps-cpu/vps-gpu) |
| [guides/README.md](guides/README.md) | Guides index: getting started, RAG, KBD, SDK, and metrics usage |
| [guides/microservice-contract-games.md](guides/microservice-contract-games.md) | Shareable connection contract for game microservices (generation integration) |
| [guides/microservice-contract-bridge.md](guides/microservice-contract-bridge.md) | Shareable connection contract for bridge microservice (ingestion + observability) |
| [operations/README.md](operations/README.md) | Operational index: deployment runbook, incident playbook, and distribution matrix |

## Reading order

1. Start with [architecture/README.md](architecture/README.md).
2. Continue with [guides/README.md](guides/README.md) for local development and consumer integration.
3. Use [operations/README.md](operations/README.md) for deployment, incident handling, and runtime maintenance.

## When to use this

- when the central platform docs are too broad for an `ai-engine`-specific change
- when you need the repository-local navigation entry for architecture, guides, and operations
