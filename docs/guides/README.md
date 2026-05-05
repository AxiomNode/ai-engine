# Guides

Last updated: 2026-05-03.

## Purpose

This section groups practical usage guides.

## Scope

Use these guides for concrete local development, integration, contract, and observability tasks.

The main distinction is:

- usage guides explain how to work with ai-engine locally or as a consumer
- architecture docs explain why the repository is shaped the way it is
- operations docs explain how to deploy, recover, and observe runtime slices

## Navigation

- [getting-started.md](getting-started.md) — Local setup and development workflow.
- [rag-usage.md](rag-usage.md) — RAG ingestion/retrieval usage.
- [rag-corpus-authoring.md](rag-corpus-authoring.md) — How to create curated RAG datasets and audit corpus coverage.
- [kbd-usage.md](kbd-usage.md) — Knowledge Base usage and integration.
- [sdk-usage.md](sdk-usage.md) — SDK response models and parsing.
- [metrics-usage.md](metrics-usage.md) — Metrics and observability usage patterns.
- [microservice-contract-games.md](microservice-contract-games.md) — Connection contract for game microservices that consume generation endpoints.
- [microservice-contract-bridge.md](microservice-contract-bridge.md) — Connection contract for the bridge microservice handling ingestion and observability.

## Reading order

1. Start with [getting-started.md](getting-started.md) for local development.
2. Use [rag-usage.md](rag-usage.md), [kbd-usage.md](kbd-usage.md), and [sdk-usage.md](sdk-usage.md) for feature-level work.
3. Use the contract guides when changing consumer integrations.
4. Use [metrics-usage.md](metrics-usage.md) when validating observability behavior.
