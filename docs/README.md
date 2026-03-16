# docs/

This folder contains the technical documentation for **ai-engine**.

| Document | Description |
|---|---|
| [project-context.md](project-context.md) | AxiomNode vision, high-level architecture, and ai-engine scope |
| [architecture.md](architecture.md) | System design: layers, components, data flows, and extension points |
| [repository-layout.md](repository-layout.md) | Folder distribution and structure conventions for root, scripts, and src |
| [../distributions/README.md](../distributions/README.md) | Deployment matrix by stage (dev/stg/pro) and environment (windows/vps-cpu/vps-gpu) |
| [technologies.md](technologies.md) | Libraries, tools, models, and standards used in the project |
| [getting-started.md](getting-started.md) | Local setup, installation, TDD workflow, and code quality checks |
| [deployment.md](deployment.md) | How to run the API server, use Docker, and connect a llama.cpp server |
| [rag-usage.md](rag-usage.md) | RAG pipeline: ingestion, retrieval, custom embedders and vector stores |
| [kbd-usage.md](kbd-usage.md) | Knowledge Base CRUD, tag/keyword search, and integration with RAG |
| [sdk-usage.md](sdk-usage.md) | SDK models for typed parsing of generate endpoint responses |
| [metrics-usage.md](metrics-usage.md) | Advanced generation metrics (cache, RAG, LLM, KBD, DB) |
| [adr-0001-cache-strategy.md](adr-0001-cache-strategy.md) | ADR describing in-memory/TinyDB/Redis cache strategy and trade-offs |
| [incident-runbook.md](incident-runbook.md) | Incident response and rollback playbooks for cache/backend/metrics issues |
| [improvement-history.md](improvement-history.md) | Chronological history of completed improvements |
| [improvement-checklist.md](improvement-checklist.md) | Proposed next-cycle checklist by phase and priority |
