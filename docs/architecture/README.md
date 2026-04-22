# Architecture Docs

This section groups architecture and design references.

## Scope

Use this section to understand what `ai-engine` owns locally:

- AI runtime slices (`ai-engine-api`, `ai-engine-stats`, llama runtime, cache)
- internal module boundaries and extension points
- repository-local technology and model decisions

Cross-repository deployment policy and service topology belong in the central `docs` repository.

## Contents

- [project-context.md](project-context.md) — Vision, scope, and context.
- [architecture.md](architecture.md) — Layers, components, and data flow.
- [technologies.md](technologies.md) — Tech stack, models, and standards.
- [repository-layout.md](repository-layout.md) — Repository structure conventions.
- [adr-0001-cache-strategy.md](adr-0001-cache-strategy.md) — Cache architecture decision.

## Reading order

1. Start with [project-context.md](project-context.md) for runtime role and platform position.
2. Continue with [architecture.md](architecture.md) for module and runtime slice boundaries.
3. Use [technologies.md](technologies.md) for dependencies, model defaults, and compatibility details.
4. Consult [adr-0001-cache-strategy.md](adr-0001-cache-strategy.md) when changing caching behavior.
