# Improvement Checklist (Proposed)

This document tracks the next improvement cycle for the ai-engine project.

## How to use this checklist

- Execute items in order of necessity (Phase 1 -> Phase 4).
- Deliver one item at a time with small, reviewable PRs.
- Mark an item as completed only after tests, lint, and type checks pass.
- Keep acceptance criteria measurable and verifiable.

## Phase 1 - Critical Reliability (Do First)

- [x] N1-01 Add integration tests for Redis cache backend path
  - Scope:
    - Validate read/write/reset behavior when Redis backend is active.
    - Validate fallback behavior when Redis is unavailable.
  - Acceptance criteria:
    - Dedicated integration tests pass in CI.
    - No regression in TinyDB/in-memory behavior.

- [x] N1-02 Add cache-key versioning and invalidation strategy
  - Scope:
    - Add cache key version namespace for schema/prompt changes.
    - Add selective invalidation by prefix/version.
  - Acceptance criteria:
    - Cache misses are intentional after version bump.
    - Invalidation can target a specific version namespace.

- [x] N1-03 Add resilience guards for persistent cache backend failures
  - Scope:
    - Ensure backend errors do not fail generation requests.
    - Emit explicit fallback metadata for observability.
  - Acceptance criteria:
    - Generation succeeds when persistent backend errors occur.
    - Fallback counters are visible in stats/metrics.

## Phase 2 - Performance and Scalability

- [x] N2-01 Add micro-benchmarks for generation latency breakdown
  - Scope:
    - Benchmark cache hit, cache miss, and fallback paths.
    - Track RAG, generation, parse, and total latency.
  - Acceptance criteria:
    - Reproducible benchmark script and baseline results committed.

- [x] N2-02 Optimize persistent index synchronization under concurrency
  - Scope:
    - Harden index consistency for multi-worker scenarios.
    - Add lock/atomicity tests where needed.
  - Acceptance criteria:
    - Concurrency tests pass with no index drift.

- [x] N2-03 Add request-level rate limiting for generation endpoints
  - Scope:
    - Add configurable limits by API key/IP.
    - Preserve compatibility with existing middleware.
  - Acceptance criteria:
    - Excess traffic receives controlled responses (429).
    - Normal traffic remains unaffected.

## Phase 3 - Observability and Operations

- [x] N3-01 Expand Prometheus metrics coverage
  - Scope:
    - Add generation outcome counters by game_type/language.
    - Add backend/fallback counters and cache saturation gauges.
  - Acceptance criteria:
    - New metrics are exposed and documented in metrics usage docs.

- [x] N3-02 Add health diagnostics for dependencies
  - Scope:
    - Report status of LLM endpoint, embedding model readiness, and cache backend.
  - Acceptance criteria:
    - Health endpoint includes dependency-specific diagnostics.

- [x] N3-03 Add structured logging correlation IDs
  - Scope:
    - Add request correlation ID across API, optimizer, and observability events.
  - Acceptance criteria:
    - A single request can be traced end-to-end in logs and metrics.

## Phase 4 - Product Hardening and Developer Experience

- [ ] N4-01 Add architecture decision record (ADR) for cache strategy
  - Scope:
    - Document why/when to use in-memory, TinyDB, and Redis.
    - Define operational trade-offs and limits.
  - Acceptance criteria:
    - ADR approved and linked from docs index.

- [ ] N4-02 Add CI matrix profile for optional extras
  - Scope:
    - Validate behavior with and without optional dependencies (rag/kbd/redis).
  - Acceptance criteria:
    - CI matrix runs green and catches optional-dependency regressions.

- [ ] N4-03 Publish runbook for incidents and rollback
  - Scope:
    - Define playbooks for cache corruption, backend outage, and metrics gaps.
  - Acceptance criteria:
    - Runbook available in docs with step-by-step recovery procedures.

## Progress Log

- 2026-03-16: Proposed checklist created for the next cycle.
- 2026-03-16: N1-01 completed. Added Redis-path tests for read/write/reset behavior with fallback coverage.
- 2026-03-16: N1-02 completed. Added cache namespace versioning and selective invalidation by namespace.
- 2026-03-16: N1-03 completed. Added persistent backend resilience guards and fallback metadata/counters.
- 2026-03-16: N2-01 completed. Added reproducible generation-path benchmark script and committed baseline results.
- 2026-03-16: N2-02 completed. Hardened TinyDB persistent-cache synchronization under concurrency with lock-protected operations and passing concurrency tests.
- 2026-03-16: N2-03 completed. Added configurable fixed-window request rate limiting for generation endpoints with 429 behavior coverage.
- 2026-03-16: N3-01 completed. Expanded Prometheus output with generation outcome counters, persistent backend/fallback counters, and cache saturation gauges.
- 2026-03-16: N3-02 completed. Extended `/health` with dependency diagnostics for generator, RAG pipeline, LLM mode/target, and cache backend status.
- 2026-03-16: N3-03 completed. Added request correlation ID propagation in API headers, optimizer metadata, logs, and observability event/metric outputs.
