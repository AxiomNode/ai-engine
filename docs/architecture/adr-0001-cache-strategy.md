# ADR-0001: Generation Cache Strategy

- Status: Accepted
- Date: 2026-03-16
- Owners: ai-engine maintainers

## Context

The generation API now serves high-throughput use cases and must keep latency low while preserving resilience under backend failures.

We need a cache strategy that:

- minimizes repeated generation latency for hot requests,
- survives process restarts when required,
- supports selective invalidation after prompt/schema updates,
- degrades safely when persistent backends are unavailable.

The project currently supports three cache layers/choices:

- in-memory LRU cache,
- TinyDB-backed persistent cache,
- Redis-backed persistent cache.

## Decision

Adopt a layered cache strategy with explicit namespace versioning and fallback behavior:

1. Read path
- Check in-memory LRU first.
- On miss, check configured persistent backend (TinyDB or Redis).
- On persistent hit, repopulate in-memory cache.

2. Write path
- Write fresh generation response to in-memory cache (when enabled).
- Attempt write to persistent backend.
- If persistent write/read fails, continue request flow and emit fallback telemetry.

3. Invalidation
- Use cache namespace version in keys.
- Support selective reset by namespace and full reset for all namespaces.

4. Concurrency
- Protect TinyDB operations with shared process lock to avoid corruption.
- Keep Redis operations idempotent and index-backed per namespace.

## Rationale

### Why in-memory LRU

- Fastest access path.
- Reduces repeated serialization and backend round-trips.
- Good for hot-key microservice workloads.

Trade-offs:
- Not shared across workers/processes.
- Volatile across restarts.

### Why TinyDB as local persistent default

- Simple file-based persistence for local/dev and single-node deployments.
- No external service dependency.

Trade-offs:
- Not ideal for high write concurrency across many workers/processes.
- File corruption risk without explicit synchronization.

### Why Redis as scalable persistent option

- Shared cache across instances.
- Better operational profile for multi-worker and distributed services.
- Native TTL and set operations simplify index handling.

Trade-offs:
- External infrastructure dependency.
- Network latency and connectivity failure modes.

## Operational Guidance

Choose backend by environment profile:

- Local development: in-memory + TinyDB.
- Single-node production with moderate load: TinyDB allowed, monitor fallback/error counters.
- Multi-instance or high concurrency: Redis preferred.

Namespace policy:

- Bump namespace on prompt/schema/model behavior changes.
- Use selective namespace reset for controlled invalidation during rollout.

## Consequences

Positive:

- Lower median latency with memory hits.
- Backward-compatible persistence options.
- Safe degradation when persistent backend fails.
- Clear observability for cache/fallback behavior.

Negative:

- Additional configuration complexity.
- Multiple backend code paths require profile-based CI and tests.

## Alternatives Considered

1. Redis only
- Rejected for local/dev simplicity and zero-dependency workflows.

2. TinyDB only
- Rejected for distributed/high-concurrency production scenarios.

3. No persistence (memory only)
- Rejected due to cold-start and restart penalties.

## Verification

- CI profile matrix validates optional dependency combinations.
- Observability metrics expose backend usage, fallback totals, and cache saturation.
- Incident runbook defines rollback and recovery actions.
