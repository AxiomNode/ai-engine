# Incident Runbook: Cache, Backend, and Metrics

This runbook defines operational response steps for common runtime incidents in ai-engine.

## Scope

Covers:

- cache corruption or inconsistency,
- persistent backend outage (TinyDB/Redis),
- observability or metrics gaps,
- llama target override or split-runtime reachability issues,
- rollback procedures for risky changes.

## Preconditions

Before applying actions:

1. Confirm active branch and deployed version.
2. Capture current health and stats snapshots.
3. Record correlation IDs and timestamps for impacted requests.

Useful endpoints:

- `GET /health`
- `GET /stats`
- `GET /cache/stats`
- `GET /metrics`
- `GET /internal/llama/target`

## Incident 1: TinyDB Cache Corruption

Symptoms:

- JSON decode errors from persistent cache operations,
- sudden drop in cache hit ratio,
- repeated persistent read/write fallback metrics.

Immediate actions:

1. Switch traffic to memory-only behavior temporarily:
- set persistent backend to disabled path or rotate namespace.
2. Invalidate persistent cache:
- `POST /cache/reset?all_namespaces=true`.
3. Restart affected service instance(s).

Recovery:

1. Restore clean TinyDB cache file.
2. Re-enable TinyDB persistent cache.
3. Verify:
- `/health` dependency status is ready,
- `/cache/stats` persistent entries recover,
- fallback/error counters stop increasing abnormally.

## Incident 2: Redis Backend Outage

Symptoms:

- connection errors to Redis,
- `persistent_fallback_total` increasing,
- generation still succeeds but with degraded cache behavior.

Immediate actions:

1. Keep service online (fallback is designed to preserve generation).
2. Confirm outage scope (network/auth/server).
3. If outage is prolonged, switch to TinyDB backend for temporary persistence.

Recovery:

1. Restore Redis connectivity.
2. Validate with cache writes and reads.
3. Reset stale namespace entries if needed:
- `POST /cache/reset?namespace=<namespace>`.

## Incident 3: Metrics Pipeline Gap

Symptoms:

- missing or stale Prometheus series,
- mismatch between `/stats` and `/metrics` counters,
- no correlation ID labels for recent traffic.

Immediate actions:

1. Check `/health` and dependency diagnostics.
2. Verify `/metrics` endpoint response format and freshness.
3. Confirm request flow still includes `X-Correlation-ID`.

Recovery:

1. Restart affected API instance.
2. Re-run synthetic requests with explicit `X-Correlation-ID`.
3. Confirm counters and labels repopulate.

## Incident 4: Split Runtime Or Llama Target Misrouting

Symptoms:

- `ai-engine-api` is healthy but generation requests fail upstream,
- the configured model host is not the host actually used at runtime,
- staging traffic reaches API/stats but not the expected llama runtime.

Immediate actions:

1. Inspect the active llama target override.
2. Compare override state with environment defaults.
3. Test reachability from the API runtime toward the selected llama endpoint.

Recovery:

1. Correct the persisted llama target override or reset it to environment defaults.
2. Re-run a synthetic generation request with correlation ID.
3. Verify that health, generation, and stats agree on the active distribution/runtime path.

## Rollback Procedure

Use this for deployments that introduce regressions.

1. Identify last known good commit on `develop`/`main`.
2. Roll back deployment to that revision.
3. Run smoke checks:
- `/health` is OK,
- `/generate` returns success,
- `/cache/stats` and `/metrics` respond.
4. Keep degraded feature disabled (for example, Redis backend) until root cause is fixed.
5. Open a follow-up hotfix branch with targeted tests.

## Communication Template

- Incident start time:
- Affected components:
- User impact:
- Mitigation applied:
- Current status:
- Next update ETA:

## Post-Incident Checklist

- [ ] Root cause documented.
- [ ] Missing tests added.
- [ ] Monitoring thresholds adjusted.
- [ ] Roll-forward plan reviewed.
- [ ] Improvement checklist updated.
