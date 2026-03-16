# Improvement Checklist

This document tracks the improvement plan for the ai-engine project.

## How we will execute

- Work in priority order (P0 -> P2).
- Implement one item at a time.
- Validate each item with tests/lint/type checks before marking it done.
- Keep changes small and reviewable.

## P0 - Critical

- [x] P0-01 Fix protected generation regression and compatibility wrappers
  - Scope:
    - Ensure `/generate` works when API key is valid.
    - Keep compatibility when wrapped generators do not expose `generate_from_context`.
  - Acceptance criteria:
    - `tests/test_api/test_app.py::TestAPIKeyAuth::test_generate_succeeds_with_correct_key` passes.
    - No new failures in API tests.

- [x] P0-02 Restore static quality gate (ruff + mypy)
  - Scope:
    - Fix current `ruff` violations in source files.
    - Fix current `mypy` errors in `src/ai_engine/sdk/models.py` and `src/ai_engine/api/optimization.py`.
  - Acceptance criteria:
    - `ruff check src tests` passes.
    - `mypy src` passes.

## P1 - High value

- [ ] P1-01 Improve cache observability correctness
  - Scope:
    - Avoid counting non-generation events as cache misses.
    - Add event classification for cleaner summary math.
  - Acceptance criteria:
    - `cache_hit_rate` reflects generation events only.

- [ ] P1-02 Make persistent cache path configurable
  - Scope:
    - Move cache file path to settings/env.
    - Keep safe defaults for local runs.
  - Acceptance criteria:
    - App reads path from config and works in tests/runtime.

- [ ] P1-03 Optimize cache stats/reset complexity
  - Scope:
    - Reduce O(n) scans in cache stats/reset paths.
    - Prepare for larger caches.
  - Acceptance criteria:
    - Cache runtime endpoints remain responsive under larger datasets.

## P2 - Strategic

- [ ] P2-01 Introduce production cache backend option (Redis)
  - Scope:
    - Optional Redis cache layer for multi-worker/multi-instance deployments.
  - Acceptance criteria:
    - Configurable backend with fallback to in-memory/local mode.

- [ ] P2-02 Export Prometheus-friendly metrics
  - Scope:
    - Add metrics endpoint or integration for scrape-based monitoring.
  - Acceptance criteria:
    - Core generation/cache/RAG metrics exposed in a scrape-compatible format.

## Progress log

- 2026-03-16: Checklist created. Next item in progress: P0-01.
- 2026-03-16: P0-01 completed. Fixed compatibility fallback in optimization service and restored protected API-key test path.
- 2026-03-16: P0-02 completed. Resolved mypy errors in SDK/optimization modules and restored lint/type checks to green.
