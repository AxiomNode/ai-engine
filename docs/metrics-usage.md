# Metrics Usage

This guide explains the advanced metrics now emitted by the generation API for
microservice workloads.

## Endpoints

- `GET /stats`
- `GET /stats/history?last_n=100`
- `GET /cache/stats`
- `POST /cache/reset`
- `GET /metrics`

The generation service records telemetry for:

- cache behavior
- RAG phase latency
- LLM phase latency
- parse latency
- retrieved RAG documents per request
- KBD and TinyDB usage
- language distribution

## New summary fields

`GET /stats` includes:

- `cache_hits`
- `cache_misses`
- `cache_hit_rate`
- `cache_layer_counts`
- `avg_rag_latency_ms`
- `avg_llm_latency_ms`
- `avg_parse_latency_ms`
- `avg_total_latency_ms`
- `kbd_hits_total`
- `db_reads_total`
- `db_writes_total`
- `language_counts`
- `cache_runtime` (runtime entries and cache configuration)

Request-level metrics now also include:

- `rag_docs_retrieved`
- `orchestration_engine`

## SDK generation endpoint

Use `POST /generate/sdk` when downstream microservices need a stable typed
payload for SDK models.

The response includes:

- `model_type`
- `metadata` (generation id, language, language id)
- `data` (model payload)
- `metrics` (request-level optimization telemetry)

## Cache controls per request

`POST /generate` and `POST /generate/sdk` accept:

- `use_cache` (default `true`)
- `force_refresh` (default `false`)

Set `force_refresh=true` to bypass cache and force a fresh model generation.

## Cache runtime endpoints

- `GET /cache/stats` returns cache runtime status and entry counts.
- `POST /cache/reset` clears in-memory and persistent generation cache and
	returns removed-entry counters.

## Prometheus export endpoint

`GET /metrics` returns scrape-compatible text exposition with key metrics,
including total calls, success rate, cache hit rate, latency averages,
KBD/DB counters, and labeled counts (game type, event type, language,
cache layer).
