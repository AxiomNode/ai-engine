# Metrics Usage

This guide explains the advanced metrics emitted by the stats service for
microservice workloads.

## Endpoints

Stats service (`ai-stats`):

- `POST /events`
- `GET /stats`
- `GET /stats/history?last_n=100`
- `GET /metrics`
- `GET /health`

Generation service (`ai-api`):

- `POST /generate`
- `POST /generate/sdk`
- `POST /ingest`
- `GET /health`

The generation service records telemetry for:

- cache behavior
- RAG phase latency
- LLM phase latency
- parse latency
- retrieved RAG documents per request
- KBD and TinyDB usage
- language distribution
- generation outcomes by game type and language
- persistent backend/fallback behavior and cache saturation
- request correlation IDs across API responses and event metrics

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
- `generation_outcome_by_game_type`
- `generation_outcome_by_language`
- `persistent_backend_counts`
- `persistent_fallback_total`
- `persistent_error_counts`
- `correlation_id_counts`

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

## Prometheus export endpoint

`GET /metrics` returns scrape-compatible text exposition with key metrics,
including total calls, success rate, cache hit rate, latency averages,
KBD/DB counters, and labeled counts (game type, event type, language,
cache layer).

Additional labeled counters now include:

- generation outcomes by game type (`success`/`failure`)
- generation outcomes by language (`success`/`failure`)
- persistent backend usage counts (`tinydb`, `redis`, `none`)
- persistent fallback and backend error totals
- request correlation ID counts (for end-to-end tracing)

Additional cache gauges now include:

- in-memory entry count
- in-memory configured capacity
- in-memory saturation ratio
- persistent entry count

## Health diagnostics

`GET /health` (on `ai-api`) includes dependency diagnostics for:

- generator readiness/type
- RAG pipeline readiness (embedder/vector store)
- LLM mode/target (`api` or `local`)
- cache backend status/namespace

Every request includes `X-Correlation-ID` in the response headers. If the
header is provided by the caller, the same value is propagated to generation
event metadata and metrics.
