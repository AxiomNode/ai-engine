# Bridge Microservice Connection Contract

## Purpose

This contract defines how a dedicated bridge microservice integrates with ai-engine for:
- Data ingestion into RAG pipelines.
- Observability event forwarding and monitoring access.

Scope:
- Bridge owns integration orchestration (from external systems to ai-engine).
- ai-engine owns ingestion processing and observability aggregation.

## Services and Base URLs

- Generation/Ingestion API base URL (example): `http://ai-api:8001`
- Observability API base URL (example): `http://ai-stats:8000`

Recommended bridge environment variables:
- `AI_ENGINE_API_BASE_URL`
- `AI_ENGINE_STATS_BASE_URL`
- `AI_ENGINE_BRIDGE_API_KEY`
- `BRIDGE_TIMEOUT_SECONDS` (recommended: 10-20s)

## Authentication and Headers

If scoped key protection is enabled, bridge requests MUST include:
- `X-API-Key: <shared-secret>`

Key mapping:
- `AI_ENGINE_BRIDGE_API_KEY` protects bridge-facing endpoints (`/ingest*`, `/stats*`, `/cache/*`, `/metrics`).
- `AI_ENGINE_STATS_API_KEY` is dedicated to internal `ai-api -> ai-stats` event publishing (`POST /events`).

Recommended tracing header:
- `X-Correlation-ID: <uuid-or-trace-id>`

Ingestion-specific optional header:
- `X-Ingest-Source: <source-name>`

## Ingestion Contract (ai-api)

### 1) Generic ingest endpoint

- Method: `POST`
- Path: `/ingest`
- Content-Type: `application/json`

Request body schema:
```json
{
  "documents": [
    {
      "content": "string",
      "doc_id": "optional-string",
      "metadata": {
        "key": "value"
      }
    }
  ]
}
```

Optional query parameter:
- `source=<source-name>`

Optional model-specific ingest endpoints:
- `POST /ingest/quiz`
- `POST /ingest/word-pass`
- `POST /ingest/true-false`

Use model-specific paths when bridge can classify the target game domain.

### Request example: ingest quiz corpus

```bash
curl -X POST "http://localhost:7001/ingest/quiz?source=lms-catalog" \
  -H "X-API-Key: <key>" \
  -H "X-Correlation-ID: c6b9d4f86df64a9f8f338ea73f59ecf5" \
  -H "X-Ingest-Source: nightly-sync" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Water cycle includes evaporation, condensation and precipitation.",
        "doc_id": "science-001",
        "metadata": {"provider": "lms", "lang": "en", "grade": "5"}
      }
    ]
  }'
```

### Ingest response

```json
{
  "ingested": 1
}
```

## Observability Contract (ai-stats)

### 1) Push event

- Method: `POST`
- Path: `/events`

Request body schema:
```json
{
  "prompt": "string",
  "response": "string",
  "latency_ms": 123.45,
  "max_tokens": 1024,
  "json_mode": true,
  "success": true,
  "game_type": "quiz",
  "error": null,
  "metadata": {
    "event_type": "generation|ingest",
    "correlation_id": "uuid",
    "distribution_version": "dev-v1"
  }
}
```

Success response:
```json
{
  "message": "Event recorded."
}
```

### 2) Monitoring reads

- `GET /health`
- `GET /stats`
- `GET /stats/history?last_n=100`
- `GET /cache/stats`
- `POST /cache/reset?namespace=<namespace>`
- `POST /cache/reset?all_namespaces=true`
- `GET /metrics` (Prometheus format)

Optional admin operation:
- `POST /stats/reset`

Monitoring/cache routes are exposed by `ai-stats`. The bridge should not call
`ai-api` public routes for monitoring concerns.

## Error Contract

Expected status codes:
- `200`: success
- `401`: missing `X-API-Key`
- `403`: invalid `X-API-Key`
- `422`: validation issue
- `5xx`: transient server-side issue

Error body shape:
```json
{
  "detail": "human-readable error message"
}
```

## Bridge Reliability Requirements

Bridge should implement:
- Retry on `429` and `5xx` with exponential backoff and jitter.
- Dead-letter queue for ingestion/event failures after max retries.
- Circuit breaker for repeated transient failures.
- Idempotency key policy for inbound messages from external systems.

## Security Recommendation (Bridge <-> ai-engine)

Recommended method:
1. Use mTLS for bridge-to-ai-engine communication in private networks.
2. Use OAuth2 Client Credentials and JWT access tokens for service authorization.
3. Keep `X-API-Key` only as temporary compatibility control.

Minimum baseline if mTLS/OAuth2 is not ready yet:
- Enforce TLS termination.
- Enforce `X-API-Key`.
- Restrict source IP ranges.
- Rotate keys regularly via secret manager.

This provides immediate protection now and a clear migration path to stronger zero-trust service-to-service security.
