# Game Microservices Connection Contract

## Purpose

This contract defines how game-specific microservices (quiz, word-pass, true-false) connect to ai-engine generation APIs.

Scope:
- ai-engine generates content.
- Each game microservice persists generated data in its own database.
- ai-engine does not persist game records in domain databases.

## Base URL and Versioning

- Base URL (example): `http://ai-api:8001`
- Distribution/version header returned by ai-engine: `X-Distribution-Version`
- Correlation header returned by ai-engine: `X-Correlation-ID`

Recommended environment variables in each game microservice:
- `AI_ENGINE_API_BASE_URL`
- `AI_ENGINE_GAMES_API_KEY`
- `AI_ENGINE_TIMEOUT_SECONDS` (recommended: 10-20s)

When split-runtime routing is active, the effective llama destination may differ from the environment default configured inside `ai-engine-api`. Game services should therefore treat API success/failure as the contract boundary and not assume where the model runtime lives.

## Authentication and Required Headers

When `AI_ENGINE_GAMES_API_KEY` is configured in ai-engine, every request to `/generate*` MUST include:
- `X-API-Key: <shared-secret>`

Recommended headers:
- `X-Correlation-ID: <uuid-or-trace-id>` (recommended for tracing)
- `X-Game-Language: en` (optional override; English is the only supported runtime language)
- `X-Difficulty-Percentage: 0..100` (optional override)

## Endpoints for Game Services

Monitoring and cache-management endpoints are intentionally out of scope for
game microservices and are provided by `ai-stats` for backoffice/operations.

### 1) Generic generation endpoint

- Method: `POST`
- Path: `/generate`
- Content-Type: `application/json`

Request body schema:
```json
{
  "query": "string",
  "topic": "string",
  "game_type": "quiz|word-pass|true_false",
  "language": "en",
  "num_questions": 5,
  "difficulty_percentage": 50,
  "letters": "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z",
  "max_tokens": 1024,
  "use_cache": true,
  "force_refresh": false
}
```

### 2) Model-specific generation endpoints (recommended)

These routes reduce payload ambiguity and simplify client code.

- Quiz: `POST /generate/quiz`
- WordPass: `POST /generate/word-pass`
- True/False: `POST /generate/true-false`

These model-specific routes are the preferred integration surface for current game services.

Common query parameters:
- `query` (required)
- `topic` (required)
- `max_tokens` (optional, default `1024`)
- `use_cache` (optional, default `true`)
- `force_refresh` (optional, default `false`)

Quiz/True-False additional query parameter:
- `num_questions` (optional, range `1..50`, default `5`)

WordPass additional query parameter:
- `letters` (optional, default `A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z`)

### Request example: quiz

```bash
curl -X POST "http://localhost:7001/generate/quiz?query=water%20cycle&topic=Science&num_questions=5" \
  -H "X-API-Key: <key>" \
  -H "X-Correlation-ID: 4bde5f2f4dc34f33a4ec8de9c0d994a7" \
  -H "X-Game-Language: en" \
  -H "X-Difficulty-Percentage: 70"
```

### Response shape (generation)

```json
{
  "game_type": "quiz",
  "topic": "Science",
  "difficulty_percentage": 70,
  "game": {
    "title": "...",
    "topic": "Science",
    "difficulty_percentage": 70,
    "questions": [
      {
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": 0,
        "explanation": "..."
      }
    ]
  }
}
```

Note: quiz question text is normalized to avoid inline options in the question title/stem.

## Error Contract

Expected status codes:
- `200`: success
- `401`: missing `X-API-Key`
- `403`: invalid `X-API-Key`
- `422`: invalid request or generation validation failure
- `429`: generation rate limit exceeded
- `503`: generator not initialized

Error body shape:
```json
{
  "detail": "human-readable error message"
}
```

## Reliability and Retry Policy

Client recommendations:
- Retry only on `429` and `5xx`.
- Use exponential backoff with jitter: 250ms, 500ms, 1000ms (max 3 retries).
- Do not retry on `401`, `403`, `422`.
- Log `X-Correlation-ID` from response headers for incident tracing.

If the platform uses admission control for generation capacity, short fast-fail `503` responses should still be treated as retriable only when the caller policy explicitly allows queue/backoff behavior.

## Persistence Responsibility (Game Microservices)

Each game microservice is responsible for:
- Storing generated content in its own domain database.
- Defining publication states (draft/published).
- Applying idempotency rules and deduplication in its persistence layer.

ai-engine is responsible for:
- Generation and optional cache usage.
- Returning structured payload and observability metadata.

## Security Recommendation (Service-to-Service)

Current platform security recommendation:
1. TLS everywhere for transport encryption.
2. mTLS between internal services (service identity + encryption).
3. OAuth2 Client Credentials with JWT access tokens for authorization.
4. Keep `X-API-Key` as short-term compatibility fallback only.

Phased rollout:
- Phase 1 (now): `X-API-Key` + private network + IP allow-list.
- Phase 2 (target): mTLS + OAuth2 scopes (for example `games.generate:quiz`).
- Phase 3 (hardening): short token TTL, key rotation, centralized secret manager, audit logs.

This phased approach minimizes migration risk while upgrading security posture.
