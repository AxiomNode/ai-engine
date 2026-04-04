# AGENTS

## Repo purpose
Python AI service for RAG ingestion/retrieval, local LLM orchestration, game generation, and observability APIs.

## Key paths
- src/ai_engine/api: FastAPI app and middleware
- src/ai_engine/rag: chunking, embeddings, retrieval pipeline
- src/ai_engine/games: schemas, prompts, generation flow
- src/ai_engine/llm: llama client and model manager
- src/ai_engine/observability: collector, middleware, stats API
- docs/: architecture, guides, operations

## Local commands
- cd src && pip install -e ".[dev]"
- cd src && pytest
- cd src && uvicorn ai_engine.api.app:app --reload

## CI/CD notes
- Service repo CI dispatches platform-infra build on push to main.
- Deployment automation currently targets dev via platform-infra.

## LLM editing rules
- Keep docs/comments in English.
- Prefer minimal, layered changes (api/rag/llm/observability boundaries).
- Add or update tests for behavior changes.
