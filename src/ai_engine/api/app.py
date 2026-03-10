"""FastAPI application for the ai-engine content generation service.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following endpoints:

- ``GET  /health``           – Liveness check.
- ``POST /generate``         – Generate a structured educational game via RAG + LLM.
- ``POST /ingest``           – Ingest documents into the RAG pipeline.
- ``GET  /stats``            – Aggregate generation statistics.
- ``GET  /stats/history``    – Recent generation event log.

Environment variables (used when ``generator`` / ``rag_pipeline`` are not
injected directly):

- ``AI_ENGINE_LLAMA_URL``       – URL of a running llama.cpp HTTP server
  (e.g. ``http://llama-server:8080/completion``).  Takes priority over local.
- ``AI_ENGINE_MODEL_PATH``      – Path to a local GGUF model file (fallback
  when ``AI_ENGINE_LLAMA_URL`` is not set).
- ``AI_ENGINE_EMBEDDING_MODEL`` – sentence-transformers model name
  (default: ``all-MiniLM-L6-v2``).
- ``AI_ENGINE_API_KEY``         – When set, every request must include an
  ``X-API-Key`` header with this value.  Absent or incorrect keys return
  HTTP 401 / 403 respectively.  Unset means no authentication required.

Run standalone::

    uvicorn ai_engine.api.app:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query, Request
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for the generation API.  "
        "Install it with:  pip install ai-engine[api]"
    ) from _imp_err

from ai_engine.api.schemas import GenerateRequest, IngestRequest  # noqa: E402
from ai_engine.api.middleware import add_api_key_middleware  # noqa: E402
from ai_engine.config import get_settings  # noqa: E402
from ai_engine.observability.collector import StatsCollector  # noqa: E402


def _build_from_env() -> tuple[Any, Any]:
    """Initialise components from environment variables.

    Reads ``AI_ENGINE_LLAMA_URL``, ``AI_ENGINE_MODEL_PATH``, and
    ``AI_ENGINE_EMBEDDING_MODEL`` to wire up the LLM client, RAG pipeline,
    and game generator.

    Returns:
        A ``(TrackedGameGenerator, RAGPipeline)`` tuple ready to attach to
        the FastAPI app state.

    Raises:
        RuntimeError: If neither ``AI_ENGINE_LLAMA_URL`` nor
            ``AI_ENGINE_MODEL_PATH`` is set.
    """
    from ai_engine.games.generator import GameGenerator
    from ai_engine.llm.llama_client import LlamaClient
    from ai_engine.observability.middleware import TrackedGameGenerator
    from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
    from ai_engine.rag.pipeline import RAGPipeline
    from ai_engine.rag.vector_store import InMemoryVectorStore

    settings = get_settings()
    api_url = settings.llama_url
    model_path = settings.model_path

    if api_url is None and model_path is None:
        raise RuntimeError(
            "Set AI_ENGINE_LLAMA_URL (HTTP server) or "
            "AI_ENGINE_MODEL_PATH (local GGUF) before starting the API."
        )

    embedding_model = settings.embedding_model
    logger.info("Loading embedding model: %s", embedding_model)
    embedder = SentenceTransformersEmbedder(model_name=embedding_model)
    pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())

    logger.info(
        "Connecting to LLM: %s",
        api_url if api_url else model_path,
    )
    llm = LlamaClient(api_url=api_url, model_path=model_path, json_mode=True)
    raw_gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)

    # Wrap for automatic stats collection
    collector = StatsCollector()
    tracked = TrackedGameGenerator(raw_gen, collector)

    return tracked, pipeline


def _get_generator(request: Request) -> Any:
    """Retrieve the generator from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.observability.middleware.TrackedGameGenerator`
        attached to ``app.state``.
    """
    return request.app.state.generator


def _get_pipeline(request: Request) -> Any:
    """Retrieve the RAG pipeline from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.rag.pipeline.RAGPipeline` attached to
        ``app.state``.
    """
    return request.app.state.rag_pipeline


def _get_collector(request: Request) -> StatsCollector:
    """Retrieve the StatsCollector from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.observability.collector.StatsCollector`
        attached to ``app.state``.
    """
    return request.app.state.collector  # type: ignore[no-any-return]


def create_app(
    generator: Any = None,
    rag_pipeline: Any = None,
    collector: StatsCollector | None = None,
) -> FastAPI:
    """Create and return a configured FastAPI generation application.

    When ``generator`` and ``rag_pipeline`` are provided they are used
    directly (useful for testing and sub-app mounting).  When omitted,
    the components are initialised from environment variables during the
    application lifespan startup event.

    Args:
        generator: A :class:`~ai_engine.observability.middleware.TrackedGameGenerator`
            (or duck-type compatible) to use for content generation.
        rag_pipeline: A :class:`~ai_engine.rag.pipeline.RAGPipeline` to use
            for document ingestion.
        collector: A :class:`~ai_engine.observability.collector.StatsCollector`
            to record generation events.  Creates a new one if ``None``.

    Returns:
        A :class:`~fastapi.FastAPI` instance ready to be served.
    """
    _collector = collector if collector is not None else StatsCollector()

    # Store mutable references so lifespan can replace them if needed.
    _state: dict[str, Any] = {
        "generator": generator,
        "rag_pipeline": rag_pipeline,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Initialise and tear down application-level resources."""
        if app.state.generator is None:
            # Build from environment variables when no explicit injection.
            _state["generator"], _state["rag_pipeline"] = _build_from_env()
            app.state.generator = _state["generator"]
            app.state.rag_pipeline = _state["rag_pipeline"]

        logger.info("ai-engine generation API started.")
        yield
        logger.info("ai-engine generation API shutting down.")

    app = FastAPI(
        title="ai-engine generation API",
        description=(
            "Content generation service for AxiomNode educational games. "
            "Exposes RAG-augmented LLM generation as a single REST endpoint."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Set state immediately so the app is usable without triggering the lifespan
    # (e.g. when using TestClient without context manager, or when dependencies
    # are injected directly).  Lifespan will fill in None values from env vars.
    app.state.generator = generator
    app.state.rag_pipeline = rag_pipeline
    app.state.collector = _collector

    # Attach API key middleware when AI_ENGINE_API_KEY is set in the environment.
    add_api_key_middleware(app)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health", tags=["monitoring"])
    def health(request: Request) -> dict[str, Any]:
        """Return basic health and event-count information.

        Returns:
            A dictionary with ``status`` and ``total_events`` keys.
        """
        return {
            "status": "ok",
            "total_events": len(_get_collector(request)),
        }

    @app.post("/generate", tags=["generation"])
    def generate(req: GenerateRequest, request: Request) -> dict[str, Any]:
        """Generate a structured educational game using RAG + LLM.

        The pipeline:
        1. Retrieves relevant context from the ingested knowledge base.
        2. Builds a game-type-specific prompt.
        3. Sends it to the LLM with JSON grammar constraint.
        4. Validates and returns the parsed game as JSON.

        Args:
            req: Generation parameters (topic, game_type, language, …).

        Returns:
            A dictionary with ``game_type`` and ``game`` keys matching the
            :class:`~ai_engine.games.schemas.GameEnvelope` structure.

        Raises:
            HTTPException 422: If JSON extraction or schema validation fails.
            HTTPException 503: If the generator is not available.
        """
        gen = _get_generator(request)
        if gen is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")
        try:
            envelope = gen.generate(
                query=req.query,
                topic=req.topic,
                game_type=req.game_type,
                language=req.language,
                num_questions=req.num_questions,
                letters=req.letters,
                max_tokens=req.max_tokens,
            )
            return {"game_type": envelope.game_type, "game": envelope.game.to_dict()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/ingest", tags=["rag"])
    def ingest(req: IngestRequest, request: Request) -> dict[str, int]:
        """Ingest documents into the RAG knowledge base.

        Documents are chunked, embedded, and stored in the in-memory vector
        store.  Call this endpoint before ``/generate`` to provide the LLM
        with relevant context for a specific topic.

        Args:
            req: List of documents to ingest.

        Returns:
            A dictionary with ``ingested`` key reporting how many documents
            were processed.

        Raises:
            HTTPException 503: If the RAG pipeline is not available.
        """
        from ai_engine.rag.document import Document

        pipeline = _get_pipeline(request)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialised.")
        docs = [
            Document(
                content=d.content,
                doc_id=d.doc_id,
                metadata=d.metadata,
            )
            for d in req.documents
        ]
        pipeline.ingest(docs)
        return {"ingested": len(docs)}

    @app.get("/stats", tags=["monitoring"])
    def get_stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics over all recorded generation events.

        Returns:
            A summary dictionary produced by
            :meth:`~ai_engine.observability.collector.StatsCollector.summary`.
        """
        return _get_collector(request).summary()

    @app.get("/stats/history", tags=["monitoring"])
    def get_history(
        request: Request,
        last_n: int | None = Query(
            default=None,
            ge=1,
            description="Return only the N most recent events.",
        ),
    ) -> list[dict[str, Any]]:
        """Return the raw generation event log.

        Args:
            last_n: If provided, only the *last_n* most recent events are
                included.

        Returns:
            A list of event dictionaries ordered oldest-first.
        """
        return _get_collector(request).history(last_n=last_n)

    return app


# Module-level app instance — used by:
#   uvicorn ai_engine.api.app:app --port 8001
app: FastAPI = create_app()
