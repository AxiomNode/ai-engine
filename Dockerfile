# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder: install Python dependencies into a clean prefix
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tooling needed by some native wheels (e.g. sentence-transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with all runtime extras into an isolated prefix.
# [api]  → FastAPI + uvicorn + httpx
# [rag]  → sentence-transformers (RAG embeddings)
# No llama-cpp-python needed: the generation service connects to llama-server via HTTP.
RUN pip install --no-cache-dir --prefix=/install -e ".[api,rag]"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime: minimal image that runs the observability API
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="ai-engine"
LABEL org.opencontainers.image.description="AxiomNode AI Engine – RAG, LLM inference and educational game generation"
LABEL org.opencontainers.image.source="https://github.com/axiomnode/ai-engine"

# Non-root user for security
RUN useradd --create-home --no-log-init appuser
WORKDIR /app

# Copy installed packages and source from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build/src ./src

# Models directory (mounted at runtime via volume — not baked into the image)
RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser

# Environment – can all be overridden in docker-compose / kubectl
ENV AI_ENGINE_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Both services expose different ports; docker-compose overrides the command
# for each. Default CMD runs the observability/stats API on port 8000.
EXPOSE 8000 8001

# Liveness probe — overridden per-service in docker-compose with $PORT.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')"

# Default: observability/stats API. Override in docker-compose for other services.
CMD ["uvicorn", "ai_engine.observability.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
