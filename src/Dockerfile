# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder base: prepare source and metadata
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder-base

WORKDIR /build

# Install build tooling needed by some native wheels (e.g. sentence-transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY ai_engine/ ./src/ai_engine/


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – builder-api: generation API dependencies (api + rag + kbd), CPU-only
# ──────────────────────────────────────────────────────────────────────────────
FROM builder-base AS builder-api

# Force CPU-only wheels for torch to avoid pulling CUDA/nvidia packages.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    ".[api,rag,kbd]"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 – builder-stats: observability API dependencies only (api)
# ──────────────────────────────────────────────────────────────────────────────
FROM builder-base AS builder-stats

# Install the package with all runtime extras into an isolated prefix.
# Use a regular install (not editable) so runtime image does not depend on
# builder filesystem paths such as /build/src.
# [api] → FastAPI + uvicorn + httpx
RUN pip install --no-cache-dir --prefix=/install ".[api]"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 – runtime base: shared runtime setup
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime-base

LABEL org.opencontainers.image.title="ai-engine"
LABEL org.opencontainers.image.description="AxiomNode AI Engine – RAG, LLM inference and educational game generation"
LABEL org.opencontainers.image.source="https://github.com/axiomnode/ai-engine"

# Non-root user for security
RUN useradd --create-home --no-log-init appuser
WORKDIR /app

# Copy source from builder
COPY --from=builder-base /build/src ./src

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


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 – runtime-api: generation API image (includes RAG deps)
# ──────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS runtime-api

COPY --from=builder-api /install /usr/local

# Default command for generation API (docker-compose can still override).
CMD ["uvicorn", "ai_engine.api.app:app", \
    "--host", "0.0.0.0", \
    "--port", "8001", \
    "--workers", "2"]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 – runtime-stats: observability-only image
# ──────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS runtime-stats

COPY --from=builder-stats /install /usr/local

# Default: observability/stats API. Override in docker-compose for other services.
CMD ["uvicorn", "ai_engine.observability.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]


# Backward-compatible target name used by older compose configs.
FROM runtime-stats AS runtime
