"""Generation API module for ai-engine.

Provides a FastAPI application that exposes the GameGenerator, RAGPipeline,
and an embedded StatsCollector as HTTP endpoints, ready to be consumed by
game microservices.

Main entry point for uvicorn::

    uvicorn ai_engine.api.app:app --host 0.0.0.0 --port 8001

Factory function for programmatic use or testing::

    from ai_engine.api.app import create_app
    app = create_app(generator=mock_gen, rag_pipeline=mock_pipeline)
"""

from ai_engine.api.app import create_app

__all__ = ["create_app"]
