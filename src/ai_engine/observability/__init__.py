"""Observability module – statistics collection and monitoring API.

Provides tools for tracking LLM generation events, computing aggregate
statistics, and exposing them through a lightweight FastAPI application.

Typical usage::

    from ai_engine.observability import StatsCollector, create_app

    collector = StatsCollector()
    app = create_app(collector)
"""

from ai_engine.observability.api import create_app
from ai_engine.observability.collector import GenerationEvent, StatsCollector

__all__ = [
    "GenerationEvent",
    "StatsCollector",
    "create_app",
]
