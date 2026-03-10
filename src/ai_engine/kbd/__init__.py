"""KBD (Knowledge Base) module.

Provides in-memory and persistent knowledge-base implementations:

- :class:`~ai_engine.kbd.knowledge_base.KnowledgeBase` – fast in-memory store.
- :class:`~ai_engine.kbd.tinydb_knowledge_base.TinyDBKnowledgeBase` – TinyDB-backed
  persistent store (requires the ``kbd`` extra: ``pip install ai-engine[kbd]``).
"""

from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.knowledge_base import KnowledgeBase

try:
    from ai_engine.kbd.tinydb_knowledge_base import TinyDBKnowledgeBase
except ImportError:
    TinyDBKnowledgeBase = None  # type: ignore[assignment,misc]

__all__ = [
    "KnowledgeEntry",
    "KnowledgeBase",
    "TinyDBKnowledgeBase",
]
