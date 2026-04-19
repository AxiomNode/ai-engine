"""Example injection system for RAG-based game generation.

This package provides a curated corpus of gold-standard game examples
and educational reference material.  The :class:`ExampleInjector`
ingests them into the RAG pipeline at application startup so the LLM
always has high-quality few-shot context.
"""

from ai_engine.examples.injector import ExampleInjector
from ai_engine.examples.corpus import get_corpus_signature

__all__ = ["ExampleInjector", "get_corpus_signature"]
