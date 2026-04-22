"""CLI demo suite for validating AI-Engine modules and integrations.

This module provides small, deterministic command-line demos that exercise
individual modules and cross-module integrations without requiring a GPU or a
real downloaded LLM model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

from ai_engine.api.schemas import DocumentInput, GenerateRequest, IngestRequest
from ai_engine.config import get_settings
from ai_engine.games.generator import GameGenerator
from ai_engine.games.schemas import QuizGame
from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.knowledge_base import KnowledgeBase
from ai_engine.llm.model_manager import list_models
from ai_engine.observability.collector import StatsCollector
from ai_engine.observability.middleware import TrackedGameGenerator, TrackedLlamaClient
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore

DemoStatus = Literal["PASS", "FAIL", "SKIP"]

_ANSI_RESET = "\033[0m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_CYAN = "\033[36m"


def _supports_color() -> bool:
    """Return whether ANSI colors should be used in current output."""
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("TERM") == "dumb":
        return False
    return bool(sys.stdout.isatty())


def _colorize(text: str, color: str) -> str:
    """Wrap text with ANSI color codes when terminal supports it."""
    if not _supports_color():
        return text
    return f"{color}{text}{_ANSI_RESET}"


def _format_status(status: DemoStatus) -> str:
    """Format demo status with color for easier scanning."""
    if status == "PASS":
        return _colorize(status, _ANSI_GREEN)
    if status == "FAIL":
        return _colorize(status, _ANSI_RED)
    return _colorize(status, _ANSI_YELLOW)


@dataclass
class DemoResult:
    """Stores the outcome of a single demo execution."""

    name: str
    status: DemoStatus
    details: str
    duration_ms: float


class DeterministicEmbedder(Embedder):
    """Lightweight deterministic embedder for local demos and tests."""

    def embed_text(self, text: str) -> list[float]:
        size = float(len(text))
        vowels = float(sum(1 for ch in text.lower() if ch in "aeiou"))
        spaces = float(text.count(" "))
        return [size, vowels, spaces]


class FakeLlamaClient:
    """Simple LLM stub returning a valid quiz JSON payload."""

    def __init__(self) -> None:
        self.default_max_tokens = 256
        self.json_mode = True

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> str:
        _prompt = prompt
        _tokens = max_tokens
        _extra = kwargs
        _ = (_prompt, _tokens, _extra)
        payload = {
            "game_type": "quiz",
            "title": "Demo Quiz",
            "questions": [
                {
                    "question": "What process turns liquid water into vapor?",
                    "options": [
                        "Condensation",
                        "Evaporation",
                        "Precipitation",
                        "Freezing",
                    ],
                    "correct_index": 1,
                    "explanation": "Evaporation transforms liquid water into vapor.",
                }
            ],
        }
        return json.dumps(payload)


class FakeGenerator:
    """Minimal generator stub used to test observability wrappers."""

    default_max_tokens = 128

    async def generate(self, *args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {"ok": True}

    async def generate_raw(self, *args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {"raw": True}


def _box(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"[DEMO] {title}")
    print("=" * 72)


def _run_demo(name: str, fn: Callable[[], str]) -> DemoResult:
    start = time.perf_counter()
    try:
        details = fn()
        status: DemoStatus = "PASS"
    except ImportError as exc:
        details = f"Optional dependency missing: {exc}"
        status = "SKIP"
    except Exception as exc:  # noqa: BLE001
        details = str(exc)
        status = "FAIL"
    elapsed = (time.perf_counter() - start) * 1000
    return DemoResult(name=name, status=status, details=details, duration_ms=elapsed)


def demo_config() -> str:
    """Validate configuration loading and defaults."""
    settings = get_settings()
    models_dir = Path(settings.models_dir)
    if not settings.embedding_model:
        raise ValueError("embedding_model is empty")
    return (
        f"embedding_model={settings.embedding_model}, "
        f"models_dir={models_dir.name}, api_key_set={bool(settings.api_key)}"
    )


def demo_llm_registry() -> str:
    """Validate model registry and local download state checks."""
    models = list_models()
    if not models:
        raise ValueError("No registered models found")
    downloaded = sum(1 for model in models if model["downloaded"])
    return f"registered_models={len(models)}, downloaded_models={downloaded}"


def demo_kbd() -> str:
    """Exercise KBD CRUD and querying functionality."""
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry("1", "Python", "Python is dynamic.", tags=["lang", "python"]))
    kb.add(KnowledgeEntry("2", "RAG", "RAG improves responses.", tags=["ai", "rag"]))

    by_tag = kb.search_by_tag("python")
    by_keyword = kb.search_by_keyword("improves")
    if len(by_tag) != 1 or len(by_keyword) != 1:
        raise ValueError("Unexpected KBD query results")
    return f"entries={len(kb)}, by_tag={len(by_tag)}, by_keyword={len(by_keyword)}"


def demo_rag() -> str:
    """Exercise RAG ingestion and retrieval using deterministic embeddings."""
    pipeline = RAGPipeline(
        embedder=DeterministicEmbedder(),
        vector_store=InMemoryVectorStore(),
        top_k=2,
    )
    pipeline.ingest(
        [
            Document(content="The water cycle includes evaporation and condensation."),
            Document(content="Plants perform photosynthesis using sunlight."),
        ]
    )
    context = pipeline.build_context("water evaporation")
    if "water cycle" not in context.lower():
        raise ValueError("Expected context not retrieved")
    return f"context_chars={len(context)}, contains_water_cycle=True"


def demo_games_with_rag_llm() -> str:
    """Exercise GameGenerator connected to RAG + fake LLM client."""
    pipeline = RAGPipeline(
        embedder=DeterministicEmbedder(),
        vector_store=InMemoryVectorStore(),
        top_k=1,
    )
    pipeline.ingest([Document(content="Evaporation changes liquid water into vapor.")])

    generator = GameGenerator(
        rag_pipeline=pipeline,
        llm_client=cast(Any, FakeLlamaClient()),
    )
    envelope = asyncio.run(
        generator.generate(
            query="water cycle",
            game_type="quiz",
            num_questions=1,
            language="en",
        )
    )
    if envelope.game_type != "quiz":
        raise ValueError("Expected quiz output")
    game = envelope.game
    if not isinstance(game, QuizGame):
        raise ValueError("Expected QuizGame instance")
    return f"game_type={envelope.game_type}, questions={len(game.questions)}"


def demo_observability() -> str:
    """Exercise collectors and tracking middleware wrappers."""
    collector = StatsCollector()

    tracked_llm = TrackedLlamaClient(FakeLlamaClient(), collector)
    asyncio.run(tracked_llm.generate("Return JSON"))

    tracked_gen = TrackedGameGenerator(FakeGenerator(), collector)
    asyncio.run(tracked_gen.generate(query="x", game_type="quiz"))

    summary = collector.summary()
    if summary["total_calls"] < 2:
        raise ValueError("Observability calls were not recorded")
    return (
        f"total_calls={summary['total_calls']}, "
        f"success_rate={summary['success_rate']}, "
        f"quiz_calls={summary['game_type_counts'].get('quiz', 0)}"
    )


def demo_api_schemas() -> str:
    """Exercise API request schema validation with concrete payloads."""
    generate_req = GenerateRequest(
        query="water",
        game_type="quiz",
        language="en",
        item_count=3,
        max_tokens=256,
    )
    ingest_req = IngestRequest(
        documents=[
            DocumentInput(
                content="Water evaporates when heated.",
                doc_id="d1",
                metadata={"source": "demo"},
            )
        ]
    )
    return (
        f"generate_tokens={generate_req.max_tokens}, "
        f"ingest_docs={len(ingest_req.documents)}"
    )


def demo_full_integration() -> str:
    """Run an end-to-end in-memory flow across KBD -> RAG -> Games -> Observability."""
    kb = KnowledgeBase()
    kb.add(
        KnowledgeEntry(
            "water-1",
            "Water Cycle",
            "Evaporation, condensation and precipitation are key steps.",
            tags=["science"],
        )
    )

    pipeline = RAGPipeline(
        embedder=DeterministicEmbedder(),
        vector_store=InMemoryVectorStore(),
        top_k=1,
    )
    pipeline.ingest(
        [
            Document(content=entry.content, metadata={"title": entry.title})
            for entry in kb.list_all()
        ]
    )

    collector = StatsCollector()
    generator = GameGenerator(
        rag_pipeline=pipeline,
        llm_client=cast(Any, FakeLlamaClient()),
    )
    tracked_generator = TrackedGameGenerator(generator, collector)

    envelope = asyncio.run(
        tracked_generator.generate(
            query="water cycle",
            game_type="quiz",
            num_questions=1,
            language="en",
        )
    )
    if envelope.game_type != "quiz":
        raise ValueError("Full integration did not produce quiz output")

    summary = collector.summary()
    return (
        f"kb_entries={len(kb)}, "
        f"retrieved_context_chars={len(pipeline.build_context('water cycle'))}, "
        f"tracked_calls={summary['total_calls']}"
    )


DEMO_CATALOG: dict[str, tuple[str, Callable[[], str]]] = {
    "config": ("Config and environment loading", demo_config),
    "llm": ("LLM model registry and availability", demo_llm_registry),
    "kbd": ("Knowledge base CRUD and search", demo_kbd),
    "rag": ("RAG ingest and retrieval", demo_rag),
    "games": ("Games generator with RAG+LLM", demo_games_with_rag_llm),
    "observability": ("Stats collector and tracking wrappers", demo_observability),
    "api": ("API schemas validation", demo_api_schemas),
    "integration": ("Cross-module end-to-end flow", demo_full_integration),
}


def _print_catalog() -> None:
    _box("Available Demos")
    for key, (label, _) in DEMO_CATALOG.items():
        print(f"- {key:12s} {label}")
    print("- all          Run all demos and print a summary table")


def _print_summary(results: list[DemoResult]) -> None:
    _box("Demo Summary")
    print(f"{'Demo':14s} {'Status':6s} {'Time(ms)':>10s}  Details")
    print("-" * 72)
    for result in results:
        status_text = _format_status(result.status)
        print(
            f"{result.name:14s} {status_text:6s} {result.duration_ms:10.2f}  {result.details}"
        )

    passed = sum(1 for item in results if item.status == "PASS")
    failed = sum(1 for item in results if item.status == "FAIL")
    skipped = sum(1 for item in results if item.status == "SKIP")
    print("-" * 72)
    totals = (
        f"TOTAL: {len(results)} | "
        f"{_colorize(f'PASS={passed}', _ANSI_GREEN)} "
        f"{_colorize(f'FAIL={failed}', _ANSI_RED)} "
        f"{_colorize(f'SKIP={skipped}', _ANSI_YELLOW)}"
    )
    print(_colorize(totals, _ANSI_CYAN) if _supports_color() else totals)


def run_selected(selected: list[str]) -> int:
    """Run selected demos and print detailed, stylized output."""
    results: list[DemoResult] = []

    for name in selected:
        label, demo_fn = DEMO_CATALOG[name]
        _box(f"{name}: {label}")
        result = _run_demo(name, demo_fn)
        results.append(result)
        print(f"Status : {_format_status(result.status)}")
        print(f"Time   : {result.duration_ms:.2f} ms")
        print(f"Result : {result.details}")

    _print_summary(results)
    return 1 if any(result.status == "FAIL" for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return command-line parser."""
    parser = argparse.ArgumentParser(
        prog="ai-engine-demo",
        description="Run simple CLI demos for AI-Engine modules and integrations.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List available demos")

    run_parser = subparsers.add_parser("run", help="Run one demo or all demos")
    run_parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        help="Demo key (config|llm|kbd|rag|games|observability|api|integration|all)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the demo suite."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "list"):
        _print_catalog()
        return 0

    if args.command == "run":
        requested = str(args.demo).strip().lower()
        if requested == "all":
            selected = list(DEMO_CATALOG.keys())
        else:
            if requested not in DEMO_CATALOG:
                parser.error(
                    f"Unknown demo '{requested}'. Use 'list' to see available demos."
                )
            selected = [requested]
        return run_selected(selected)

    parser.error(f"Unknown command '{args.command}'.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
