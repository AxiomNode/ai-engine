"""CLI demo: RAG + KBD + game generation using a local GGUF model.

Usage
-----
    python scripts/demo_rag_kbd.py

Requirements
------------
    pip install -e ".[rag,llm,games]"
    python -m ai_engine.llm download

The demo runs in three stages:
  1. KBD: build an in-memory knowledge base and run basic queries.
  2. RAG: ingest entries and retrieve contextual chunks.
  3. Game: generate a quiz with the local LLM and RAG context.
"""

from __future__ import annotations

import logging
from typing import Any

PRIORITY = ["qwen2.5-7b", "phi-3.5-mini", "qwen2.5-3b"]
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def locate_model_path() -> str:
    """Resolve the first available local model from the priority list."""
    from ai_engine.llm.model_manager import MODELS, model_path

    for candidate in PRIORITY:
        if candidate not in MODELS:
            continue
        try:
            resolved = str(model_path(candidate))
            log.info("Using model '%s' at: %s", candidate, resolved)
            return resolved
        except FileNotFoundError:
            log.info("Model '%s' not found, trying next...", candidate)

    log.error("No downloaded model found.")
    log.error("Run: python -m ai_engine.llm download")
    log.info("Available models to download:")
    for name in PRIORITY:
        log.info("  python -m ai_engine.llm download %s", name)
    raise SystemExit(1)


def run_kbd_demo() -> Any:
    """Create an in-memory KBD and print a couple of example searches."""
    from ai_engine.kbd.entry import KnowledgeEntry
    from ai_engine.kbd.knowledge_base import KnowledgeBase

    log.info("\n=== KBD ===")

    kb = KnowledgeBase()
    kb.add(
        KnowledgeEntry(
            entry_id="1",
            title="Fotosintesis",
            content=(
                "La fotosintesis es el proceso por el cual las plantas convierten "
                "la luz solar, el agua y el dioxido de carbono en glucosa y oxigeno. "
                "Ocurre principalmente en los cloroplastos, usando la clorofila como pigmento."
            ),
            tags=["biologia", "plantas", "energia"],
        )
    )
    kb.add(
        KnowledgeEntry(
            entry_id="2",
            title="Sistema solar",
            content=(
                "El sistema solar esta formado por el Sol y todos los cuerpos celestes "
                "que orbitan a su alrededor: 8 planetas, lunas, asteroides y cometas. "
                "Los planetas en orden son: Mercurio, Venus, Tierra, Marte, Jupiter, "
                "Saturno, Urano y Neptuno."
            ),
            tags=["astronomia", "fisica", "planetas"],
        )
    )
    kb.add(
        KnowledgeEntry(
            entry_id="3",
            title="Segunda Guerra Mundial",
            content=(
                "La Segunda Guerra Mundial (1939-1945) fue el conflicto armado mas grande "
                "de la historia. Involucro a la mayoria de las naciones del mundo. "
                "Termino con la derrota de Alemania nazi y Japon imperial."
            ),
            tags=["historia", "guerra", "siglo XX"],
        )
    )

    print("\n[KBD] search_by_tag('biologia'):")
    for entry in kb.search_by_tag("biologia"):
        print(f"  [{entry.entry_id}] {entry.title}")

    print("\n[KBD] search_by_keyword('planetas'):")
    for entry in kb.search_by_keyword("planetas"):
        print(f"  [{entry.entry_id}] {entry.title}")

    print(f"\n[KBD] Total entries: {len(kb)}")
    return kb


def run_rag_demo(kb: Any) -> Any:
    """Ingest KBD entries in the RAG pipeline and print retrieved context."""
    from ai_engine.rag.chunker import Chunker
    from ai_engine.rag.document import Document
    from ai_engine.rag.pipeline import RAGPipeline
    from ai_engine.rag.vector_store import InMemoryVectorStore

    log.info("\n=== RAG ===")
    try:
        from ai_engine.rag.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install -e '.[rag]'")
        raise SystemExit(1)

    embedder = SentenceTransformersEmbedder()
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=InMemoryVectorStore(),
        chunker=Chunker(chunk_size=300, chunk_overlap=50),
        top_k=3,
    )

    docs = [
        Document(
            content=entry.content, metadata={"title": entry.title, "tags": entry.tags}
        )
        for entry in kb.list_all()
    ]
    pipeline.ingest(docs)
    log.info("Ingested %d documents into the vector store.", len(docs))

    query = "Que proceso usan las plantas para obtener energia?"
    context = pipeline.build_context(query)
    print(f"\n[RAG] Query: {query}")
    print(f"[RAG] Retrieved context:\n{context}\n")
    return pipeline


def run_game_demo(model_path: str, pipeline: Any) -> None:
    """Generate and print a quiz using local LLM + RAG context."""
    from ai_engine.games.generator import GameGenerator
    from ai_engine.games.schemas import QuizGame
    from ai_engine.llm.llama_client import LlamaClient

    log.info("\n=== GAME GENERATION ===")

    llm = LlamaClient(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        temperature=0.3,
        json_mode=True,
        default_max_tokens=1024,
    )
    generator = GameGenerator(
        rag_pipeline=pipeline,
        llm_client=llm,
        default_language="es",
    )

    print("[GAME] Generating a quiz about 'fotosintesis'...")
    try:
        envelope = generator.generate(
            query="fotosintesis plantas cloroplastos glucosa",
            topic="Fotosintesis",
            game_type="quiz",
            num_questions=3,
        )
    except Exception as exc:
        log.error("Game generation failed: %s", exc)
        log.error(
            "Is the model fully downloaded? Run: python -m ai_engine.llm download"
        )
        raise SystemExit(1) from exc

    print("\n[GAME] Result:")
    print(f"  Type     : {envelope.game_type}")
    print(f"  Topic    : {envelope.game.topic}")
    print("  Language : es")
    print("  Questions:")
    game = envelope.game
    if not isinstance(game, QuizGame):
        raise ValueError("Expected quiz game output for this demo")
    for i, question in enumerate(game.questions, 1):
        print(f"    {i}. {question.question}")
        for option_idx, option in enumerate(question.options):
            marker = "*" if option_idx == question.correct_index else " "
            print(f"       [{marker}] {option}")


def main() -> int:
    """Run the complete KBD + RAG + game generation demo."""
    model_path = locate_model_path()
    kb = run_kbd_demo()
    pipeline = run_rag_demo(kb)
    run_game_demo(model_path, pipeline)
    print("\nDemo complete.")
    return 0
