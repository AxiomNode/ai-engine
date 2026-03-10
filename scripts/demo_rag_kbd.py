"""Quick demo: RAG + KBD + game generation with the local Qwen2.5-7B model.

Usage
-----
    python scripts/demo_rag_kbd.py

Requirements
------------
    pip install -e ".[rag,llm,games]"
    python -m ai_engine.llm download          # downloads qwen2.5-7b (~4.8 GB)

The script runs in three stages:
  1. KBD  — build a small in-memory knowledge base and query it.
  2. RAG  — ingest those same entries into the RAG pipeline and retrieve context.
  3. Game — generate a quiz using the retrieved context + the local LLM.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the downloaded model (tries each registered model in priority order)
# ---------------------------------------------------------------------------

from ai_engine.llm.model_manager import MODELS, model_path

PRIORITY = ["qwen2.5-7b", "phi-3.5-mini", "qwen2.5-3b"]

MODEL_PATH = None
for _candidate in PRIORITY:
    if _candidate not in MODELS:
        continue
    try:
        MODEL_PATH = model_path(_candidate)
        log.info("Using model '%s' at: %s", _candidate, MODEL_PATH)
        break
    except FileNotFoundError:
        log.info("Model '%s' not found, trying next...", _candidate)

if MODEL_PATH is None:
    log.error("No downloaded model found.")
    log.error("Run:  python -m ai_engine.llm download")
    log.info("Available models to download:")
    for name in PRIORITY:
        log.info("  python -m ai_engine.llm download %s", name)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. KBD — Knowledge Base demo
# ---------------------------------------------------------------------------

log.info("\n=== KBD ===")

from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

kb.add(KnowledgeEntry(
    entry_id="1",
    title="Fotosíntesis",
    content=(
        "La fotosíntesis es el proceso por el cual las plantas convierten "
        "la luz solar, el agua y el dióxido de carbono en glucosa y oxígeno. "
        "Ocurre principalmente en los cloroplastos, usando la clorofila como pigmento."
    ),
    tags=["biología", "plantas", "energía"],
))

kb.add(KnowledgeEntry(
    entry_id="2",
    title="Sistema solar",
    content=(
        "El sistema solar está formado por el Sol y todos los cuerpos celestes "
        "que orbitan a su alrededor: 8 planetas, lunas, asteroides y cometas. "
        "Los planetas en orden son: Mercurio, Venus, Tierra, Marte, Júpiter, "
        "Saturno, Urano y Neptuno."
    ),
    tags=["astronomía", "física", "planetas"],
))

kb.add(KnowledgeEntry(
    entry_id="3",
    title="Segunda Guerra Mundial",
    content=(
        "La Segunda Guerra Mundial (1939-1945) fue el conflicto armado más grande "
        "de la historia. Involucró a la mayoría de las naciones del mundo. "
        "Terminó con la derrota de Alemania nazi y Japón imperial."
    ),
    tags=["historia", "guerra", "siglo XX"],
))

# --- Queries ---
print("\n[KBD] search_by_tag('biología'):")
for e in kb.search_by_tag("biología"):
    print(f"  [{e.entry_id}] {e.title}")

print("\n[KBD] search_by_keyword('planetas'):")
for e in kb.search_by_keyword("planetas"):
    print(f"  [{e.entry_id}] {e.title}")

print(f"\n[KBD] Total entries: {len(kb)}")

# ---------------------------------------------------------------------------
# 2. RAG — Ingestion and retrieval demo
# ---------------------------------------------------------------------------

log.info("\n=== RAG ===")

from ai_engine.rag.document import Document
from ai_engine.rag.chunker import Chunker
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.pipeline import RAGPipeline

# Import embedder (requires: pip install -e ".[rag]")
try:
    from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
except ImportError:
    log.error("sentence-transformers not installed. Run: pip install -e '.[rag]'")
    sys.exit(1)

embedder = SentenceTransformersEmbedder()       # all-MiniLM-L6-v2 by default
vector_store = InMemoryVectorStore()
pipeline = RAGPipeline(
    embedder=embedder,
    vector_store=vector_store,
    chunker=Chunker(chunk_size=300, chunk_overlap=50),
    top_k=3,
)

# Convert KBD entries into RAG Documents and ingest them
docs = [
    Document(content=entry.content, metadata={"title": entry.title, "tags": entry.tags})
    for entry in kb.list_all()
]
pipeline.ingest(docs)
log.info("Ingested %d documents into the vector store.", len(docs))

# Retrieve context for a query
query = "¿Qué proceso usan las plantas para obtener energía?"
context = pipeline.build_context(query)
print(f"\n[RAG] Query: {query}")
print(f"[RAG] Retrieved context:\n{context}\n")

# ---------------------------------------------------------------------------
# 3. Game generation — Quiz via local LLM
# ---------------------------------------------------------------------------

log.info("\n=== GAME GENERATION ===")

from ai_engine.llm.llama_client import LlamaClient
from ai_engine.games.generator import GameGenerator

# Load the local GGUF model (lazy — loads on first generate() call)
llm = LlamaClient(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    n_gpu_layers=0,     # Set to -1 to offload all layers to GPU
    temperature=0.3,    # Low temperature for consistent structured output
    json_mode=True,     # Enforce valid JSON via GBNF grammar
    default_max_tokens=1024,
)

generator = GameGenerator(
    rag_pipeline=pipeline,
    llm_client=llm,
    default_language="es",
)

print("[GAME] Generating a quiz about 'fotosíntesis'...")
try:
    envelope = generator.generate(
        query="fotosíntesis plantas cloroplastos glucosa",
        topic="Fotosíntesis",
        game_type="quiz",
        num_questions=3,
    )
    print("\n[GAME] Result:")
    print(f"  Type     : {envelope.game_type}")
    print(f"  Topic    : {envelope.topic}")
    print(f"  Language : {envelope.language}")
    print(f"  Questions:")
    game = envelope.game
    for i, q in enumerate(game.questions, 1):  # type: ignore[attr-defined]
        print(f"    {i}. {q.question}")
        for opt in q.options:                  # type: ignore[attr-defined]
            marker = "✓" if opt == q.correct_answer else " "  # type: ignore[attr-defined]
            print(f"       [{marker}] {opt}")
except Exception as exc:
    log.error("Game generation failed: %s", exc)
    log.error("Is the model fully downloaded? Run: python -m ai_engine.llm download")
    sys.exit(1)

print("\nDemo complete.")
