"""Hallucination benchmark for the ai-engine generation pipeline.

Tests grounding quality by ingesting KNOWN content and verifying that:
1. RAG retrieval returns relevant chunks (similarity threshold filtering)
2. Generated content is structurally valid
3. Prompt grounding instructions are present
4. Cache keys differentiate difficulty levels
5. Low-relevance queries yield fewer/no results (anti-hallucination)

Uses the REAL sentence-transformers embedder for realistic similarity scores.
Uses mock LLM responses to test the full pipeline validation chain.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from ai_engine.games.generator import GameGenerator
from ai_engine.games.prompts import _SYSTEM, get_prompt
from ai_engine.games.schemas import GameEnvelope
from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore

# Try real embedder; skip entire module if sentence-transformers is absent.
try:
    from ai_engine.rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

pytestmark = pytest.mark.skipif(
    not _HAS_ST, reason="sentence-transformers not installed"
)

# ------------------------------------------------------------------
# Known corpus — factual content we control entirely
# ------------------------------------------------------------------

KNOWN_DOCUMENTS = [
    Document(
        content=(
            "La fotosíntesis es el proceso mediante el cual las plantas convierten "
            "la luz solar, el agua y el dióxido de carbono en glucosa y oxígeno. "
            "Este proceso ocurre en los cloroplastos, específicamente en la clorofila. "
            "La ecuación general es: 6CO2 + 6H2O + luz → C6H12O6 + 6O2. "
            "La fotosíntesis consta de dos fases: la fase luminosa, que ocurre en "
            "los tilacoides, y la fase oscura o ciclo de Calvin, que ocurre en el estroma."
        ),
        doc_id="bio-fotosintesis",
        metadata={"subject": "biología", "topic": "fotosíntesis"},
    ),
    Document(
        content=(
            "La mitosis es un proceso de división celular que produce dos células "
            "hijas genéticamente idénticas a la célula madre. Consta de cuatro fases: "
            "profase, metafase, anafase y telofase. Durante la profase, los cromosomas "
            "se condensan y se hace visible el huso mitótico. En la metafase, los "
            "cromosomas se alinean en el plano ecuatorial. En la anafase, las cromátidas "
            "hermanas se separan hacia los polos opuestos. En la telofase, se forman "
            "dos núcleos nuevos."
        ),
        doc_id="bio-mitosis",
        metadata={"subject": "biología", "topic": "mitosis"},
    ),
    Document(
        content=(
            "La Revolución Francesa comenzó en 1789 con la Toma de la Bastilla el "
            "14 de julio. Las causas principales fueron la crisis económica, la "
            "desigualdad del sistema de estamentos y las ideas ilustradas de Voltaire, "
            "Rousseau y Montesquieu. La Declaración de los Derechos del Hombre y del "
            "Ciudadano fue aprobada el 26 de agosto de 1789. El lema revolucionario "
            "fue 'Libertad, Igualdad, Fraternidad'. La revolución culminó con la "
            "ejecución de Luis XVI en enero de 1793."
        ),
        doc_id="hist-rev-francesa",
        metadata={"subject": "historia", "topic": "revolución francesa"},
    ),
    Document(
        content=(
            "El teorema de Pitágoras establece que en un triángulo rectángulo, "
            "el cuadrado de la hipotenusa es igual a la suma de los cuadrados de "
            "los catetos: a² + b² = c². Este teorema fue conocido por los babilonios "
            "antes de Pitágoras. Un ejemplo: si un cateto mide 3 y el otro 4, la "
            "hipotenusa mide 5, ya que 9 + 16 = 25. Las ternas pitagóricas más "
            "comunes son (3,4,5), (5,12,13) y (8,15,17)."
        ),
        doc_id="mat-pitagoras",
        metadata={"subject": "matemáticas", "topic": "pitágoras"},
    ),
]

# Facts we can verify against — extracted from the corpus above
FOTOSINTESIS_FACTS = {
    "cloroplastos",
    "clorofila",
    "glucosa",
    "oxígeno",
    "tilacoides",
    "calvin",
    "estroma",
    "luminosa",
}

MITOSIS_FACTS = {
    "profase",
    "metafase",
    "anafase",
    "telofase",
    "cromosomas",
    "huso",
    "cromátidas",
}

REV_FRANCESA_FACTS = {
    "1789",
    "bastilla",
    "voltaire",
    "rousseau",
    "montesquieu",
    "luis xvi",
    "1793",
    "libertad",
    "igualdad",
    "fraternidad",
}

PITAGORAS_FACTS = {
    "hipotenusa",
    "catetos",
    "triángulo rectángulo",
    "babilonios",
    "3,4,5",
    "5,12,13",
}

UNRELATED_QUERIES = [
    "recetas de cocina italiana para pasta carbonara",
    "programación en rust con async await",
    "estrategias de marketing digital en redes sociales",
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _build_pipeline() -> RAGPipeline:
    """Build a real pipeline with sentence-transformers and ingest known docs."""
    embedder = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
    store = InMemoryVectorStore()
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    pipeline = RAGPipeline(embedder=embedder, vector_store=store, chunker=chunker)
    pipeline.ingest(KNOWN_DOCUMENTS)
    return pipeline


def _count_facts_in_text(text: str, facts: set[str]) -> tuple[int, int]:
    """Count how many known facts appear in text. Returns (found, total)."""
    lower = text.lower()
    found = sum(1 for fact in facts if fact.lower() in lower)
    return found, len(facts)


class _GroundedMockLLM:
    """Mock LLM that generates quiz from the ACTUAL context it receives.

    Parses the context block from the prompt and builds questions
    referencing real content — simulating a well-grounded model.
    """

    async def generate(self, prompt: str, max_tokens: int = 1024, **kw) -> str:
        # Extract context between ### Context and ### Requirements
        ctx_match = re.search(
            r"### Context\n(.*?)\n### Requirements", prompt, re.DOTALL
        )
        context = ctx_match.group(1).strip() if ctx_match else ""

        # Determine game type from prompt
        if '"game_type": "word-pass"' in prompt:
            return self._word_pass_from_context(context)
        if '"game_type": "true_false"' in prompt:
            return self._true_false_from_context(context)
        return self._quiz_from_context(context)

    def _quiz_from_context(self, context: str) -> str:
        words = context.split()
        # Use actual words from context for options to simulate grounding
        data = {
            "game_type": "quiz",
            "title": "Quiz basado en contexto",
            "questions": [
                {
                    "question": f"Según el contexto, ¿qué proceso involucra {words[3] if len(words) > 3 else 'elementos'}?",
                    "options": [
                        words[0] if words else "A",
                        words[5] if len(words) > 5 else "B",
                        words[10] if len(words) > 10 else "C",
                        words[15] if len(words) > 15 else "D",
                    ],
                    "correct_index": 0,
                    "explanation": f"Basado en el contexto: {context[:100]}",
                }
            ],
        }
        return json.dumps(data, ensure_ascii=False)

    def _true_false_from_context(self, context: str) -> str:
        data = {
            "game_type": "true_false",
            "title": "Verdadero o Falso",
            "statements": [
                {
                    "statement": context[:80] if context else "Afirmación de prueba",
                    "is_true": True,
                    "explanation": "Extraído del contexto proporcionado.",
                }
            ],
        }
        return json.dumps(data, ensure_ascii=False)

    def _word_pass_from_context(self, context: str) -> str:
        words = [w for w in context.split() if len(w) > 3][:2]
        data = {
            "game_type": "word-pass",
            "title": "Rosco educativo",
            "words": [
                {
                    "letter": (words[0][0].upper() if words else "A"),
                    "hint": f"Término del contexto: {context[:50]}",
                    "answer": words[0] if words else "Alfa",
                    "starts_with": True,
                },
            ],
        }
        return json.dumps(data, ensure_ascii=False)


class _HallucinatingMockLLM:
    """Mock LLM that IGNORES context and invents content — simulates hallucination."""

    async def generate(self, prompt: str, max_tokens: int = 1024, **kw) -> str:
        topic = "tema"
        for line in prompt.splitlines():
            if "- Topic:" in line:
                topic = line.split("- Topic:", 1)[1].strip() or topic
                break

        data = {
            "game_type": "quiz",
            "title": "Quiz inventado",
            "questions": [
                {
                    "question": f"En el tema {topic}, ¿cuál es la capital de Atlantis?",
                    "options": ["Poseidonia", "Neptuno", "Oceania", "Aquapolis"],
                    "correct_index": 0,
                    "explanation": f"Aunque se menciona {topic}, Poseidonia fue la capital legendaria de Atlantis.",
                },
                {
                    "question": f"Sobre {topic}, ¿en qué año se descubrió el elemento Unobtanium?",
                    "options": ["2045", "2030", "2099", "1999"],
                    "correct_index": 0,
                    "explanation": f"En esta respuesta sobre {topic}, el Unobtanium fue descubierto en 2045 por Dr. Fiction.",
                },
            ],
        }
        return json.dumps(data, ensure_ascii=False)


class _InvalidJSONMockLLM:
    """Mock LLM that returns invalid JSON on first attempt, valid on retry."""

    def __init__(self):
        self._calls = 0

    async def generate(self, prompt: str, max_tokens: int = 1024, **kw) -> str:
        self._calls += 1
        if self._calls == 1:
            return "Hmm let me think... { broken json here ]["

        topic = "matemáticas"
        for line in prompt.splitlines():
            if "- Topic:" in line:
                topic = line.split("- Topic:", 1)[1].strip() or topic
                break

        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Retry Quiz",
                "questions": [
                    {
                        "question": f"¿Qué concepto básico pertenece a {topic}?",
                        "options": [f"Un principio de {topic}", "Un planeta", "Un animal", "Una capital inventada"],
                        "correct_index": 0,
                        "explanation": f"Recovered after retry with a question aligned to {topic}.",
                    }
                ],
            }
        )


# ==================================================================
# TEST SUITE
# ==================================================================


class TestRetrievalQuality:
    """Test that RAG retrieval returns relevant chunks and filters irrelevant ones."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return _build_pipeline()

    def test_relevant_query_returns_docs(self, pipeline):
        """On-topic query should retrieve documents with high similarity."""
        docs = pipeline.retrieve("fotosíntesis en plantas")
        assert len(docs) > 0, "Expected at least 1 document for on-topic query"
        scores = pipeline.retriever.last_scores
        assert all(
            s >= 0.3 for s in scores
        ), f"All scores should be ≥ 0.3 (min_score), got {scores}"

    def test_relevant_query_has_high_similarity(self, pipeline):
        """On-topic query should have mean similarity > 0.4."""
        pipeline.retrieve("fases de la mitosis celular")
        scores = pipeline.retriever.last_scores
        mean_score = sum(scores) / len(scores) if scores else 0
        assert (
            mean_score > 0.4
        ), f"Mean similarity {mean_score:.3f} too low for on-topic query"

    def test_unrelated_query_returns_fewer_docs(self, pipeline):
        """Off-topic query should return fewer docs after min_score filtering."""
        for query in UNRELATED_QUERIES:
            docs = pipeline.retrieve(query)
            scores = pipeline.retriever.last_scores
            # With min_score=0.3, most unrelated queries should return 0–2 docs
            assert len(docs) <= 3, (
                f"Unrelated query '{query}' returned {len(docs)} docs "
                f"(scores: {[round(s,3) for s in scores]}). "
                "min_score filter may not be working."
            )

    def test_context_contains_source_facts(self, pipeline):
        """Retrieved context for 'fotosíntesis' should contain known facts."""
        context = pipeline.build_context("fotosíntesis cloroplastos plantas")
        found, total = _count_facts_in_text(context, FOTOSINTESIS_FACTS)
        ratio = found / total
        assert ratio >= 0.5, (
            f"Context grounding ratio {ratio:.0%} ({found}/{total} facts) — "
            "retrieval is not pulling relevant content"
        )

    def test_history_context_grounding(self, pipeline):
        """Retrieved context for 'Revolución Francesa' should contain known facts."""
        context = pipeline.build_context("Revolución Francesa causas 1789")
        found, total = _count_facts_in_text(context, REV_FRANCESA_FACTS)
        ratio = found / total
        assert ratio >= 0.5, (
            f"Context grounding ratio {ratio:.0%} ({found}/{total} facts) — "
            "history retrieval not grounded"
        )

    def test_math_context_grounding(self, pipeline):
        """Retrieved context for 'Pitágoras' should contain known facts."""
        context = pipeline.build_context("teorema de Pitágoras triángulo")
        found, total = _count_facts_in_text(context, PITAGORAS_FACTS)
        ratio = found / total
        assert (
            ratio >= 0.5
        ), f"Context grounding ratio {ratio:.0%} ({found}/{total} facts)"

    def test_cross_topic_isolation(self, pipeline):
        """Query about math should NOT primarily return biology content."""
        context = pipeline.build_context("teorema de Pitágoras hipotenusa")
        # Math facts should dominate over biology facts
        math_found, _ = _count_facts_in_text(context, PITAGORAS_FACTS)
        bio_found, _ = _count_facts_in_text(context, FOTOSINTESIS_FACTS)
        assert math_found >= bio_found, (
            f"Cross-topic contamination: math query returned more bio facts "
            f"({bio_found}) than math facts ({math_found})"
        )


class TestPromptGrounding:
    """Test that prompt templates include anti-hallucination instructions."""

    def test_system_prompt_has_grounding_instruction(self):
        """System prompt must instruct model to use ONLY provided context."""
        assert (
            "EXCLUSIVELY" in _SYSTEM or "exclusively" in _SYSTEM.lower()
        ), "System prompt lacks explicit grounding instruction"
        assert (
            "fabricat" in _SYSTEM.lower() or "invent" in _SYSTEM.lower()
        ), "System prompt doesn't warn against fabrication"

    def test_quiz_prompt_passes_context(self):
        """Quiz prompt must include the context block."""
        prompt = get_prompt("quiz", context="TEST_CONTEXT_MARKER", language="es")
        assert "TEST_CONTEXT_MARKER" in prompt
        assert "EXCLUSIVELY" in prompt or "exclusively" in prompt.lower()

    def test_word_pass_prompt_passes_context(self):
        """Word-pass prompt must include the context block."""
        prompt = get_prompt("word-pass", context="TEST_CONTEXT_WP", language="es")
        assert "TEST_CONTEXT_WP" in prompt

    def test_true_false_prompt_passes_context(self):
        """True/false prompt must include the context block."""
        prompt = get_prompt("true_false", context="TEST_CONTEXT_TF", language="es")
        assert "TEST_CONTEXT_TF" in prompt


class TestGenerationGrounding:
    """Test that generation pipeline produces grounded output from known context."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return _build_pipeline()

    def test_grounded_llm_produces_valid_quiz(self, pipeline):
        """Grounded mock LLM should produce a valid quiz envelope."""
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_GroundedMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("fotosíntesis en plantas", game_type="quiz"))
        assert isinstance(envelope, GameEnvelope)
        assert envelope.game_type == "quiz"
        game_dict = envelope.game.to_dict()
        assert len(game_dict.get("questions", [])) >= 1

    def test_grounded_llm_quiz_references_context(self, pipeline):
        """Grounded quiz should contain words from the actual RAG context."""
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_GroundedMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("fotosíntesis en plantas", game_type="quiz"))
        game_text = json.dumps(envelope.game.to_dict(), ensure_ascii=False).lower()
        # The grounded mock uses words from context, so at least some bio words should appear
        found, total = _count_facts_in_text(
            game_text, {"fotosíntesis", "plantas", "luz", "proceso"}
        )
        assert found >= 1, "Grounded quiz doesn't reference any context terms"

    def test_grounded_llm_true_false(self, pipeline):
        """Grounded mock should produce valid true/false game."""
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_GroundedMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("mitosis celular", game_type="true_false"))
        assert envelope.game_type == "true_false"

    def test_grounded_llm_word_pass(self, pipeline):
        """Grounded mock should produce valid word-pass game."""
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_GroundedMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("Revolución Francesa", game_type="word-pass"))
        assert envelope.game_type == "word-pass"

    def test_hallucinating_llm_still_passes_structure(self, pipeline):
        """Even a hallucinating LLM produces structurally valid output.

        This shows that structural validation alone is NOT sufficient
        to detect hallucinations — content verification is needed.
        """
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_HallucinatingMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("fotosíntesis", game_type="quiz"))
        # Structurally valid — this is the problem with hallucinations
        assert isinstance(envelope, GameEnvelope)
        game_text = json.dumps(envelope.game.to_dict(), ensure_ascii=False).lower()
        # But content is NOT grounded — "atlantis" and "unobtanium" are fabricated
        assert "atlantis" in game_text, "Expected hallucinated content from mock"
        # Verify these terms do NOT appear in our corpus
        for doc in KNOWN_DOCUMENTS:
            assert "atlantis" not in doc.content.lower()
            assert "unobtanium" not in doc.content.lower()

    def test_retry_mechanism_works(self, pipeline):
        """Invalid JSON on first attempt should trigger retry and succeed."""
        gen = GameGenerator(
            rag_pipeline=pipeline,
            llm_client=_InvalidJSONMockLLM(),
            default_language="es",
        )
        envelope = _run(gen.generate("matemáticas", game_type="quiz"))
        assert isinstance(envelope, GameEnvelope)
        assert gen.last_run_metrics.get("retry_used") is True


class TestCacheKeyDifferentiation:
    """Test that cache keys properly differentiate parameters."""

    def test_different_difficulty_different_key(self):
        """Cache key must differ when difficulty_percentage changes."""
        from ai_engine.api.optimization import GenerationOptimizationService
        from ai_engine.api.schemas import GenerateRequest

        req_easy = GenerateRequest(
            query="fotosíntesis",
            game_type="quiz",
            language="es",
            difficulty_percentage=20,
        )
        req_hard = GenerateRequest(
            query="fotosíntesis",
            game_type="quiz",
            language="es",
            difficulty_percentage=80,
        )

        svc = GenerationOptimizationService.__new__(GenerationOptimizationService)
        svc._cache_namespace = "v1"

        key_easy = svc._cache_key(req_easy)
        key_hard = svc._cache_key(req_hard)
        assert (
            key_easy != key_hard
        ), "Cache keys are identical for different difficulty_percentage values!"

    def test_same_params_same_key(self):
        """Identical requests should produce the same cache key."""
        from ai_engine.api.optimization import GenerationOptimizationService
        from ai_engine.api.schemas import GenerateRequest

        req1 = GenerateRequest(
            query="fotosíntesis", game_type="quiz", difficulty_percentage=50
        )
        req2 = GenerateRequest(
            query="fotosíntesis", game_type="quiz", difficulty_percentage=50
        )

        svc = GenerationOptimizationService.__new__(GenerationOptimizationService)
        svc._cache_namespace = "v1"

        assert svc._cache_key(req1) == svc._cache_key(req2)


class TestTemperatureConfig:
    """Test that LLM generation parameters are tuned for factual content."""

    def test_default_temperature_is_low(self):
        """Default temperature should be ≤ 0.4 for educational content."""
        from ai_engine.llm.llama_client import LlamaClient

        client = LlamaClient(api_url="http://test:8080")
        assert client.temperature <= 0.4, (
            f"Temperature {client.temperature} is too high for factual generation. "
            "Educational content should use ≤ 0.4 to reduce hallucinations."
        )

    def test_top_p_is_conservative(self):
        """top_p should be ≤ 0.95 for controlled generation."""
        from ai_engine.llm.llama_client import LlamaClient

        client = LlamaClient(api_url="http://test:8080")
        assert client.top_p <= 0.95, f"top_p {client.top_p} is too permissive"

    def test_json_grammar_available(self):
        """JSON GBNF grammar should be available for structured output."""
        from ai_engine.llm.llama_client import JSON_GRAMMAR

        assert "object" in JSON_GRAMMAR
        assert "array" in JSON_GRAMMAR
        assert "string" in JSON_GRAMMAR


class TestMetricsCompleteness:
    """Test that observability metrics track hallucination-relevant signals."""

    def test_summary_includes_retry_rate(self):
        """Summary should report retry_rate for monitoring prompt/model quality."""
        from ai_engine.observability.collector import StatsCollector

        collector = StatsCollector()
        summary = collector.summary()
        assert "retry_rate" in summary, "Missing retry_rate metric"
        assert "retry_used_count" in summary, "Missing retry_used_count metric"

    def test_summary_includes_rag_similarity(self):
        """Summary should report avg_rag_similarity for retrieval quality monitoring."""
        from ai_engine.observability.collector import StatsCollector

        collector = StatsCollector()
        summary = collector.summary()
        assert "avg_rag_similarity" in summary, "Missing avg_rag_similarity metric"
        assert (
            "avg_rag_context_length_chars" in summary
        ), "Missing context length metric"

    def test_prometheus_includes_new_metrics(self):
        """Prometheus output should include retry and similarity metrics."""
        from ai_engine.observability.collector import (
            GenerationEvent,
            StatsCollector,
            summary_to_prometheus,
        )

        collector = StatsCollector()
        collector.record(
            GenerationEvent(
                timestamp=1.0,
                prompt_chars=100,
                response_chars=200,
                latency_ms=50.0,
                max_tokens=1024,
                json_mode=True,
                success=True,
                game_type="quiz",
                metadata={
                    "event_type": "generation",
                    "retry_used": True,
                    "avg_rag_similarity": 0.72,
                    "rag_context_length_chars": 1500,
                    "cache_hit": False,
                },
            )
        )
        text = summary_to_prometheus(collector.summary())
        assert "ai_engine_retry_rate" in text
        assert "ai_engine_retry_used_count" in text
        assert "ai_engine_avg_rag_similarity" in text
        assert "ai_engine_avg_rag_context_length_chars" in text
