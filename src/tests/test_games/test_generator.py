"""Tests for ai_engine.games.generator – GameGenerator with mock LLM."""

import asyncio
import json

import pytest

from ai_engine.games.generator import GameGenerator
from ai_engine.games.schemas import (
    GameEnvelope,
    QuizGame,
    TrueFalseGame,
    WordPassGame,
)
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore

# ------------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously for test convenience."""
    return asyncio.run(coro)


class _FixedEmbedder(Embedder):
    """Embedder that returns a fixed vector for deterministic tests."""

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _MockLLMQuiz:
    """Mock LLM that returns a valid quiz JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        topic = "math"
        marker = "Focus specifically on this topic/request:"
        for line in prompt.splitlines():
            if marker in line:
                topic = line.split(marker, 1)[1].strip() or topic
                break

        data = {
            "game_type": "quiz",
            "title": "Mock Quiz",
            "questions": [
                {
                    "question": f"Which option best matches the requested topic {topic}?",
                    "options": [
                        "An unrelated concept",
                        f"A core fact about {topic}",
                        "A random planet",
                        "A generic animal",
                    ],
                    "correct_index": 1,
                    "explanation": f"This mock explanation stays focused on {topic}.",
                }
            ],
        }
        return json.dumps(data)


class _SpyLLMQuiz(_MockLLMQuiz):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.prompts.append(prompt)
        return await super().generate(prompt, max_tokens=max_tokens, **kwargs)


class _MockLLMWordPass:
    """Mock LLM that returns a valid word-pass JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "word-pass",
            "title": "Mock Rosco",
            "words": [
                {
                    "letter": "A",
                    "hint": "First letter",
                    "answer": "Alpha",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Second letter",
                    "answer": "Beta",
                    "starts_with": True,
                },
            ],
        }
        return json.dumps(data)


class _MockLLMTrueFalse:
    """Mock LLM that returns a valid true/false JSON."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "true_false",
            "title": "Mock T/F",
            "statements": [
                {
                    "statement": "The sky is blue",
                    "is_true": True,
                    "explanation": "Rayleigh scattering.",
                },
            ],
        }
        return json.dumps(data)


class _MockLLMBadOutput:
    """Mock LLM that returns non-JSON garbage."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return "I cannot help you with that. Sorry!"


class _MockLLMRetryThenValid:
    """Mock LLM that fails once, then returns valid JSON on retry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return '{"game_type":"quiz","title":"Broken"'
        data = {
            "game_type": "quiz",
            "title": "Recovered Quiz",
            "questions": [
                {
                    "question": "What is 2+2 in basic math?",
                    "options": ["3", "4", "5", "6"],
                    "correct_index": 1,
                    "explanation": "Basic math arithmetic shows that 2+2 equals 4.",
                }
            ],
        }
        return json.dumps(data)


class _MockLLMMalformedQuizPayload:
    """Mock LLM that returns an invalid quiz payload."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "educational-game",
            "questions": [
                {
                    "question": "",
                    "options": [""],
                    "correct_index": 9,
                }
            ],
        }
        return json.dumps(data)


class _MockLLMSemanticRetryThenValid:
    """Mock LLM that first returns semantically invalid JSON, then valid JSON."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "game_type": "quiz",
                    "title": "Retry Quiz",
                    "questions": [
                        {
                            "question": "",
                            "options": [
                                "Photosynthesis",
                                "Respiration",
                                "Evaporation",
                                "Condensation",
                            ],
                            "correct_index": 0,
                            "explanation": "Plants use photosynthesis to transform light energy.",
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Photosynthesis Basics",
                "questions": [
                    {
                        "question": "What process allows plants to convert light into energy?",
                        "options": [
                            "Photosynthesis",
                            "Respiration",
                            "Evaporation",
                            "Condensation",
                        ],
                        "correct_index": 0,
                        "explanation": "Plants use photosynthesis to transform light energy into chemical energy.",
                    }
                ],
            }
        )


class _MockLLMOffTopicRetryThenValid:
    """Mock LLM that first returns a structurally valid but off-topic quiz."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "game_type": "quiz",
                    "title": "Fotosintesis",
                    "questions": [
                        {
                            "question": "Que proceso usan las plantas para convertir la luz en energia?",
                            "options": [
                                "Fotosintesis",
                                "Respiracion",
                                "Fermentacion",
                                "Condensacion",
                            ],
                            "correct_index": 0,
                            "explanation": "La fotosintesis transforma la energia luminosa en energia quimica.",
                        },
                        {
                            "question": "Cual es el planeta mas grande del sistema solar?",
                            "options": ["Marte", "Venus", "Jupiter", "Mercurio"],
                            "correct_index": 2,
                            "explanation": "Jupiter es el planeta mas grande del sistema solar.",
                        },
                    ],
                }
            )
        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Fotosintesis",
                "questions": [
                    {
                        "question": "Que organulo celular realiza la fotosintesis en las plantas?",
                        "options": [
                            "Cloroplasto",
                            "Mitocondria",
                            "Nucleo",
                            "Ribosoma",
                        ],
                        "correct_index": 0,
                        "explanation": "Los cloroplastos contienen clorofila y son el lugar donde ocurre la fotosintesis.",
                    },
                    {
                        "question": "Que gas absorben las plantas para llevar a cabo la fotosintesis?",
                        "options": [
                            "Dioxido de carbono",
                            "Oxigeno",
                            "Helio",
                            "Nitrogeno",
                        ],
                        "correct_index": 0,
                        "explanation": "La fotosintesis utiliza dioxido de carbono junto con agua y luz solar.",
                    },
                ],
            }
        )


class _MockLLMTopicRootMatch:
    """Mock LLM that uses an inflected topic form instead of the exact token."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        data = {
            "game_type": "quiz",
            "title": "Matematica aplicada",
            "questions": [
                {
                    "question": "Que propiedad matematica describe la suma de los angulos internos de un triangulo?",
                    "options": [
                        "Siempre suman 180 grados",
                        "Siempre suman 90 grados",
                        "Siempre suman 360 grados",
                        "Depende del color del triangulo",
                    ],
                    "correct_index": 0,
                    "explanation": "La geometria matematica euclidiana establece que suman 180 grados.",
                }
            ],
        }
        return json.dumps(data)


class _MockLLMLongOffTopicMixed:
    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Fotosintesis",
                "questions": [
                    {
                        "question": "Que gas absorben las plantas para llevar a cabo la fotosintesis?",
                        "options": [
                            "Dioxido de carbono",
                            "Oxigeno",
                            "Helio",
                            "Nitrogeno",
                        ],
                        "correct_index": 0,
                        "explanation": "La fotosintesis utiliza dioxido de carbono junto con agua y luz solar.",
                    },
                    {
                        "question": "Cual es la capital de Francia?",
                        "options": ["Madrid", "Paris", "Roma", "Berlin"],
                        "correct_index": 1,
                        "explanation": "Paris es la capital francesa.",
                    },
                ],
            }
        )


class _MockLLMBroadCategoryTopic:
    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Science and Nature",
                "questions": [
                    {
                        "question": "What process allows plants to convert sunlight into chemical energy?",
                        "options": [
                            "Photosynthesis",
                            "Respiration",
                            "Fermentation",
                            "Condensation",
                        ],
                        "correct_index": 0,
                        "explanation": "Photosynthesis lets plants transform light into stored chemical energy.",
                    }
                ],
            }
        )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def rag_pipeline():
    emb = _FixedEmbedder()
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(embedder=emb, vector_store=store)
    pipeline.ingest(
        [
            Document(content="Python is a high-level language.", doc_id="d1"),
            Document(content="The Earth orbits the Sun.", doc_id="d2"),
        ]
    )
    return pipeline


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestGameGeneratorInit:

    def test_requires_rag_pipeline(self):
        with pytest.raises(ValueError, match="rag_pipeline"):
            GameGenerator(rag_pipeline=None, llm_client=_MockLLMQuiz())

    def test_requires_llm_client(self, rag_pipeline):
        with pytest.raises(ValueError, match="llm_client"):
            GameGenerator(rag_pipeline=rag_pipeline, llm_client=None)


class TestGameGeneratorQuiz:

    def test_generate_quiz(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMQuiz())
        result = _run(gen.generate(query="Python", game_type="quiz"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "quiz"
        assert isinstance(result.game, QuizGame)
        assert result.game.title == "Mock Quiz"
        assert len(result.game.questions) == 1
        assert result.game.questions[0].correct_index == 1

    def test_generate_raw_returns_dict(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMQuiz())
        result = _run(gen.generate_raw(query="Python"))
        assert isinstance(result, dict)
        assert result["game_type"] == "quiz"

    def test_generate_includes_query_topic_in_prompt(self, rag_pipeline):
        llm = _SpyLLMQuiz()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        _run(gen.generate(query="fotosintesis", game_type="quiz", language="es"))

        assert llm.prompts
        assert "Focus specifically on this topic/request: fotosintesis" in llm.prompts[0]


class TestGameGeneratorWordPass:

    def test_generate_word_pass(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMWordPass())
        result = _run(gen.generate(query="letters", game_type="word-pass"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "word-pass"
        assert isinstance(result.game, WordPassGame)
        assert len(result.game.words) == 2


class TestGameGeneratorTrueFalse:

    def test_generate_true_false(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMTrueFalse())
        result = _run(gen.generate(query="sky", game_type="true_false"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "true_false"
        assert isinstance(result.game, TrueFalseGame)
        assert len(result.game.statements) == 1


class TestGameGeneratorErrorHandling:

    def test_bad_llm_output_raises_value_error(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBadOutput())
        with pytest.raises(ValueError, match="Failed to extract JSON"):
            _run(gen.generate(query="anything"))

    def test_retries_once_and_recovers_when_first_output_is_invalid(self, rag_pipeline):
        llm = _MockLLMRetryThenValid()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(gen.generate(query="math", game_type="quiz"))

        assert envelope.game_type == "quiz"
        assert envelope.game.title == "Recovered Quiz"
        assert len(llm.calls) == 2
        first_prompt, first_tokens = llm.calls[0]
        second_prompt, second_tokens = llm.calls[1]
        assert first_tokens == 1024
        assert second_tokens > first_tokens
        assert second_prompt.startswith(first_prompt)

    def test_raises_after_retry_if_both_outputs_are_invalid(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBadOutput())

        with pytest.raises(ValueError, match="after retry"):
            _run(gen.generate(query="anything"))

    def test_normalizes_malformed_quiz_payload(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMMalformedQuizPayload(),
        )

        with pytest.raises(ValueError, match="missing text"):
            _run(gen.generate(query="x", game_type="quiz"))

    def test_uses_fallback_title_when_missing(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMMalformedQuizPayload(),
        )

        with pytest.raises(ValueError, match="missing text"):
            _run(gen.generate(query="x", game_type="quiz"))

        normalized = gen._normalize_generated_payload(
            data={
                "game_type": "quiz",
                "title": "",
                "questions": [
                    {
                        "question": "Pregunta valida",
                        "options": ["A", "B"],
                        "correct_index": 0,
                        "explanation": "Explicacion breve.",
                    }
                ],
            },
            game_type="quiz",
            language="es",
            topic=None,
            difficulty_percentage=50,
        )
        assert normalized["title"] == "Quiz educativo"

    def test_retries_once_and_recovers_when_json_is_semantically_invalid(self, rag_pipeline):
        llm = _MockLLMSemanticRetryThenValid()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(gen.generate(query="photosynthesis", game_type="quiz"))

        assert envelope.game_type == "quiz"
        assert envelope.game.title == "Photosynthesis Basics"
        assert len(llm.calls) == 2
        first_prompt, first_tokens = llm.calls[0]
        second_prompt, second_tokens = llm.calls[1]
        assert second_tokens > first_tokens
        assert second_prompt.startswith(first_prompt)
        assert "missing text" in second_prompt
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert gen.last_run_metrics["semantic_retry_error"] == "Quiz question 0 is missing text"

    def test_retries_when_quiz_contains_off_topic_question(self, rag_pipeline):
        llm = _MockLLMOffTopicRetryThenValid()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(gen.generate(query="fotosintesis", game_type="quiz", language="es"))

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 2
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert "off-topic" in gen.last_run_metrics["semantic_retry_error"]

    def test_prunes_off_topic_retry_residue_when_retry_still_contains_noise(self, rag_pipeline):
        class _MockLLMRetryStillMixed:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
                self.calls.append((prompt, max_tokens))
                return json.dumps(
                    {
                        "game_type": "quiz",
                        "title": "Fotosintesis",
                        "questions": [
                            {
                                "question": "Que gas absorben las plantas para llevar a cabo la fotosintesis?",
                                "options": [
                                    "Dioxido de carbono",
                                    "Oxigeno",
                                    "Helio",
                                    "Nitrogeno",
                                ],
                                "correct_index": 0,
                                "explanation": "La fotosintesis utiliza dioxido de carbono junto con agua y luz solar.",
                            },
                            {
                                "question": "Cual es la capital de Francia?",
                                "options": ["Madrid", "Paris", "Roma", "Berlin"],
                                "correct_index": 1,
                                "explanation": "Paris es la capital francesa.",
                            },
                        ],
                    }
                )

        llm = _MockLLMRetryStillMixed()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(gen.generate(query="fotosintesis", game_type="quiz", language="es"))

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith("Que gas absorben")
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_prunes_off_topic_items_when_retry_is_skipped_after_slow_first_call(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMLongOffTopicMixed())

        async def _fake_generate_json_with_retry(prompt: str, max_tokens: int):
            raw_output = await gen.llm_client.generate(prompt, max_tokens=max_tokens)
            return raw_output, raw_output, {
                "llm_total_ms": 95_000.0,
                "parse_total_ms": 1.0,
                "retry_used": False,
            }

        gen._generate_json_with_retry = _fake_generate_json_with_retry  # type: ignore[method-assign]

        envelope = _run(gen.generate(query="fotosintesis", game_type="quiz", language="es"))

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith("Que gas absorben")
        assert gen.last_run_metrics["semantic_retry_used"] is False
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_accepts_inflected_topic_matches(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMTopicRootMatch())

        envelope = _run(gen.generate(query="matematicas", game_type="quiz", language="es"))

        assert envelope.game_type == "quiz"
        assert envelope.game.questions[0].question.startswith("Que propiedad matematica")

    def test_skips_lexical_topic_alignment_for_instruction_heavy_broad_category_queries(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMBroadCategoryTopic())

        envelope = _run(
            gen.generate(
                query=(
                    "Science & Nature conceptos esenciales comparativa historica "
                    "preguntas con contexto trivia es opcion multiple evitar verdadero falso"
                ),
                game_type="quiz",
                language="es",
            )
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert gen.last_run_metrics["semantic_retry_used"] is False

    def test_generate_passes_metadata_preferences_to_rag(self):
        class _SpyPipeline:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def build_context(self, query: str, top_k: int | None = None, **kwargs) -> str:
                self.calls.append({"query": query, "top_k": top_k, **kwargs})
                return "context"

        pipeline = _SpyPipeline()
        gen = GameGenerator(rag_pipeline=pipeline, llm_client=_MockLLMQuiz())

        _run(gen.generate(query="Historia", game_type="quiz", language="es"))

        assert pipeline.calls
        assert pipeline.calls[0]["metadata_preferences"] == {
            "language": "es",
            "game_type": "quiz",
        }
