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
        markers = (
            "Focus specifically on this topic/request:",
            "- Topic:",
        )
        for line in prompt.splitlines():
            for marker in markers:
                if marker in line:
                    topic = line.split(marker, 1)[1].strip() or topic
                    break
            else:
                continue
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


class _MockLLMRecoverableQuizAliases:
    """Mock LLM that returns recoverable quiz field aliases."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            {
                "game": {
                    "game_type": "quiz",
                    "title": "Alias Quiz",
                    "entries": [
                        {
                            "question": "What process powers plant growth?",
                            "options": [
                                {"text": "A) Respiration"},
                                {"text": "B) Photosynthesis"},
                                {"text": "C) Condensation"},
                            ],
                            "correct_answer": "B",
                            "explanation": "Photosynthesis converts light into chemical energy.",
                        }
                    ],
                }
            }
        )


class _MockLLMQuizListPayload:
    """Mock LLM that returns a top-level list of quiz questions."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            [
                {
                    "question": "Which organelle performs photosynthesis?",
                    "options": ["Nucleus", "Chloroplast", "Ribosome", "Vacuole"],
                    "correct_answer": "Chloroplast",
                    "explanation": "Photosynthesis happens in chloroplasts.",
                }
            ]
        )


class _MockLLMQuizMalformedRetry:
    """Mock LLM that returns an off-topic quiz first, then truncated near-valid JSON."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "game_type": "quiz",
                    "title": "Teorema de Pitagoras",
                    "questions": [
                        {
                            "question": "Que relacion establece el teorema de Pitagoras en un triangulo rectangulo?",
                            "options": [
                                "La hipotenusa y los catetos",
                                "La capital de Francia",
                                "La tabla periodica",
                                "El sistema solar",
                            ],
                            "correct_index": 0,
                            "explanation": "Relaciona la hipotenusa con los catetos.",
                        },
                        {
                            "question": "Cual es la capital de Francia?",
                            "options": ["Paris", "Roma", "Berlin", "Lisboa"],
                            "correct_index": 0,
                            "explanation": "Paris es la capital de Francia.",
                        },
                    ],
                }
            )
        return """{
  \"game_type\": \"quiz\",
  \"title\": \"Teorema de Pitagoras\",
  \"questions\": [
    {
      \"question\": \"Segun el teorema de Pitagoras, si los catetos miden 3 y 4, cuanto mide la hipotenusa?\",
      \"options\": [\"5\", \"6\", \"7\", \"8\"],
      \"correct_index\": 0,
      \"explanation\": \"Porque 3^2 + 4^2 = 5^2\"
    },
    {
      \"question\": \"En el teorema de Pitagoras, que lado es el mas largo del triangulo rectangulo?\",
      \"options\": [\"Hipotenusa\", \"Cateto menor\", \"Base\", \"Altura\"],
      \"correct_index\": 0,
      \"explanation\": \"La hipotenusa es el lado opuesto al angulo recto y el mas largo\"
    }
  ]"""


class _MockLLMSingleQuizObject:
    """Mock LLM that returns a single quiz question object."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            {
                "question": "How many players per team are on court in basketball?",
                "options": ["4", "5", "6", "7"],
                "correct_answer": "5",
                "explanation": "Each team plays with five players on the court.",
            }
        )


class _MockLLMRecoverableWordPassAliases:
    """Mock LLM that returns recoverable word-pass field aliases."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return json.dumps(
            {
                "game_type": "word-pass",
                "title": "Alias Rosco",
                "entries": [
                    {
                        "letter": "A",
                        "clue": "Chemical element essential for water.",
                        "solution": "Agua",
                        "relation": "starts_with",
                    },
                    {
                        "letter": "B",
                        "definition": "Term that contains the letter B.",
                        "word": "Abeto",
                        "relation": "contains",
                    },
                ],
            }
        )


class _MockLLMWordPassMalformedRetry:
    """Mock LLM that returns empty words first, then loosely formatted entries."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append((prompt, max_tokens))
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "game_type": "word-pass",
                    "title": "Photosynthesis",
                    "words": [],
                }
            )
        return """{
  \"game_type\": \"word-pass\",
  \"title\": \"Photosynthesis\",
  \"words\": [
    {
      \"letter\": \"C\",
      \"hint\": \"A green pigment essential for photosynthesis in plants.\",
      \"answer\": \"Chlorophyll\",
      \"starts_with\": true,
    \"letter\": \"P\",
      \"hint\": \"The process by which plants convert light into chemical energy.\",
      \"answer\": \"Photosynthesis\",
      \"starts_with\": true
  ]
}"""


class _MockLLMWordPassAlwaysBroken:
    """Mock LLM that never produces a valid word-pass payload."""

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return "[\n \t\t\t\t\t"


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


class _MockLLMOffTopicRetryThenBroken:
    """Mock LLM that first returns a salvageable off-topic quiz, then a broken retry."""

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
                            "question": "Cual es la capital de Francia?",
                            "options": ["Madrid", "Paris", "Roma", "Berlin"],
                            "correct_index": 1,
                            "explanation": "Paris es la capital de Francia.",
                        },
                    ],
                }
            )
        return json.dumps(
            {
                "game_type": "quiz",
                "title": "Fotosintesis",
                "questions": [],
            }
        )


class _MockLLMOffTopicRetryThenUnparseable:
    """Mock LLM that first returns a salvageable quiz, then truncated junk."""

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
                            "question": "Cual es la capital de Francia?",
                            "options": ["Madrid", "Paris", "Roma", "Berlin"],
                            "correct_index": 1,
                            "explanation": "Paris es la capital de Francia.",
                        },
                    ],
                }
            )
        return "[1,2,3,4,5,6,7,8,9,10"


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
        assert "- Topic: fotosintesis" in llm.prompts[0]


class TestGameGeneratorWordPass:

    def test_generate_word_pass(self, rag_pipeline):
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=_MockLLMWordPass())
        result = _run(gen.generate(query="letters", game_type="word-pass"))

        assert isinstance(result, GameEnvelope)
        assert result.game_type == "word-pass"
        assert isinstance(result.game, WordPassGame)
        assert len(result.game.words) == 2

    def test_generate_word_pass_recovers_alias_fields(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMRecoverableWordPassAliases(),
        )

        result = _run(gen.generate(query="letters", game_type="word-pass"))

        assert result.game.words[0].answer == "Agua"
        assert result.game.words[0].starts_with is True
        assert result.game.words[1].answer == "Abeto"
        assert result.game.words[1].starts_with is False

    def test_generate_word_pass_salvages_malformed_retry_entries(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMWordPassMalformedRetry(),
        )

        result = _run(
            gen.generate(query="photosynthesis", game_type="word-pass", language="en")
        )

        assert len(result.game.words) == 2
        assert result.game.words[0].letter == "C"
        assert result.game.words[1].answer == "Photosynthesis"

    def test_generate_word_pass_enriches_topic_signal_when_hint_is_too_generic(
        self, rag_pipeline
    ):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMRecoverableWordPassAliases(),
        )

        result = _run(
            gen.generate(query="renaissance art", game_type="word-pass", language="en")
        )

        assert "renaissance art" in result.game.words[0].hint.lower()

    def test_generate_word_pass_uses_fallback_when_model_never_returns_valid_json(
        self, rag_pipeline
    ):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMWordPassAlwaysBroken(),
        )

        result = _run(
            gen.generate(query="renaissance art", game_type="word-pass", language="en")
        )

        assert len(result.game.words) >= 1
        assert "renaissance art" in result.game.words[0].hint.lower()
        assert gen.last_run_metrics["word_pass_fallback_used"] is True


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

    def test_recovers_quiz_alias_fields_and_nested_game_payload(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMRecoverableQuizAliases(),
        )

        envelope = _run(gen.generate(query="photosynthesis", game_type="quiz"))

        assert envelope.game.title == "Alias Quiz"
        assert envelope.game.questions[0].correct_index == 1
        assert envelope.game.questions[0].options[1] == "Photosynthesis"

    def test_recovers_top_level_quiz_list_payload(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMQuizListPayload(),
        )

        envelope = _run(gen.generate(query="photosynthesis", game_type="quiz"))

        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].correct_index == 1

    def test_recovers_single_quiz_object_payload(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline,
            llm_client=_MockLLMSingleQuizObject(),
        )

        envelope = _run(gen.generate(query="basketball", game_type="quiz"))

        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].correct_index == 1

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

    def test_retries_once_and_recovers_when_json_is_semantically_invalid(
        self, rag_pipeline
    ):
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
        assert (
            gen.last_run_metrics["semantic_retry_error"]
            == "Quiz question 0 is missing text"
        )

    def test_retries_when_quiz_contains_off_topic_question(self, rag_pipeline):
        llm = _MockLLMOffTopicRetryThenValid()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(
            gen.generate(query="fotosintesis", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 2
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert "off-topic" in gen.last_run_metrics["semantic_retry_error"]

    def test_prunes_off_topic_retry_residue_when_retry_still_contains_noise(
        self, rag_pipeline
    ):
        class _MockLLMRetryStillMixed:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            async def generate(
                self, prompt: str, max_tokens: int = 256, **kwargs
            ) -> str:
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

        envelope = _run(
            gen.generate(query="fotosintesis", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith("Que gas absorben")
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_falls_back_to_salvaged_initial_payload_when_retry_is_worse(
        self, rag_pipeline
    ):
        llm = _MockLLMOffTopicRetryThenBroken()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(
            gen.generate(query="fotosintesis", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith(
            "Que proceso usan las plantas"
        )
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert (
            gen.last_run_metrics["semantic_retry_fallback_to_initial_payload"] is True
        )
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_falls_back_to_salvaged_initial_payload_when_retry_is_unparseable(
        self, rag_pipeline
    ):
        llm = _MockLLMOffTopicRetryThenUnparseable()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(
            gen.generate(query="fotosintesis", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith(
            "Que proceso usan las plantas"
        )
        assert len(llm.calls) == 2
        assert gen.last_run_metrics["semantic_retry_used"] is True
        assert (
            gen.last_run_metrics["semantic_retry_fallback_to_initial_payload"] is True
        )
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_salvages_malformed_quiz_retry_output(self, rag_pipeline):
        llm = _MockLLMQuizMalformedRetry()
        gen = GameGenerator(rag_pipeline=rag_pipeline, llm_client=llm)

        envelope = _run(
            gen.generate(query="teorema de pitagoras", game_type="quiz", language="es")
        )

        assert len(envelope.game.questions) == 2
        assert envelope.game.questions[0].correct_index == 0
        assert "pitagoras" in envelope.game.questions[0].question.lower()

    def test_prunes_off_topic_items_when_retry_is_skipped_after_slow_first_call(
        self, rag_pipeline
    ):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline, llm_client=_MockLLMLongOffTopicMixed()
        )

        async def _fake_generate_json_with_retry(prompt: str, max_tokens: int):
            raw_output = await gen.llm_client.generate(prompt, max_tokens=max_tokens)
            return (
                raw_output,
                raw_output,
                {
                    "llm_total_ms": 95_000.0,
                    "parse_total_ms": 1.0,
                    "retry_used": False,
                },
            )

        gen._generate_json_with_retry = _fake_generate_json_with_retry  # type: ignore[method-assign]

        envelope = _run(
            gen.generate(query="fotosintesis", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert len(envelope.game.questions) == 1
        assert envelope.game.questions[0].question.startswith("Que gas absorben")
        assert gen.last_run_metrics["semantic_retry_used"] is False
        assert gen.last_run_metrics["topic_pruning_used"] is True
        assert gen.last_run_metrics["topic_pruning_removed_items"] == 1

    def test_accepts_inflected_topic_matches(self, rag_pipeline):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline, llm_client=_MockLLMTopicRootMatch()
        )

        envelope = _run(
            gen.generate(query="matematicas", game_type="quiz", language="es")
        )

        assert envelope.game_type == "quiz"
        assert envelope.game.questions[0].question.startswith(
            "Que propiedad matematica"
        )

    def test_skips_lexical_topic_alignment_for_instruction_heavy_broad_category_queries(
        self, rag_pipeline
    ):
        gen = GameGenerator(
            rag_pipeline=rag_pipeline, llm_client=_MockLLMBroadCategoryTopic()
        )

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

            def build_context(
                self, query: str, top_k: int | None = None, **kwargs
            ) -> str:
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
