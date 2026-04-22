"""Eval harness runner for ai-engine (ADR 0009).

Loads a baseline dataset and runs each case against a real
:class:`GameGenerator` instance backed by a canned LLM and a tiny
in-memory RAG pipeline. Aggregates success/latency metrics and writes a
JSON report.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai_engine.games.generator import GameGenerator
from ai_engine.games.prompts import get_prompt_version
from ai_engine.games.schemas import GameEnvelope
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.vector_store import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------


def _quiz_payload(topic: str, language: str, num_questions: int) -> dict[str, Any]:
    return {
        "game_type": "quiz",
        "title": f"Quiz {language}: {topic}",
        "questions": [
            {
                "question": f"[{language}] Q{i + 1} about {topic}?",
                "options": [
                    f"[{language}] core fact about {topic}",
                    "Unrelated option A",
                    "Unrelated option B",
                    "Unrelated option C",
                ],
                "correct_index": 0,
                "explanation": f"[{language}] Correct because it matches {topic}.",
            }
            for i in range(num_questions)
        ],
    }


def _word_pass_payload(topic: str, language: str, num_questions: int) -> dict[str, Any]:
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    return {
        "game_type": "word-pass",
        "title": f"WordPass {language}: {topic}",
        "words": [
            {
                "letter": letters[i % len(letters)],
                "hint": f"[{language}] hint about {topic} #{i + 1}",
                "answer": f"answer{i + 1}",
                "starts_with": True,
            }
            for i in range(num_questions)
        ],
    }


def _true_false_payload(
    topic: str, language: str, num_questions: int
) -> dict[str, Any]:
    return {
        "game_type": "true_false",
        "title": f"TF {language}: {topic}",
        "statements": [
            {
                "statement": f"[{language}] Statement {i + 1} about {topic}.",
                "is_true": i % 2 == 0,
                "explanation": f"[{language}] Tied to {topic}.",
            }
            for i in range(num_questions)
        ],
    }


_PAYLOAD_BUILDERS = {
    "quiz": _quiz_payload,
    "word-pass": _word_pass_payload,
    "true_false": _true_false_payload,
}


class CannedLLM:
    """Deterministic LLM stub: returns a valid JSON for the requested game."""

    def __init__(self, game_type: str, topic: str, language: str, num_questions: int):
        self.game_type = game_type
        self.topic = topic
        self.language = language
        self.num_questions = num_questions
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        self.calls += 1
        builder = _PAYLOAD_BUILDERS[self.game_type]
        payload = builder(self.topic, self.language, self.num_questions)
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Tiny RAG pipeline
# ---------------------------------------------------------------------------


class _FixedEmbedder(Embedder):
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _build_rag_pipeline(topic: str, language: str) -> RAGPipeline:
    pipeline = RAGPipeline(
        embedder=_FixedEmbedder(), vector_store=InMemoryVectorStore()
    )
    pipeline.ingest(
        [
            Document(
                content=f"Reference context about {topic} written in {language}.",
                metadata={"language": language, "game_type": "quiz"},
            )
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    game_type: str
    language: str
    prompt_version: str
    success: bool
    latency_ms: float
    items_returned: int
    title_non_empty: bool
    error: str | None = None


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.success)

    @property
    def success_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def latency_p50_ms(self) -> float:
        if not self.cases:
            return 0.0
        return float(statistics.median(c.latency_ms for c in self.cases))

    @property
    def latency_p95_ms(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_ms for c in self.cases)
        idx = max(0, int(round(0.95 * (len(latencies) - 1))))
        return float(latencies[idx])

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "success_rate": self.success_rate,
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
            },
            "cases": [asdict(c) for c in self.cases],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cases = data.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError(f"Dataset {path} must contain a top-level 'cases' list")
    return cases


def _items_in_envelope(envelope: GameEnvelope) -> int:
    game = envelope.game
    for attr in ("questions", "words", "statements"):
        seq = getattr(game, attr, None)
        if seq is not None:
            return len(seq)
    return 0


async def _run_case(case: dict[str, Any]) -> CaseResult:
    case_id = case["id"]
    game_type = case["game_type"]
    topic = case["topic"]
    language = case["language"]
    num_questions = int(case["num_questions"])
    expected_min = int(case.get("expected_min_items", 1))

    llm = CannedLLM(game_type, topic, language, num_questions)
    rag = _build_rag_pipeline(topic, language)
    generator = GameGenerator(
        rag_pipeline=rag, llm_client=llm, default_language=language
    )

    started = time.perf_counter()
    try:
        envelope = await generator.generate(
            query=topic,
            game_type=game_type,
            language=language,
            num_questions=num_questions,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        items = _items_in_envelope(envelope)
        title = (envelope.game.title or "").strip()
        success = items >= expected_min and bool(title)
        return CaseResult(
            case_id=case_id,
            game_type=game_type,
            language=language,
            prompt_version=get_prompt_version(game_type),
            success=success,
            latency_ms=latency_ms,
            items_returned=items,
            title_non_empty=bool(title),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - record any failure mode
        latency_ms = (time.perf_counter() - started) * 1000.0
        return CaseResult(
            case_id=case_id,
            game_type=game_type,
            language=language,
            prompt_version=get_prompt_version(game_type),
            success=False,
            latency_ms=latency_ms,
            items_returned=0,
            title_non_empty=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_eval(dataset_path: Path) -> EvalReport:
    """Run the eval harness against the dataset at *dataset_path*."""
    cases = load_dataset(dataset_path)
    report = EvalReport()
    for case in cases:
        report.cases.append(asyncio.run(_run_case(case)))
    return report


def write_report(report: EvalReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "baseline.yaml"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[2] / ".cache" / "eval" / "results.json"
)


def main() -> int:
    report = run_eval(DEFAULT_DATASET)
    write_report(report, DEFAULT_REPORT)
    print(json.dumps(report.to_dict()["summary"], indent=2))
    return 0 if report.success_rate == 1.0 else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
