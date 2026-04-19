"""Curated quality benchmark for real generation queries.

Runs a fixed matrix of topical generation requests against a live ai-engine
instance and evaluates:
  - HTTP success rate
  - structural validity
  - topical alignment at item level
  - placeholder/filler rate
  - latency

Usage:
    python src/scripts/benchmarks/benchmark_generation_quality.py \
      --base-url http://localhost:8001 \
      --api-key KEY
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PLACEHOLDER_PATTERNS = [
    r"multiple choice question",
    r"pregunta de opcion multiple",
    r"placeholder",
    r"sample answer",
    r"example question",
    r"\bTODO\b",
    r"\bFIXME\b",
]

STOPWORDS = {
    "about",
    "con",
    "de",
    "del",
    "el",
    "en",
    "la",
    "las",
    "los",
    "para",
    "por",
    "que",
    "sobre",
    "the",
    "una",
    "y",
}

MIN_KEYWORD_LEN = 4
ROOT_PREFIX_LEN = 6


@dataclass(frozen=True)
class BenchmarkCase:
    game_type: str
    language: str
    category_id: str
    query: str


@dataclass
class CaseResult:
    game_type: str
    language: str
    category_id: str
    query: str
    status_code: int
    latency_ms: float
    structural_valid: bool
    topical_alignment: bool
    placeholder_free: bool
    item_count: int
    topical_item_hits: int
    error: str | None = None


CASES: list[BenchmarkCase] = [
    BenchmarkCase("quiz", "es", "17", "fotosintesis"),
    BenchmarkCase("quiz", "es", "17", "ciclo de calvin"),
    BenchmarkCase("quiz", "es", "19", "teorema de pitagoras"),
    BenchmarkCase("quiz", "es", "23", "revolucion francesa"),
    BenchmarkCase("quiz", "es", "21", "reglas del baloncesto"),
    BenchmarkCase("quiz", "en", "17", "cellular respiration"),
    BenchmarkCase("quiz", "en", "18", "binary search trees"),
    BenchmarkCase("quiz", "en", "23", "roman republic"),
    BenchmarkCase("word-pass", "es", "17", "sistema solar"),
    BenchmarkCase("word-pass", "es", "23", "independencia de mexico"),
    BenchmarkCase("word-pass", "en", "17", "photosynthesis"),
    BenchmarkCase("word-pass", "en", "25", "renaissance art"),
]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _topic_signals(query: str) -> tuple[list[str], list[str]]:
    keywords: list[str] = []
    for token in _normalize(query).split():
        if len(token) < MIN_KEYWORD_LEN or token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    roots: list[str] = []
    for keyword in keywords:
        if len(keyword) < ROOT_PREFIX_LEN:
            continue
        root = keyword[:ROOT_PREFIX_LEN]
        if root not in roots:
            roots.append(root)
    return keywords, roots


def _has_topic_signal(text: str, keywords: list[str], roots: list[str]) -> bool:
    normalized = _normalize(text)
    if any(keyword in normalized for keyword in keywords):
        return True
    tokens = normalized.split()
    return any(token.startswith(root) for root in roots for token in tokens)


def _has_placeholder(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def _request_case(base_url: str, api_key: str, case: BenchmarkCase) -> tuple[int, dict[str, Any] | None, float, str | None]:
    endpoint = f"{base_url}/generate/{case.game_type}"
    params = [
        f"query={case.query.replace(' ', '%20')}",
        f"language={case.language}",
        f"category_id={case.category_id}",
        "difficulty_percentage=50",
        "use_cache=false",
        "force_refresh=true",
    ]
    if case.game_type == "quiz":
        params.append("item_count=3")
    if case.game_type == "word-pass":
        params.append("item_count=20")
    request = Request(
        f"{endpoint}?{'&'.join(params)}",
        headers={"X-API-Key": api_key},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=1800) as response:
            payload = json.loads(response.read())
            return response.status, payload, (time.perf_counter() - started) * 1000, None
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        return exc.code, None, (time.perf_counter() - started) * 1000, body
    except (URLError, TimeoutError) as exc:
        return 0, None, (time.perf_counter() - started) * 1000, str(exc)


def _evaluate_case(case: BenchmarkCase, payload: dict[str, Any] | None, status_code: int, latency_ms: float, error: str | None) -> CaseResult:
    if payload is None:
        return CaseResult(
            game_type=case.game_type,
            language=case.language,
            category_id=case.category_id,
            query=case.query,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            structural_valid=False,
            topical_alignment=False,
            placeholder_free=False,
            item_count=0,
            topical_item_hits=0,
            error=error,
        )

    game = payload.get("game", payload)
    keywords, roots = _topic_signals(case.query)
    item_count = 0
    topical_hits = 0
    placeholder_free = True
    structural_valid = True

    if case.game_type == "quiz":
        items = game.get("questions", []) if isinstance(game, dict) else []
        if not isinstance(items, list) or not items:
            structural_valid = False
            items = []
        for item in items:
            if not isinstance(item, dict):
                structural_valid = False
                continue
            bundle = " ".join(
                [
                    str(item.get("question", "")),
                    str(item.get("explanation", "")),
                    " ".join(str(option) for option in item.get("options", [])),
                ]
            )
            if not item.get("question") or not isinstance(item.get("options"), list):
                structural_valid = False
            item_count += 1
            if _has_topic_signal(bundle, keywords, roots):
                topical_hits += 1
            if _has_placeholder(bundle):
                placeholder_free = False
    else:
        items = game.get("words", []) if isinstance(game, dict) else []
        if not isinstance(items, list) or not items:
            structural_valid = False
            items = []
        for item in items:
            if not isinstance(item, dict):
                structural_valid = False
                continue
            bundle = " ".join(
                [
                    str(item.get("letter", "")),
                    str(item.get("hint", "")),
                    str(item.get("answer", "")),
                ]
            )
            if not item.get("letter") or not item.get("hint") or not item.get("answer"):
                structural_valid = False
            item_count += 1
            if _has_topic_signal(bundle, keywords, roots):
                topical_hits += 1
            if _has_placeholder(bundle):
                placeholder_free = False

    ratio = (topical_hits / item_count) if item_count else 0.0
    required_ratio = 0.75 if case.game_type == "quiz" else 0.4
    return CaseResult(
        game_type=case.game_type,
        language=case.language,
        category_id=case.category_id,
        query=case.query,
        status_code=status_code,
        latency_ms=round(latency_ms, 2),
        structural_valid=structural_valid,
        topical_alignment=ratio >= required_ratio,
        placeholder_free=placeholder_free,
        item_count=item_count,
        topical_item_hits=topical_hits,
        error=error,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run curated generation quality benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--output",
        default=str(Path("docs") / "generation-quality-benchmark.json"),
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        print("--api-key is required", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for case in CASES:
        status_code, payload, latency_ms, error = _request_case(
            args.base_url.rstrip("/"), args.api_key, case
        )
        results.append(_evaluate_case(case, payload, status_code, latency_ms, error))

    summary = {
        "total": len(results),
        "http_success_rate": round(
            sum(1 for result in results if result.status_code == 200) / max(len(results), 1),
            4,
        ),
        "structural_valid_rate": round(
            sum(1 for result in results if result.structural_valid) / max(len(results), 1),
            4,
        ),
        "topical_alignment_rate": round(
            sum(1 for result in results if result.topical_alignment) / max(len(results), 1),
            4,
        ),
        "placeholder_free_rate": round(
            sum(1 for result in results if result.placeholder_free) / max(len(results), 1),
            4,
        ),
        "avg_latency_ms": round(
            sum(result.latency_ms for result in results) / max(len(results), 1),
            2,
        ),
        "results": [asdict(result) for result in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())