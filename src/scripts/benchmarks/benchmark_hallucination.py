"""Hallucination & content-quality benchmark for game generation.

Sends real generation requests to a running ai-engine instance and/or
analyses cached generations from a local JSON file, evaluating each
response for structural completeness and hallucination indicators.

Usage — online (needs running ai-engine + llama-server):
    python scripts/benchmarks/benchmark_hallucination.py --base-url http://localhost:7001 --api-key KEY

Usage — offline (analyses data/generation_cache.json):
    python scripts/benchmarks/benchmark_hallucination.py --offline

Both modes can be combined: --offline also runs online tests if --base-url
is reachable.

Metrics collected per sample:
  - structural_valid:  Has required fields and non-empty content arrays
  - language_match:    Title/content appears to be in the requested language
  - category_relevant: Title/content has some relation to the category
  - min_items_met:     >= N questions/words generated (quiz=3, wordpass=10)
  - no_placeholder:    No obvious filler like "Pregunta de opcion multiple"
  - latency_ms:        Round-trip time (0 for offline)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
except ImportError:
    raise SystemExit("Standard library urllib required")

# Default paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_FILE = _SCRIPT_DIR.parent.parent / "data" / "generation_cache.json"


# ── Language detection heuristics ─────────────────────────────────────

_LANG_MARKERS: dict[str, list[str]] = {
    "es": ["el", "la", "los", "las", "del", "que", "con", "una", "por", "para"],
    "en": ["the", "is", "are", "was", "were", "of", "and", "for", "with", "this"],
    "fr": ["le", "la", "les", "des", "est", "sont", "une", "avec", "pour", "dans"],
    "de": ["der", "die", "das", "ist", "und", "ein", "eine", "mit", "fur", "von"],
    "it": ["il", "la", "le", "di", "che", "una", "con", "per", "sono", "del"],
}

PLACEHOLDER_PATTERNS = [
    r"pregunta de opci[oó]n m[uú]ltiple",
    r"opcion [A-D]",
    r"placeholder",
    r"example question",
    r"sample answer",
    r"\bTODO\b",
    r"\bFIXME\b",
]


def _detect_language(text: str) -> str | None:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    best, best_score = None, 0
    for lang, markers in _LANG_MARKERS.items():
        score = sum(1 for m in markers if m in words)
        if score > best_score:
            best, best_score = lang, score
    return best if best_score >= 2 else None


def _has_placeholder(text: str) -> bool:
    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class SampleResult:
    game_type: str
    language: str
    category_id: str
    category_name: str
    structural_valid: bool = False
    language_match: bool = False
    category_relevant: bool = False
    min_items_met: bool = False
    no_placeholder: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    item_count: int = 0


@dataclass
class BenchmarkSummary:
    total: int = 0
    structural_valid: int = 0
    language_match: int = 0
    category_relevant: int = 0
    min_items_met: int = 0
    no_placeholder: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    avg_items: float = 0.0
    results: list[SampleResult] = field(default_factory=list)


# ── Categories & languages to test ───────────────────────────────────

SAMPLE_CATEGORIES = [
    ("9", "General Knowledge"),
    ("11", "Entertainment: Film"),
    ("17", "Science & Nature"),
    ("21", "Sports"),
    ("23", "History"),
]

SAMPLE_LANGUAGES = ["es", "en"]

GAME_TYPES = ["quiz", "word-pass"]


# ── HTTP helper ───────────────────────────────────────────────────────

def _generate(base_url: str, game_type: str, language: str, category_id: str, api_key: str | None = None) -> tuple[dict[str, Any] | None, float, str | None]:
    url = f"{base_url}/generate"
    params = {
        "query": f"Educational content about the topic",
        "game_type": game_type,
        "language": language,
        "category_id": category_id,
        "difficulty_percentage": "50",
    }
    if game_type == "word-pass":
        params["letters"] = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z"

    body = json.dumps(params).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = Request(url, data=body, headers=headers, method="POST")

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            latency = (time.perf_counter() - t0) * 1000
            return data, latency, None
    except HTTPError as e:
        latency = (time.perf_counter() - t0) * 1000
        err_body = e.read().decode(errors="replace")[:200]
        return None, latency, f"HTTP {e.code}: {err_body}"
    except (URLError, TimeoutError) as e:
        latency = (time.perf_counter() - t0) * 1000
        return None, latency, str(e)


# ── Evaluation ────────────────────────────────────────────────────────

def _evaluate(data: dict[str, Any], game_type: str, language: str, category_name: str) -> SampleResult:
    result = SampleResult(
        game_type=game_type,
        language=language,
        category_id="",
        category_name=category_name,
    )

    game = data.get("game", data)
    if not isinstance(game, dict):
        result.error = "game field is not a dict"
        return result

    # Structural validity
    if game_type == "quiz":
        questions = game.get("questions", [])
        result.structural_valid = (
            isinstance(questions, list)
            and len(questions) > 0
            and all(
                isinstance(q, dict)
                and q.get("question")
                and isinstance(q.get("options"), list)
                and len(q.get("options", [])) >= 2
                and isinstance(q.get("correct_index"), int)
                for q in questions
            )
        )
        result.item_count = len(questions) if isinstance(questions, list) else 0
        result.min_items_met = result.item_count >= 3
        all_text = " ".join(
            q.get("question", "") + " " + " ".join(q.get("options", []))
            for q in (questions if isinstance(questions, list) else [])
        )
    elif game_type == "word-pass":
        words = game.get("words", [])
        result.structural_valid = (
            isinstance(words, list)
            and len(words) > 0
            and all(
                isinstance(w, dict)
                and w.get("letter")
                and w.get("hint")
                and w.get("answer")
                for w in words
            )
        )
        result.item_count = len(words) if isinstance(words, list) else 0
        result.min_items_met = result.item_count >= 10
        all_text = " ".join(
            w.get("hint", "") + " " + w.get("answer", "")
            for w in (words if isinstance(words, list) else [])
        )
    else:
        all_text = json.dumps(game)

    # Language match
    title = game.get("title", "")
    detected = _detect_language(title + " " + all_text)
    result.language_match = detected == language

    # Category relevance (loose: category name words appear in content)
    cat_words = set(re.findall(r"\b\w{4,}\b", category_name.lower()))
    content_words = set(re.findall(r"\b\w{4,}\b", (title + " " + all_text).lower()))
    # At least 1 category keyword in the content, OR title references the topic
    result.category_relevant = len(cat_words & content_words) > 0

    # Placeholder detection
    result.no_placeholder = not _has_placeholder(title + " " + all_text)

    return result


# ── Offline cache analysis ────────────────────────────────────────────

def _load_cache_entries(cache_path: Path) -> list[dict[str, Any]]:
    """Parse generation_cache.json and return list of payload dicts with metadata."""
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    entries: list[dict[str, Any]] = []
    for _key, entry in cache.get("entries", {}).items():
        content_raw = entry.get("content", "")
        try:
            content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
        except (json.JSONDecodeError, TypeError):
            continue
        payload = content.get("payload", content)
        if isinstance(payload, dict) and "game" in payload:
            entries.append(payload)
    return entries


def run_offline_benchmark(cache_path: Path) -> BenchmarkSummary:
    """Evaluate all cached generations without making HTTP requests."""
    entries = _load_cache_entries(cache_path)
    if not entries:
        print(f"  No valid entries found in {cache_path}")
        return BenchmarkSummary()

    summary = BenchmarkSummary()
    item_counts: list[int] = []

    total = len(entries)
    print(f"\n{'='*70}")
    print(f"  OFFLINE HALLUCINATION BENCHMARK — {total} cached generations")
    print(f"  Cache file: {cache_path}")
    print(f"{'='*70}\n")

    for idx, payload in enumerate(entries, 1):
        game_type = payload.get("game_type", "quiz")
        game = payload.get("game", {})
        title = game.get("title", "<no title>")
        topic = game.get("topic", "")

        # Infer language from sdk_payload metadata if available
        sdk = payload.get("sdk_payload", {})
        meta = sdk.get("metadata", {})
        lang = meta.get("language", "")
        if not lang:
            # Fallback: detect from content
            lang = _detect_language(title + " " + topic) or "es"

        label = f"[{idx}/{total}] {game_type} | {lang} | {topic or title}"
        print(f"  {label[:72]:<72} ", end="", flush=True)

        sr = _evaluate(payload, game_type, lang, topic or title)
        sr.category_id = ""
        sr.category_name = topic or title
        sr.latency_ms = 0

        summary.total += 1

        if sr.structural_valid:
            summary.structural_valid += 1
        if sr.language_match:
            summary.language_match += 1
        if sr.category_relevant:
            summary.category_relevant += 1
        if sr.min_items_met:
            summary.min_items_met += 1
        if sr.no_placeholder:
            summary.no_placeholder += 1
        item_counts.append(sr.item_count)

        flags = []
        if not sr.structural_valid:
            flags.append("STRUCT")
        if not sr.language_match:
            flags.append("LANG")
        if not sr.category_relevant:
            flags.append("CAT")
        if not sr.min_items_met:
            flags.append("ITEMS")
        if not sr.no_placeholder:
            flags.append("PLACEHOLDER")

        status = "OK" if not flags else f"FAIL[{','.join(flags)}]"
        print(f"{status} ({sr.item_count} items)")

        summary.results.append(sr)

    summary.avg_items = sum(item_counts) / len(item_counts) if item_counts else 0
    _print_summary(summary)
    return summary


# ── Main ──────────────────────────────────────────────────────────────

def run_benchmark(base_url: str, samples_per_combo: int = 1, api_key: str | None = None) -> BenchmarkSummary:
    summary = BenchmarkSummary()
    latencies: list[float] = []
    item_counts: list[int] = []

    combos = [
        (gt, lang, cid, cname)
        for gt in GAME_TYPES
        for lang in SAMPLE_LANGUAGES
        for cid, cname in SAMPLE_CATEGORIES
    ]

    total = len(combos) * samples_per_combo
    print(f"\n{'='*70}")
    print(f"  HALLUCINATION BENCHMARK — {total} generations")
    print(f"  ai-engine: {base_url}")
    print(f"{'='*70}\n")

    idx = 0
    for gt, lang, cid, cname in combos:
        for _ in range(samples_per_combo):
            idx += 1
            label = f"[{idx}/{total}] {gt} | {lang} | {cname}"
            print(f"  {label} ... ", end="", flush=True)

            data, latency, error = _generate(base_url, gt, lang, cid, api_key=api_key)
            summary.total += 1
            latencies.append(latency)

            if error or data is None:
                summary.errors += 1
                sr = SampleResult(
                    game_type=gt, language=lang,
                    category_id=cid, category_name=cname,
                    latency_ms=latency, error=error,
                )
                summary.results.append(sr)
                print(f"ERROR ({latency:.0f}ms) — {error}")
                continue

            sr = _evaluate(data, gt, lang, cname)
            sr.category_id = cid
            sr.latency_ms = latency

            if sr.structural_valid:
                summary.structural_valid += 1
            if sr.language_match:
                summary.language_match += 1
            if sr.category_relevant:
                summary.category_relevant += 1
            if sr.min_items_met:
                summary.min_items_met += 1
            if sr.no_placeholder:
                summary.no_placeholder += 1
            item_counts.append(sr.item_count)

            flags = []
            if not sr.structural_valid:
                flags.append("STRUCT")
            if not sr.language_match:
                flags.append("LANG")
            if not sr.category_relevant:
                flags.append("CAT")
            if not sr.min_items_met:
                flags.append("ITEMS")
            if not sr.no_placeholder:
                flags.append("PLACEHOLDER")

            status = "OK" if not flags else f"FAIL[{','.join(flags)}]"
            print(f"{status} ({sr.item_count} items, {latency:.0f}ms)")

            summary.results.append(sr)

    summary.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    summary.avg_items = sum(item_counts) / len(item_counts) if item_counts else 0

    _print_summary(summary)
    return summary


def _print_summary(s: BenchmarkSummary) -> None:
    valid = s.total - s.errors
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total requests:      {s.total}")
    print(f"  Successful:          {valid}  ({_pct(valid, s.total)})")
    print(f"  Errors (HTTP/net):   {s.errors}  ({_pct(s.errors, s.total)})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Structural valid:    {s.structural_valid}/{valid}  ({_pct(s.structural_valid, valid)})")
    print(f"  Language match:      {s.language_match}/{valid}  ({_pct(s.language_match, valid)})")
    print(f"  Category relevant:   {s.category_relevant}/{valid}  ({_pct(s.category_relevant, valid)})")
    print(f"  Min items met:       {s.min_items_met}/{valid}  ({_pct(s.min_items_met, valid)})")
    print(f"  No placeholders:     {s.no_placeholder}/{valid}  ({_pct(s.no_placeholder, valid)})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg latency:         {s.avg_latency_ms:.0f}ms")
    print(f"  Avg items/game:      {s.avg_items:.1f}")

    # Hallucination score: 1 - (language_match + category_relevant + no_placeholder) / (3 * valid)
    if valid > 0:
        quality = (s.language_match + s.category_relevant + s.no_placeholder) / (3 * valid)
        hallucination_rate = 1 - quality
        print(f"  ─────────────────────────────────────────")
        print(f"  Content quality:     {quality*100:.1f}%")
        print(f"  Hallucination rate:  {hallucination_rate*100:.1f}%")
    print(f"{'='*70}\n")

    # Per-type breakdown
    for gt in GAME_TYPES:
        gt_results = [r for r in s.results if r.game_type == gt and r.error is None]
        if not gt_results:
            continue
        n = len(gt_results)
        print(f"  [{gt}] {n} samples:")
        print(f"    Structural:   {sum(1 for r in gt_results if r.structural_valid)}/{n}")
        print(f"    Lang match:   {sum(1 for r in gt_results if r.language_match)}/{n}")
        print(f"    Cat relevant: {sum(1 for r in gt_results if r.category_relevant)}/{n}")
        print(f"    Avg items:    {sum(r.item_count for r in gt_results)/n:.1f}")
        print()


def _pct(num: int, den: int) -> str:
    return f"{num/den*100:.1f}%" if den else "N/A"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination benchmark for ai-engine")
    parser.add_argument("--base-url", default="http://localhost:7001", help="ai-engine base URL")
    parser.add_argument("--api-key", default="dev_games_test_key_2026", help="X-API-Key header value")
    parser.add_argument("--samples", type=int, default=1, help="Samples per combination")
    parser.add_argument("--offline", action="store_true", help="Analyse data/generation_cache.json instead of live requests")
    parser.add_argument("--cache-file", type=str, default=None, help="Path to cache JSON (default: data/generation_cache.json)")
    args = parser.parse_args()

    if args.offline:
        cache_path = Path(args.cache_file) if args.cache_file else _CACHE_FILE
        if not cache_path.exists():
            print(f"ERROR: cache file not found at {cache_path}")
            sys.exit(1)
        summary = run_offline_benchmark(cache_path)
    else:
        summary = run_benchmark(args.base_url, args.samples, api_key=args.api_key)

    # Exit code: 1 if hallucination rate > 50%
    valid = summary.total - summary.errors
    if valid > 0:
        quality = (summary.language_match + summary.category_relevant + summary.no_placeholder) / (3 * valid)
        sys.exit(0 if quality >= 0.5 else 1)
    else:
        sys.exit(1)
