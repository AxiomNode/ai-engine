"""Audit curated RAG corpus coverage and recommend next documents."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ai_engine.examples.corpus import get_full_corpus
from ai_engine.games.catalog import get_supported_game_types

TARGET_LANGUAGES = ["en"]
TARGET_CATEGORIES = [
    "General Knowledge",
    "Entertainment: Books",
    "Entertainment: Film",
    "Entertainment: Music",
    "Entertainment: Video Games",
    "Science & Nature",
    "Science: Computers",
    "Science: Mathematics",
    "Sports",
    "Geography",
    "History",
    "Politics",
    "Art",
    "Animals",
]
MIN_RESOURCE_DOCS_PER_CATEGORY = 2
MIN_GAME_EXAMPLES_PER_TYPE = 2


@dataclass(frozen=True)
class CoverageRecommendation:
    """Actionable recommendation for filling RAG corpus gaps."""

    priority: str
    area: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "priority": self.priority,
            "area": self.area,
            "message": self.message,
        }


def audit_corpus(entries: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return coverage metrics and recommendations for the curated RAG corpus."""
    corpus = list(entries) if entries is not None else get_full_corpus()
    documents = [_normalize_entry(entry) for entry in corpus]

    kind_counts = Counter(doc["kind"] for doc in documents)
    language_counts = Counter(doc["language"] for doc in documents if doc["language"])
    category_counts = Counter(doc["category"] for doc in documents if doc["category"])
    topic_counts = Counter(doc["topic"] for doc in documents if doc["topic"])
    game_type_counts = Counter(
        doc["game_type"] for doc in documents if doc["game_type"]
    )

    resources_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_game_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    language_category_pairs: set[tuple[str, str]] = set()
    non_english_documents: list[str] = []

    for doc in documents:
        if doc["kind"] == "educational_resource" and doc["category"]:
            resources_by_category[doc["category"]].append(doc)
        if doc["kind"] == "game_example" and doc["game_type"]:
            examples_by_game_type[doc["game_type"]].append(doc)
        if doc["language"] and doc["category"]:
            language_category_pairs.add((doc["language"], doc["category"]))
        if doc["language"] and doc["language"] not in TARGET_LANGUAGES:
            non_english_documents.append(doc["doc_id"])

    recommendations = _build_recommendations(
        resources_by_category=resources_by_category,
        examples_by_game_type=examples_by_game_type,
        language_category_pairs=language_category_pairs,
        non_english_documents=non_english_documents,
    )

    return {
        "documents": len(documents),
        "kinds": dict(sorted(kind_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "topics": dict(sorted(topic_counts.items())),
        "game_types": dict(sorted(game_type_counts.items())),
        "targets": {
            "languages": TARGET_LANGUAGES,
            "categories": TARGET_CATEGORIES,
            "game_types": get_supported_game_types(),
            "min_resource_docs_per_category": MIN_RESOURCE_DOCS_PER_CATEGORY,
            "min_game_examples_per_type": MIN_GAME_EXAMPLES_PER_TYPE,
        },
        "coverage": {
            "category_resource_docs": {
                category: len(resources_by_category.get(category, []))
                for category in TARGET_CATEGORIES
            },
            "game_examples": {
                game_type: len(examples_by_game_type.get(game_type, []))
                for game_type in get_supported_game_types()
            },
            "language_category_pairs": sorted(
                f"{language}:{category}"
                for language, category in language_category_pairs
            ),
            "non_english_documents": sorted(non_english_documents),
        },
        "recommendations": [
            recommendation.to_dict() for recommendation in recommendations
        ],
    }


def _normalize_entry(entry: dict[str, Any]) -> dict[str, str]:
    raw_metadata = entry.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    return {
        "doc_id": str(entry.get("doc_id") or ""),
        "kind": str(metadata.get("kind") or "unknown"),
        "source": str(metadata.get("source") or "unknown"),
        "language": str(metadata.get("language") or ""),
        "category": str(metadata.get("category") or ""),
        "topic": str(metadata.get("topic") or metadata.get("sub_topic") or ""),
        "game_type": str(metadata.get("game_type") or ""),
    }


def _build_recommendations(
    *,
    resources_by_category: dict[str, list[dict[str, Any]]],
    examples_by_game_type: dict[str, list[dict[str, Any]]],
    language_category_pairs: set[tuple[str, str]],
    non_english_documents: list[str],
) -> list[CoverageRecommendation]:
    recommendations: list[CoverageRecommendation] = []

    if non_english_documents:
        recommendations.append(
            CoverageRecommendation(
                priority="high",
                area="language:non-english",
                message=(
                    f"Remove or migrate {len(non_english_documents)} non-English "
                    "document(s); the project runtime corpus is English-only."
                ),
            )
        )

    for category in TARGET_CATEGORIES:
        resource_count = len(resources_by_category.get(category, []))
        if resource_count == 0:
            recommendations.append(
                CoverageRecommendation(
                    priority="high",
                    area=f"category:{category}",
                    message=(
                        "Add at least two educational_resource documents with factual, "
                        "source-labeled context for this category."
                    ),
                )
            )
        elif resource_count < MIN_RESOURCE_DOCS_PER_CATEGORY:
            recommendations.append(
                CoverageRecommendation(
                    priority="medium",
                    area=f"category:{category}",
                    message=(
                        f"Add {MIN_RESOURCE_DOCS_PER_CATEGORY - resource_count} more "
                        "educational_resource document(s) to improve retrieval diversity."
                    ),
                )
            )

    for language in TARGET_LANGUAGES:
        for category in TARGET_CATEGORIES:
            if (language, category) not in language_category_pairs:
                recommendations.append(
                    CoverageRecommendation(
                        priority="medium",
                        area=f"language-category:{language}:{category}",
                        message="Add English RAG context for this category.",
                    )
                )

    for game_type in get_supported_game_types():
        example_count = len(examples_by_game_type.get(game_type, []))
        if example_count < MIN_GAME_EXAMPLES_PER_TYPE:
            recommendations.append(
                CoverageRecommendation(
                    priority="medium",
                    area=f"game_type:{game_type}",
                    message=(
                        f"Add {MIN_GAME_EXAMPLES_PER_TYPE - example_count} approved "
                        "game_example document(s) for schema/style grounding."
                    ),
                )
            )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        recommendations,
        key=lambda item: (priority_order.get(item.priority, 9), item.area),
    )
