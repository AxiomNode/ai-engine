"""Structured planning helpers for grounded game generation.

The planning layer is intentionally deterministic. It turns retrieved RAG
documents into a compact intermediate representation that can be embedded in
prompts and inspected later without changing public API contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_engine.rag.document import Document


@dataclass(frozen=True)
class GenerationPlanEvidence:
    """One evidence item selected from retrieved documents."""

    source_label: str
    excerpt: str


@dataclass(frozen=True)
class GenerationPlan:
    """Compact, typed intermediate representation for prompt grounding."""

    topic: str
    game_type: str
    language: str
    source_labels: tuple[str, ...]
    evidence: tuple[GenerationPlanEvidence, ...]

    def to_prompt_block(self) -> str:
        lines = [
            "Generation plan:",
            f"- topic: {self.topic}",
            f"- game_type: {self.game_type}",
            f"- language: {self.language}",
        ]
        if self.source_labels:
            lines.append("- sources: " + "; ".join(self.source_labels))
        if self.evidence:
            lines.append("- evidence:")
            for index, item in enumerate(self.evidence, start=1):
                lines.append(f"  {index}. {item.source_label}: {item.excerpt}")
        return "\n".join(lines)


def build_generation_plan(
    *,
    topic: str,
    game_type: str,
    language: str,
    documents: list[Document],
    max_evidence: int = 3,
) -> GenerationPlan | None:
    """Build a deterministic intermediate plan from retrieved documents."""
    normalized_topic = str(topic or "").strip()
    if not normalized_topic or not documents:
        return None

    evidence: list[GenerationPlanEvidence] = []
    source_labels: list[str] = []

    for document in documents[: max(1, max_evidence)]:
        source_label = _build_source_label(document)
        excerpt = _build_excerpt(document.content)
        if source_label and source_label not in source_labels:
            source_labels.append(source_label)
        if excerpt:
            evidence.append(
                GenerationPlanEvidence(
                    source_label=source_label or "source",
                    excerpt=excerpt,
                )
            )

    if not evidence and not source_labels:
        return None

    return GenerationPlan(
        topic=normalized_topic,
        game_type=str(game_type or "quiz").strip() or "quiz",
        language=str(language or "es").strip() or "es",
        source_labels=tuple(source_labels),
        evidence=tuple(evidence),
    )


def _build_source_label(document: Document) -> str:
    metadata = document.metadata or {}
    parts: list[str] = []
    for key in ("source", "subject", "topic", "category", "kind", "language"):
        value = metadata.get(key)
        if value:
            parts.append(f"{key}={value}")
    if not parts and document.doc_id:
        parts.append(str(document.doc_id))
    return " | ".join(parts)


def _build_excerpt(content: str, max_chars: int = 220) -> str:
    normalized = " ".join(str(content or "").split())
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."