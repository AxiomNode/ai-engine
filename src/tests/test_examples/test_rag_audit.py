from __future__ import annotations

from ai_engine.examples.rag_audit import audit_corpus


def test_audit_corpus_reports_current_coverage() -> None:
    report = audit_corpus()

    assert report["documents"] >= 20
    assert report["kinds"]["educational_resource"] >= 1
    assert report["languages"] == {"en": report["documents"]}
    assert "Science & Nature" in report["categories"]
    assert "quiz" in report["game_types"]
    assert report["targets"]["min_resource_docs_per_category"] == 2


def test_audit_corpus_recommends_missing_category_resources() -> None:
    report = audit_corpus(
        [
            {
                "doc_id": "only-quiz",
                "content": "{}",
                "metadata": {
                    "kind": "game_example",
                    "game_type": "quiz",
                    "language": "en",
                    "category": "General Knowledge",
                },
            }
        ]
    )

    high_priority_areas = {
        item["area"] for item in report["recommendations"] if item["priority"] == "high"
    }
    assert "category:Science & Nature" in high_priority_areas
    assert "category:History" in high_priority_areas


def test_audit_corpus_recommends_game_examples_by_type() -> None:
    report = audit_corpus([])

    recommendation_areas = {item["area"] for item in report["recommendations"]}
    assert "game_type:quiz" in recommendation_areas
    assert "game_type:word-pass" in recommendation_areas
    assert "game_type:true_false" in recommendation_areas


def test_audit_corpus_flags_non_english_documents() -> None:
    report = audit_corpus(
        [
            {
                "doc_id": "legacy-es",
                "content": "Contenido heredado",
                "metadata": {
                    "kind": "educational_resource",
                    "language": "es",
                    "category": "Science & Nature",
                    "topic": "legacy",
                },
            }
        ]
    )

    assert report["coverage"]["non_english_documents"] == ["legacy-es"]
    assert any(
        item["area"] == "language:non-english" for item in report["recommendations"]
    )
