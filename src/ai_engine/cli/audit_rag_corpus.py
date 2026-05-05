"""Audit curated RAG corpus coverage and print recommendations."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ai_engine.examples.rag_audit import audit_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit ai-engine curated RAG corpus coverage."
    )
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low"],
        default=None,
        help="Only print recommendations at this priority level.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_corpus()
    if args.priority:
        report = {
            **report,
            "recommendations": [
                item
                for item in report["recommendations"]
                if item.get("priority") == args.priority
            ],
        }

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    print(f"RAG corpus documents: {report['documents']}")
    print(f"Kinds: {_format_counter(report['kinds'])}")
    print(f"Languages: {_format_counter(report['languages'])}")
    print(f"Game types: {_format_counter(report['game_types'])}")
    print("\nTop recommendations:")
    for recommendation in report["recommendations"][:20]:
        print(
            f"- [{recommendation['priority']}] {recommendation['area']}: "
            f"{recommendation['message']}"
        )
    if len(report["recommendations"]) > 20:
        print(f"... {len(report['recommendations']) - 20} more recommendations hidden")
    return 0


def _format_counter(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in values.items())


if __name__ == "__main__":
    raise SystemExit(main())
