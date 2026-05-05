from __future__ import annotations

import json

from ai_engine.cli import audit_rag_corpus


def test_audit_rag_corpus_cli_outputs_json(capsys) -> None:
    exit_code = audit_rag_corpus.main(["--format", "json", "--priority", "high"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["documents"] >= 1
    assert all(item["priority"] == "high" for item in output["recommendations"])


def test_audit_rag_corpus_cli_outputs_text(capsys) -> None:
    exit_code = audit_rag_corpus.main(["--format", "text"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RAG corpus documents:" in output
    assert "Top recommendations:" in output
