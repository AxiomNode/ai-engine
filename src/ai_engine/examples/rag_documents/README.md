# RAG Documents

This folder contains versioned English-only corpus files loaded by
`ai_engine.examples.rag_seed` and injected into the runtime RAG pipeline.

## File Layers

- `educational_seed.jsonl`: first gold factual resources.
- `educational_expansion.jsonl`: broad coverage for target categories.
- `category_depth_expansion.jsonl`: extra depth for every target category.
- `approved_game_examples.jsonl`: approved examples for concrete game schemas.
- `approved_game_examples_depth.jsonl`: category-completion approved examples.

## Rules

- Keep one JSON object per line.
- Keep all runtime entries in English with `metadata.language = "en"`.
- Use stable `doc_id` values; do not reuse an ID for different content.
- Factual resources use `metadata.kind = "educational_resource"`.
- Approved examples use `metadata.kind = "game_example"` and must validate
  against `GameEnvelope` through the tests.
- Commit JSONL sources, not generated Chroma database files.

## Validation

From `src/`:

```bash
pytest tests/test_examples/test_rag_seed.py tests/test_examples/test_rag_audit.py
python -m ai_engine.cli.audit_rag_corpus --format json
python -m ai_engine.cli.evaluate_rag_quality --format text
```