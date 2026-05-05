# RAG Corpus Authoring Guide

Last updated: 2026-05-05.

## Purpose

This guide explains how to grow the ai-engine RAG corpus with curated documents
that improve educational game generation.

The goal is not to dump random text into a vector database. The goal is to build
a small, inspectable, high-quality corpus with metadata that lets retrieval pick
the right context for each game type, category, language, and topic.

## Recommended Corpus Layers

Use three layers, in this order:

1. `educational_resource`: factual, neutral explanations by topic.
2. `game_example`: approved examples that teach schema, style, and difficulty.
3. `rejection_note`: later, short examples of rejected outputs and why they failed.

Start with the first two. Do not add rejection notes until evaluation workflows
use them explicitly.

## File Format

Use JSONL. One document per line.

Required top-level fields:

- `doc_id`: stable unique id.
- `content`: text that will be chunked and embedded.
- `metadata`: retrieval metadata.

Required metadata fields for educational resources:

- `kind`: `educational_resource`
- `source`: curation batch or source id
- `language`: `en`
- `category`: product category, for example `Science & Nature`
- `topic`: concrete topic, for example `photosynthesis`

Recommended metadata fields:

- `difficulty_band`: `introductory`, `intermediate`, or `advanced`
- `quality_tier`: `draft`, `reviewed`, or `gold`
- `audience`: optional learner profile
- `source_url`: optional source link when available and allowed

## Templates

Use these templates as copy references:

- [../templates/rag-educational-resource.jsonl](../templates/rag-educational-resource.jsonl)
- [../templates/rag-game-example.jsonl](../templates/rag-game-example.jsonl)

## What To Add First

Prioritize breadth before depth:

1. Add at least two `educational_resource` documents per target category.
2. Keep all runtime corpus entries in English.
3. Add at least two approved `game_example` documents per game type.
4. For weak categories, prefer factual resources over more examples.

Current target categories are audited by:

```bash
cd src
python -m ai_engine.cli.audit_rag_corpus
```

Machine-readable output:

```bash
python -m ai_engine.cli.audit_rag_corpus --format json
```

High-priority gaps only:

```bash
python -m ai_engine.cli.audit_rag_corpus --priority high
```

After adding documents, run the retrieval quality benchmark:

```bash
python -m ai_engine.cli.evaluate_rag_quality --format text
```

Track at least these metrics in your TFM evidence notes:

- hit rate should remain close to 100% for curated benchmark cases
- MRR should improve or stay stable after adding targeted documents
- p95 retrieval latency should stay low enough for interactive generation
- non-English document count should stay at zero in runtime corpus audits

## Building The Vector Index

After adding JSONL documents, build or rebuild the local Chroma index:

```bash
cd src
python -m ai_engine.cli.build_rag_index \
  --backend chroma \
  --path data/chroma \
  --collection ai_engine_default \
  --clear \
  --query "photosynthesis and chloroplasts"
```

Do not commit generated Chroma database files. Commit the JSONL source documents
and tests/docs only.

## Quality Rules

Good RAG documents:

- are short enough to stay focused
- explain one topic clearly
- include common misconceptions
- avoid unsupported trivia
- avoid copyrighted passages copied verbatim
- include category, language, topic, and source metadata

Bad RAG documents:

- mix many unrelated topics
- contain marketing prose or vague summaries
- include secrets, user data, or private operational data
- rely on the LLM to infer category/language from text alone

## Suggested Datasets For AxiomNode

Current versioned files under `ai_engine/examples/rag_documents/`:

- `educational_seed.jsonl`: initial gold educational resources.
- `educational_expansion.jsonl`: broader factual coverage across all target categories.
- `category_depth_expansion.jsonl`: extra depth for every target category.
- `approved_game_examples.jsonl`: curated outputs validated against the game models.
- `approved_game_examples_depth.jsonl`: category-completion approved examples.

Recommended next split files as the corpus grows:

- `science_seed.jsonl`: biology, physics, chemistry, astronomy, earth science.
- `history_seed.jsonl`: ancient civilizations, revolutions, world wars, Cold War.
- `computers_seed.jsonl`: algorithms, databases, networks, security, AI basics.
- `geography_seed.jsonl`: maps, climate, tectonics, demography, ecosystems.
- `math_seed.jsonl`: probability, geometry, algebra, statistics, measurement.
- category-specific approved example files when a single file becomes hard to review.

Approved game examples must be valid against `GameEnvelope` and the concrete
models in `ai_engine.games.schemas`. Tests extract the embedded JSON blocks and
validate them, so schema drift should fail before the examples reach runtime RAG.

For the TFM, keep a small evidence table mapping each corpus file to:

- source or curation method
- category coverage
- English category/topic coverage
- validation date
- known limitations

That makes the AI component measurable and defensible instead of just prompt-based.
