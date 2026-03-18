# KBD Usage Guide

The KBD (Knowledge Base Directory) module (`ai_engine.kbd`) provides a
lightweight **structured knowledge store** based on simple dataclasses.

Unlike the RAG layer (which stores text chunks as embedding vectors for
semantic search), KBD is an **in-memory CRUD store** indexed by a unique ID.
It is best suited for curated, human-authored entries — a glossary, a
reference manual, a set of learning objectives — that need to be queried by
tag or keyword.

---

## Concepts

| Term | Meaning |
|---|---|
| **KnowledgeEntry** | A single record with an ID, title, content, optional tags, and metadata |
| **KnowledgeBase** | The container: indexes entries by ID, supports CRUD and tag/keyword search |
| **Tag** | A free-form string label attached to an entry (e.g. `"biology"`, `"chapter-3"`) |

---

## Installation

KBD has no external runtime dependencies. The optional `tinydb` extra is
declared for future persistent storage but the core module works out of
the box:

```bash
pip install -e .        # core only — includes KBD
```

---

## KnowledgeEntry model

```python
from ai_engine.kbd.entry import KnowledgeEntry

entry = KnowledgeEntry(
    entry_id="bio-001",
    title="Photosynthesis",
    content="Photosynthesis is the process by which green plants convert "
            "sunlight into glucose using CO₂ and water.",
    tags=["biology", "plants", "energy"],
    metadata={"source": "chapter4.pdf", "grade_level": 8},
)

print(entry.entry_id)   # "bio-001"
print(entry.tags)       # ["biology", "plants", "energy"]
```

### Validation rules

| Field | Rule |
|---|---|
| `entry_id` | Must be a non-empty string |
| `title` | Must be a non-empty string |
| `content` | Must be a string (can be empty) |
| `tags` | List of strings, defaults to `[]` |
| `metadata` | Dict, defaults to `{}` |

Violations raise `ValueError` or `TypeError` in `__post_init__`.

### Serialization

```python
# to dict
d = entry.to_dict()
# {"entry_id": "bio-001", "title": "Photosynthesis", "content": "...", "tags": [...], "metadata": {...}}

# from dict (round-trip)
restored = KnowledgeEntry.from_dict(d)
assert restored.entry_id == entry.entry_id
```

---

## KnowledgeBase CRUD

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()
```

### add

Stores a new entry. If an entry with the same `entry_id` already exists,
it is **replaced** (upsert behaviour).

```python
kb.add(KnowledgeEntry("bio-001", "Photosynthesis", "... content ...", tags=["biology"]))
kb.add(KnowledgeEntry("phy-001", "Newton's Laws", "... content ...", tags=["physics"]))
```

### get

Returns the entry for a given ID, or `None` if not found.

```python
entry = kb.get("bio-001")
if entry:
    print(entry.title)
```

### update

Replaces an existing entry. Raises `KeyError` if the entry does not exist.

```python
updated = KnowledgeEntry(
    entry_id="bio-001",
    title="Photosynthesis (revised)",
    content="Updated explanation...",
    tags=["biology", "plants"],
)
kb.update(updated)
```

### delete

Removes an entry by ID. Raises `KeyError` if not found.

```python
kb.delete("phy-001")
```

### Existence check and size

```python
print("bio-001" in kb)   # True
print(len(kb))           # 1
```

---

## Querying

### list_all

Returns every entry as a list (unordered).

```python
all_entries = kb.list_all()
for entry in all_entries:
    print(entry.entry_id, entry.title)
```

### search_by_tag

Case-insensitive tag match. Returns all entries that contain the given tag.

```python
results = kb.search_by_tag("biology")
# matches entries with tag "biology", "Biology", "BIOLOGY", etc.
for entry in results:
    print(entry.title)
```

### search_by_keyword

Case-insensitive full-text search over `title` and `content`.

```python
results = kb.search_by_keyword("sunlight")
for entry in results:
    print(entry.title, "—", entry.content[:60])
```

---

## Typical patterns

### Building a subject glossary

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()

terms = [
    ("mitosis", "Cell division producing two identical daughter cells.", ["biology", "cell"]),
    ("meiosis", "Cell division producing four genetically distinct gametes.", ["biology", "cell"]),
    ("osmosis", "Movement of water across a semi-permeable membrane.", ["biology", "chemistry"]),
]

for i, (title, content, tags) in enumerate(terms):
    kb.add(KnowledgeEntry(
        entry_id=f"term-{i:03d}",
        title=title.capitalize(),
        content=content,
        tags=tags,
    ))

# Find all cell biology terms
cell_terms = kb.search_by_tag("cell")
print([e.title for e in cell_terms])  # ["Mitosis", "Meiosis"]
```

### Feeding KBD entries into the RAG pipeline

KBD and RAG are independent but complementary. You can convert KBD entries
into RAG `Document` objects to get semantic search on top of your curated
knowledge base:

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

# 1. Populate KBD
kb = KnowledgeBase()
kb.add(KnowledgeEntry("e1", "Photosynthesis", "Plants convert sunlight...", tags=["biology"]))
kb.add(KnowledgeEntry("e2", "Respiration", "Cells break down glucose...", tags=["biology"]))

# 2. Build RAG pipeline from KBD entries
pipeline = RAGPipeline(
    embedder=SentenceTransformersEmbedder(),
    vector_store=InMemoryVectorStore(),
)
pipeline.ingest([
    Document(content=e.content, doc_id=e.entry_id, metadata={"title": e.title, "tags": e.tags})
    for e in kb.list_all()
])

# 3. Semantic search over KBD content
context = pipeline.build_context("how do organisms get energy?")
print(context)
```

### Persistence (manual JSON export)

The built-in store is in-memory. To persist, export to JSON and reload:

```python
import json
from pathlib import Path

# Save
data = [e.to_dict() for e in kb.list_all()]
Path("knowledge_base.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

# Load
from ai_engine.kbd.entry import KnowledgeEntry
data = json.loads(Path("knowledge_base.json").read_text())
kb = KnowledgeBase()
for d in data:
    kb.add(KnowledgeEntry.from_dict(d))
```

---

## Complete working example

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

# Build the knowledge base
kb = KnowledgeBase()

entries = [
    KnowledgeEntry("phys-001", "Newton's First Law",
                   "An object at rest stays at rest unless acted on by a force.",
                   tags=["physics", "mechanics"]),
    KnowledgeEntry("phys-002", "Newton's Second Law",
                   "Force equals mass times acceleration: F = ma.",
                   tags=["physics", "mechanics"]),
    KnowledgeEntry("phys-003", "Newton's Third Law",
                   "For every action there is an equal and opposite reaction.",
                   tags=["physics", "mechanics"]),
    KnowledgeEntry("bio-001", "Photosynthesis",
                   "Plants convert CO2 and water into glucose using sunlight.",
                   tags=["biology", "plants"]),
]
for e in entries:
    kb.add(e)

# Queries
print("All physics entries:")
for e in kb.search_by_tag("physics"):
    print(f"  [{e.entry_id}] {e.title}")

print("\nKeyword search 'force':")
for e in kb.search_by_keyword("force"):
    print(f"  {e.title}: {e.content}")

print(f"\nTotal entries: {len(kb)}")

# Update one entry
kb.update(KnowledgeEntry(
    "bio-001", "Photosynthesis",
    "Photosynthesis: 6CO2 + 6H2O + light → C6H12O6 + 6O2",
    tags=["biology", "plants", "chemistry"],
))
print("\nUpdated:", kb.get("bio-001").content)

# Remove an entry
kb.delete("bio-001")
print("After delete:", kb.get("bio-001"))   # None
```
