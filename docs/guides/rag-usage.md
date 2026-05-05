# RAG Usage Guide

The RAG (Retrieval-Augmented Generation) module (`ai_engine.rag`) is the core
data layer of ai-engine. It provides a pipeline to **ingest documents, store
their embeddings, and retrieve the most relevant chunks** to build context for
an LLM prompt.

Measure retrieval quality and latency against the curated benchmark cases:

```bash
python -m ai_engine.cli.evaluate_rag_quality --format text
python -m ai_engine.cli.evaluate_rag_quality --format json
```

The quality report tracks:

- hit rate: how many benchmark queries retrieve an expected document/topic
- MRR: mean reciprocal rank of the first matching retrieved chunk
- average, p95, and max retrieval latency

Use these metrics before and after adding documents, changing embeddings,
switching vector stores, or tuning reranker settings.

---

## Concepts

| Term | Meaning |
|---|---|
| **Document** | A piece of text with optional metadata and a unique ID |
| **Chunk** | A smaller sub-document produced by splitting a large `Document` |
| **Embedding** | A dense vector (list of floats) that encodes the semantic meaning of text |
| **Vector store** | A database that stores chunks and their embeddings and answers similarity queries |
| **Retrieval** | Finding the chunks most semantically similar to a query |
| **Context** | The concatenated content of retrieved chunks, passed to the LLM |

---

## Installation

```bash
pip install -e ".[rag]"
```

---

## Quick start

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder

# 1. Build the pipeline
embedder = SentenceTransformersEmbedder()          # default: paraphrase-multilingual-MiniLM-L12-v2
store    = InMemoryVectorStore()
pipeline = RAGPipeline(embedder=embedder, vector_store=store, top_k=3)

# 2. Ingest documents
docs = [
    Document(content="The water cycle describes how water evaporates...", doc_id="water-1"),
    Document(content="Photosynthesis is the process by which plants...", doc_id="photo-1"),
    Document(content="Newton's first law states that an object at rest...", doc_id="newton-1"),
]
pipeline.ingest(docs)   # chunks → embeds → stores

# 3. Retrieve relevant chunks
results = pipeline.retrieve("how do plants make food?")
for doc in results:
    print(doc.doc_id, doc.content[:80])

# 4. Build a context string
context = pipeline.build_context("how do plants make food?")
print(context)          # ready to inject into an LLM prompt
```

---

## Document model

```python
from ai_engine.rag.document import Document

doc = Document(
    content="Full text of the document goes here.",
    doc_id="my-doc-001",           # optional unique ID
    metadata={"source": "chapter3.pdf", "page": 12},   # arbitrary key-value pairs
)

print(len(doc))          # character count — 38
print(doc.doc_id)        # "my-doc-001"
print(doc.metadata)      # {"source": "chapter3.pdf", "page": 12}
```

---

## Chunking

`Chunker` splits documents into overlapping fixed-size windows. Overlap
preserves context across chunk boundaries so that retrieved chunks are
coherent even when an important concept is near the edge of a window.

```python
from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document

chunker = Chunker(
    chunk_size=500,    # max characters per chunk (default: 500)
    chunk_overlap=50,  # characters shared between consecutive chunks (default: 50)
)

doc = Document(content="A very long text ..." * 100, doc_id="large-doc")
chunks = chunker.split(doc)

print(len(chunks))             # number of chunks
print(chunks[0].doc_id)        # "large-doc#0"
print(chunks[0].metadata)      # {"chunk_index": 0, "chunk_total": N, ...parent metadata}
```

`RAGPipeline` uses a `Chunker` internally during `ingest()`. To skip chunking
(e.g. when documents are already pre-chunked), pass `chunker=None` — but note
the pipeline still instantiates a default chunker. For truly pre-chunked
ingestion, use a large `chunk_size` so no split occurs.

---

## Embedders

`Embedder` is an abstract base class. Implement it to use any embedding model.

### Built-in: SentenceTransformersEmbedder

```python
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder

# Default model: paraphrase-multilingual-MiniLM-L12-v2 (384-dimensional, multilingual)
emb = SentenceTransformersEmbedder()

# Custom model
emb = SentenceTransformersEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")

vec = emb.embed_text("hello world")   # list[float], length 384
```

### Custom embedder example (OpenAI)

```python
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.document import Document
import openai

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        response = openai.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        # Batch call for efficiency
        texts = [doc.content for doc in documents]
        response = openai.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
```

The custom embedder plugs in directly:
```python
pipeline = RAGPipeline(embedder=OpenAIEmbedder(), vector_store=InMemoryVectorStore())
```

---

## Vector stores

`VectorStore` is an abstract base class. Two implementations are available:

### Curated seed corpus and persistent index build

`ai-engine` ships a versioned RAG seed corpus under
`ai_engine/examples/rag_documents/`. These JSONL documents are part of the
runtime corpus injected at API startup, together with the Python-defined game
examples and educational resources.

Use the build CLI when you want a reproducible local vector database instead
of relying only on runtime `/ingest` calls:

```bash
cd src
python -m ai_engine.cli.build_rag_index \
    --backend chroma \
    --path data/chroma \
    --collection ai_engine_default \
    --clear \
    --query "vector databases for RAG"
```

The command:

- loads the full curated corpus
- embeds it with the configured SentenceTransformers model
- writes it to the selected vector store backend
- optionally runs a smoke retrieval query

Relevant environment defaults:

- `AI_ENGINE_VECTOR_STORE_BACKEND`
- `AI_ENGINE_VECTOR_STORE_PATH`
- `AI_ENGINE_VECTOR_STORE_COLLECTION`
- `AI_ENGINE_EMBEDDING_MODEL`
- `AI_ENGINE_EMBEDDING_DEVICE`
- `AI_ENGINE_EMBEDDING_BATCH_SIZE`

### InMemoryVectorStore (built-in)

```python
from ai_engine.rag.vector_store import InMemoryVectorStore

store = InMemoryVectorStore()
# Suitable for development, testing, and small datasets.
# Not persistent — cleared when the process exits.
store.clear()           # remove all documents
```

### Custom persistent store example (ChromaDB)

```python
from ai_engine.rag.vector_store import VectorStore
from ai_engine.rag.document import Document
import chromadb

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "ai_engine") -> None:
        client = chromadb.Client()
        self._collection = client.get_or_create_collection(collection_name)

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[doc.doc_id or str(i) for i, doc in enumerate(documents)],
            documents=[doc.content for doc in documents],
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents],
        )

    def search(self, query_embedding: list[float], top_k: int = 5):
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = []
        for i, content in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            doc_id = results["ids"][0][i]
            score = 1 - results["distances"][0][i]   # convert distance to similarity
            docs.append((Document(content=content, metadata=meta, doc_id=doc_id), score))
        return docs

    def clear(self) -> None:
        self._collection.delete(where={})
```

---

## RAGPipeline in depth

```python
pipeline = RAGPipeline(
    embedder=embedder,
    vector_store=store,
    chunker=Chunker(chunk_size=400, chunk_overlap=40),  # optional
    top_k=5,            # default number of chunks to retrieve per query
    llm_client=llm,     # optional — only needed for pipeline.generate()
)
```

### `pipeline.ingest(documents)`

Chunks, embeds, and stores a list of `Document` objects. Idempotent for
new documents — calling it twice with the same docs adds duplicates (design
choice: no deduplication by default).

```python
pipeline.ingest([
    Document(content="Chapter 1 text...", doc_id="ch1"),
    Document(content="Chapter 2 text...", doc_id="ch2"),
])
```

### `pipeline.retrieve(query, top_k=None)`

Returns a list of `Document` chunks ranked by cosine similarity to the query.

```python
chunks = pipeline.retrieve("What is osmosis?", top_k=3)
for chunk in chunks:
    print(chunk.doc_id, chunk.metadata.get("chunk_index"))
```

### `pipeline.build_context(query, top_k=None)`

Shortcut that retrieves and joins chunk contents into a single string,
ready to be inserted into an LLM prompt.

```python
context = pipeline.build_context("What is osmosis?")
prompt = f"Answer based on this context:\n{context}\n\nQuestion: What is osmosis?"
```

### `pipeline.generate(query, goal, max_tokens=256)` *(requires `llm_client`)*

End-to-end generation: retrieves context then calls the LLM.
Returns raw dict from the parsed JSON response.

```python
result = pipeline.generate(
    query="water cycle",
    goal="Create a quiz about the water cycle for 8th graders.",
    max_tokens=512,
)
```

---

## Observability with RAG

Wrap the LLM client in `TrackedLlamaClient` to record every generation
call automatically:

```python
from ai_engine.llm import LlamaClient, model_path
from ai_engine.observability import StatsCollector
from ai_engine.observability.middleware import TrackedLlamaClient

collector = StatsCollector()
llm = TrackedLlamaClient(
    LlamaClient(model_path=str(model_path()), json_mode=True),
    collector,
)

pipeline = RAGPipeline(embedder=embedder, vector_store=store, llm_client=llm)

# After some calls:
print(collector.summary())
```

---

## Complete working example

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
from ai_engine.llm import LlamaClient, model_path

# --- Build pipeline ---
embedder = SentenceTransformersEmbedder()
pipeline = RAGPipeline(
    embedder=embedder,
    vector_store=InMemoryVectorStore(),
    top_k=4,
    llm_client=LlamaClient(model_path=str(model_path()), json_mode=True),
)

# --- Ingest a corpus ---
corpus = [
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information encoded in base pairs.",
    "Proteins are synthesised by ribosomes using mRNA templates.",
    "The cell membrane controls what enters and exits the cell.",
]
pipeline.ingest([Document(content=t, doc_id=str(i)) for i, t in enumerate(corpus)])

# --- Retrieve ---
context = pipeline.build_context("how is genetic information used?")
print("Context:", context[:200])

# --- Generate (requires downloaded model) ---
result = pipeline.generate(
    query="how is genetic information used?",
    goal="Explain to a 10-year-old how DNA makes proteins.",
)
print(result)
```
