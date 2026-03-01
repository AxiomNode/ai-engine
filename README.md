# ai-engine

Motor de IA — RAG y Base de Conocimiento (KBD)

## Descripción

**ai-engine** proporciona los bloques fundamentales para construir sistemas de
**Retrieval-Augmented Generation (RAG)** y una **Base de Datos de Conocimiento (KBD)**
en Python.

## Estructura del proyecto

```
src/ai_engine/
├── rag/                  # Módulo RAG
│   ├── document.py       # Modelo de documento
│   ├── chunker.py        # División de texto en fragmentos
│   ├── embedder.py       # Interfaz de embeddings (abstracta)
│   ├── vector_store.py   # Almacén vectorial (abstracto + InMemory)
│   ├── retriever.py      # Recuperación de documentos relevantes
│   └── pipeline.py       # Orquestador end-to-end
└── kbd/                  # Módulo KBD
    ├── entry.py          # Modelo de entrada de conocimiento
    └── knowledge_base.py # Gestión de la base de conocimiento
```

## Instalación (desarrollo)

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Uso rápido

### RAG

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
# Implementa Embedder con tu modelo preferido (OpenAI, HuggingFace, etc.)

pipeline = RAGPipeline(embedder=my_embedder, vector_store=InMemoryVectorStore())
pipeline.ingest([Document(content="Python es un lenguaje...", doc_id="1")])
context = pipeline.build_context("¿Qué es Python?")
```

### KBD

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()
kb.add(KnowledgeEntry("1", "Python", "Python es un lenguaje de alto nivel", tags=["python"]))
results = kb.search_by_tag("python")
```
