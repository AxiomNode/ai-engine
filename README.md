# ai-engine

Motor de IA — RAG, LLM local y generación de juegos educativos

## Descripción

**ai-engine** proporciona los bloques fundamentales para construir sistemas de
**Retrieval-Augmented Generation (RAG)**, una **Base de Datos de Conocimiento (KBD)**,
integración con **modelos de lenguaje locales** (llama.cpp / GGUF) y un
**generador de juegos educativos** estructurados (quiz, pasapalabra, verdadero/falso)
en Python.

## Estructura del proyecto

```
src/ai_engine/
├── rag/                          # Módulo RAG
│   ├── document.py               # Modelo de documento
│   ├── chunker.py                # División de texto en fragmentos
│   ├── embedder.py               # Interfaz de embeddings (abstracta)
│   ├── vector_store.py           # Almacén vectorial (abstracto + InMemory)
│   ├── retriever.py              # Recuperación de documentos relevantes
│   ├── pipeline.py               # Orquestador end-to-end
│   ├── utils.py                  # Helpers (extracción de JSON, etc.)
│   └── embedders/
│       └── sentence_transformers.py  # Embedder con SentenceTransformers
├── llm/                          # Módulo LLM
│   ├── llama_client.py           # Cliente llama.cpp (API HTTP + local GGUF)
│   └── model_manager.py          # Descarga y gestión de modelos GGUF
├── games/                        # Módulo de juegos educativos
│   ├── schemas.py                # Modelos de datos (Quiz, Pasapalabra, T/F)
│   ├── prompts.py                # Plantillas de prompts por tipo de juego
│   └── generator.py              # Orquestador RAG + LLM → juego estructurado
└── kbd/                          # Módulo KBD
    ├── entry.py                  # Modelo de entrada de conocimiento
    └── knowledge_base.py         # Gestión de la base de conocimiento
```

## Instalación

### Desarrollo

```bash
pip install -e ".[dev]"
```

### Con soporte LLM local (llama.cpp)

```bash
pip install -e ".[llm]"
```

### Todo incluido (RAG + LLM + juegos)

```bash
pip install -e ".[games]"
```

## Descargar modelo LLM

El modelo recomendado es **Phi-3.5-mini-instruct** (Q4_K_M, ~2.4 GB),
excelente para generación JSON estructurada:

```bash
python -m ai_engine.llm.model_manager download
```

Para ver todos los modelos disponibles:

```bash
python -m ai_engine.llm.model_manager list
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

### Generar un juego educativo

```python
from ai_engine.rag import RAGPipeline, Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
from ai_engine.llm import LlamaClient, model_path
from ai_engine.games import GameGenerator

# 1. Configurar RAG
embedder = SentenceTransformersEmbedder()
pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())
pipeline.ingest([Document(content="El ciclo del agua consiste en...", doc_id="1")])

# 2. Configurar LLM local
llm = LlamaClient(model_path=str(model_path()), json_mode=True)

# 3. Generar quiz
gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)
game = gen.generate(
    query="ciclo del agua",
    topic="Ciencias Naturales",
    game_type="quiz",        # "quiz" | "pasapalabra" | "true_false"
    num_questions=5,
    language="es",
)
print(game.game.to_dict())
```

### KBD

```python
from ai_engine.kbd import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()
kb.add(KnowledgeEntry("1", "Python", "Python es un lenguaje de alto nivel", tags=["python"]))
results = kb.search_by_tag("python")
```
