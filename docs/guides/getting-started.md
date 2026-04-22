# Getting Started

This guide covers setting up a local development environment, running the
test suite, and making your first code contribution.

---

## Prerequisites

- **Python ≥ 3.10** — verify with `python --version`.
- **Git** — the project uses Git Flow branching (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).
- (Optional) **CUDA toolkit** if you want GPU acceleration for local LLM inference.

---

## 1. Clone and install

```bash
git clone <repo-url>
cd ai-engine
cd src

# Create an isolated virtual environment
python -m venv .venv

# Activate it
# Linux / macOS
. .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash)
source .venv/Scripts/activate
```

### Install in editable mode

Choose the extras that match what you need:

```bash
# Minimal — core only (requests)
pip install -e .

# Development tools (pytest, ruff, black, isort, mypy, pre-commit)
pip install -e ".[dev]"

# RAG + embeddings (sentence-transformers, chromadb)
pip install -e ".[rag]"

# Local LLM inference (llama-cpp-python)
pip install -e ".[llm]"

# Educational game generation (llm + rag combined)
pip install -e ".[games]"

# Observability API (FastAPI + uvicorn)
pip install -e ".[api]"

# Everything at once
pip install -e ".[dev,rag,llm,games,api]"
```

> Tip: For day-to-day development you typically want `[dev,games,api]`.

If you are working on split-runtime or deployment-sensitive behavior, add `redis` as well:

```bash
pip install -e ".[dev,games,api,redis]"
```

---

## 2. Install pre-commit hooks

```bash
pre-commit install
```

After this, `black`, `isort`, `ruff`, and general file checks run
automatically on every `git commit`. You can also run them manually:

```bash
pre-commit run --all-files
```

---

## 3. Download an LLM model (optional)

Only needed if you plan to run local inference or game generation.

```bash
# Download the default model (Qwen2.5-7B-Instruct Q4_K_M, ~4.8 GB)
python -m ai_engine.llm.model_manager download

# List all available models and their download status
python -m ai_engine.llm.model_manager list

# Download a lighter model (e.g. for resource-constrained environments)
python -m ai_engine.llm.model_manager download qwen2.5-3b

# Specify a custom directory (also respected via env var AI_ENGINE_MODELS_DIR)
AI_ENGINE_MODELS_DIR=/data/models python -m ai_engine.llm.model_manager download
```

---

## 4. Run the test suite

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=ai_engine --cov-report=term-missing

# Run a specific module's tests
pytest tests/test_rag/

# Run a single test file
pytest tests/test_llm/test_model_manager.py -v
```

All tests must pass before opening a PR. The minimum accepted coverage is **80%**.

---

## 5. Code quality checks

Run these before committing (pre-commit handles them automatically, but it's
useful to run them manually during development):

```bash
# Lint
ruff check ai_engine/ tests/

# Auto-fix lint issues
ruff check --fix ai_engine/ tests/

# Format
black ai_engine/ tests/

# Sort imports
isort ai_engine/ tests/

# Type check
mypy ai_engine/ --ignore-missing-imports
```

---

## 6. TDD workflow

All new code follows **Test-Driven Development**:

1. **Write a failing test** that describes the desired behaviour.
2. **Run** `pytest path/to/test_file.py -v` — it must fail.
3. **Implement** the minimum code to make it pass.
4. **Run** `pytest` again — all tests must be green.
5. **Refactor** while keeping tests green.
6. Commit with a Conventional Commit message in English.

Example:

```bash
# 1. Write test in tests/test_rag/test_my_feature.py
# 2. Confirm it fails
pytest tests/test_rag/test_my_feature.py -v

# 3. Implement in ai_engine/rag/my_feature.py
# 4. Confirm it passes
pytest tests/test_rag/test_my_feature.py -v

# 5. Run full suite
pytest

# 6. Commit
git add .
git commit -m "feat(rag): add my feature"
```

---

## 7. Project structure reference

```
ai-engine/
├── src/
│   ├── ai_engine/          # Source code
│   │   ├── rag/            # RAG layer
│   │   ├── llm/            # LLM layer
│   │   ├── games/          # Educational game generation
│   │   ├── observability/  # Stats & API
│   │   └── kbd/            # Knowledge base
│   ├── tests/              # Test suites
│   ├── notebooks/          # Jupyter demo notebooks
│   ├── models/             # Downloaded GGUF files (gitignored)
│   ├── data/               # Runtime artefacts (gitignored)
│   └── pyproject.toml      # Build system + dependencies + tool config
├── AGENTS.md               # Contribution rules
└── CONTRIBUTING.md         # Developer quick-reference
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AI_ENGINE_MODELS_DIR` | `<project_root>/src/models/` | Directory where GGUF files are stored and searched. |
| `AI_ENGINE_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Default sentence-transformers model used by the API/runtime path. |

## Local verification checklist

Before treating the environment as ready, verify:

1. `pytest` runs successfully for the slice you are touching
2. the selected GGUF model is present if local inference is required
3. any llama HTTP target you use is compatible with `/completion` or `/v1/completions`
4. the distribution or `.env` files match the intended stage/environment

Set it in your shell or in a `.env` file loaded by your runner:

```bash
export AI_ENGINE_MODELS_DIR=/my/fast/disk/models
```
