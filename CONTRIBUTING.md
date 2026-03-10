# Contributing to ai-engine

Thank you for contributing! Please read this guide before opening a PR.

## Workflow — Git Flow

All development follows the **Git Flow** branching model:

| Branch | Purpose |
|---|---|
| `main` | Production-ready code only |
| `develop` | Integration branch — target of all feature PRs |
| `feature/<TICKET>-<description>` | New features / improvements |
| `release/<version>` | Release prep (bump version, final fixes) |
| `hotfix/<TICKET>-<description>` | Urgent production fixes |

### Step-by-step

```bash
# 1. Create branch from develop
git checkout develop && git pull
git checkout -b feature/TICKET-123-my-feature

# 2. Work using TDD (write failing test first, then implement)
# 3. Commit following Conventional Commits
git commit -m "feat: add X to Y module"

# 4. Push and open a PR → develop
git push -u origin feature/TICKET-123-my-feature
```

## Commit Messages (Conventional Commits)

Format: `<type>(<scope>): <description>`

| Type | When |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Build, config, tooling changes |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | Code change without feature/fix |
| `perf` | Performance improvement |

All commit messages **must be in English**.

## Test-Driven Development (TDD)

1. Write a **failing** test that describes the desired behavior.
2. Implement the **minimal** code to make it pass.
3. Refactor while keeping tests green.

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=src/ai_engine --cov-report=term-missing
```

Target: **≥ 80% coverage** on changed modules.

## Local Setup

```bash
# Clone and install in editable mode with dev tools
git clone <repo>
cd ai-engine
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Quality

The following tools run automatically via `pre-commit` and CI:

```bash
black src/ tests/          # formatting
isort src/ tests/          # import ordering
ruff check src/ tests/     # linting
mypy src/ai_engine/        # type checking
```

Run them manually before pushing:

```bash
black src/ tests/ && isort src/ tests/ && ruff check src/ tests/ && mypy src/ai_engine/ --ignore-missing-imports
```

## PR Checklist

Before requesting review, verify:

- [ ] Tests added/updated for every change.
- [ ] All CI checks pass locally.
- [ ] README or relevant docs updated if the public API changed.
- [ ] No `TODO`, `print()`, or debug statements left.
- [ ] Commit messages follow Conventional Commits and are in English.
- [ ] At least one approving review required before merge.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AI_ENGINE_MODELS_DIR` | Directory where GGUF models are stored | `<project_root>/models/` |

Never commit secrets or credentials. Use environment variables or a secrets manager.
