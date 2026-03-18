# Agents & Contribution Rules for AI-Engine

Purpose
-------
This document defines the agent rules, development workflow, and code standards for the AI-Engine repository. All contributors must follow these rules to ensure consistent quality, reproducibility, and maintainability.

Core Rules
----------
1) Git Flow (branching and releases)
   - Use Git Flow branching model: `main` (production), `develop` (integration), `feature/*`, `release/*`, `hotfix/*`.
   - Branch naming examples: `feature/TICKET-123-descriptive-name`, `hotfix/TICKET-456-urgent-fix`.
   - Create a PR from a feature/release/hotfix branch into `develop` (or `main` for hotfixes/release merges per project policy).
   - Require at least one approving review and passing CI before merging.

2) Test-Driven Development (TDD)
   - Write tests first: add a failing test that describes the desired behavior.
   - Implement the minimal code to make the test pass.
   - Refactor keeping tests green.
   - All new features and bug fixes must include tests.
   - Use `pytest` for unit tests and integration tests. Keep tests deterministic and fast.

3) Documentation and Comments (English-only)
   - All documentation, commit messages, docstrings, and inline comments MUST be written in English.
   - Docstrings should follow the project's chosen style (e.g., Google or NumPy). Add examples where helpful.
   - Update high-level docs (README, architecture files) for any public API, behavior change, or new component.
   - Every public module, class, method, and function MUST have a docstring. Code without documentation will not be accepted.
   - Generate or update documentation as part of every change — this includes inline comments explaining non-obvious logic and module-level docstrings for new files.

4) Clean Code & Architecture
   - Follow clean-code principles: single responsibility, modularity, explicit interfaces, and simple functions.
   - Prefer composition over large inheritance hierarchies. Keep functions short and testable.
   - Separate layers: API/orchestration, reasoning/LLM layer, RAG/data layer, persistence, and observability.
   - Use static typing where practical (`mypy`) and enforce formatting/quality with linters.

Enforcement & CI
-----------------
- CI must run on every PR and must include at minimum:
  - `pytest` (unit + integration tests)
  - Linting (`ruff`/`flake8` or project linter)
  - Type checking (`mypy`) if configured
  - Test coverage report; aim for at least 80% on changed modules (project may set higher bar later)
- Use `pre-commit` hooks to run formatters and linters locally (`black`, `isort`, `ruff`).

PR Checklist (required before merge)
-----------------------------------
- Tests added/updated for changes.
- All CI checks passing.
- Documentation updated (README or relevant docs).
- No TODOs or debug prints left.
- At least one approving review and no unresolved comments.

Developer Guidelines (practical tips)
-----------------------------------
- Commit messages: follow Conventional Commits (e.g., `feat:`, `fix:`, `chore:`).
- Small, focused PRs are preferred.
- Use fixtures and test doubles to make tests reliable.
- Keep secrets out of the repo; use environment variables or secret management for credentials.

Why these rules?
-----------------
These rules prioritize reproducible research-quality engineering, maintainable code, and safe production practices for the AI components (agents, RAG, LLM integrations). TDD reduces regressions, Git Flow enables predictable releases, English documentation improves collaboration, and clean code practices make reasoning engines easier to audit and extend.

If you have suggestions or improvements to these rules, open an issue or a PR proposing the change and follow the same process (tests + CI + review).
