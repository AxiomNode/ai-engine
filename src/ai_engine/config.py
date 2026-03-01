"""Configuration loader for AI-Engine supporting dev/local/prod profiles.

It loads YAML files from the `config/` directory and merges
environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load config for `env` (local, dev, prod). Defaults to env var `AI_ENGINE_ENV` or `local`.

    Environment variables such as `MODEL_PATH`, `CHROMA_PERSIST_DIR`, `DATABASE_URL`
    will override YAML values when present.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for configuration. Install extras 'llm' or 'pip install pyyaml'.")

    env = env or os.environ.get("AI_ENGINE_ENV", "local")
    path = CONFIG_DIR / f"{env}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Simple env overrides
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        cfg.setdefault("model", {})["path"] = model_path

    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR")
    if chroma_dir:
        cfg.setdefault("vector_store", {})["chroma_persist_dir"] = chroma_dir

    db_uri = os.environ.get("DATABASE_URL")
    if db_uri:
        cfg.setdefault("database", {})["uri"] = db_uri

    # Expose raw env values for use by code
    cfg.setdefault("_env", {})["AI_ENGINE_ENV"] = env
    cfg.setdefault("_env", {})["MODEL_PATH"] = cfg.get("model", {}).get("path")

    return cfg
