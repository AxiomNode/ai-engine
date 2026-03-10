"""Model management utilities – download, verify, and locate GGUF models.

The recommended model for structured JSON educational-game generation is
**Qwen2.5-7B-Instruct** (GGUF Q4_K_M, ~4.8 GB):

- Best Spanish language quality among local <10 B models.
- Excellent structured JSON output with GBNF grammar constraints.
- Higher factual accuracy than 3 B variants — critical for educational content.
- Apache-2.0 licence — free for commercial use.
- Requires ~5.5 GB peak RAM; fits on an 8 GB GPU or modern CPU.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Model registry – add new models here
# ------------------------------------------------------------------

MODELS: dict[str, dict[str, Any]] = {
    "qwen2.5-7b": {
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"
            "/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
        "size_mb": 4800,
        "description": (
            "Qwen2.5-7B-Instruct Q4_K_M – recommended for AxiomNode. "
            "Best Spanish quality + JSON fidelity among local models. ~4.8 GB download."
        ),
        "n_ctx": 4096,
        "sha256": None,  # Optional; set to enable integrity check.
    },
    "phi-3.5-mini": {
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF"
            "/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        ),
        "size_mb": 2390,
        "description": (
            "Phi-3.5-mini-instruct Q4_K_M – 3.8 B params, lightweight fallback "
            "for structured JSON generation. ~2.4 GB download."
        ),
        "n_ctx": 4096,
        "sha256": None,
    },
    "qwen2.5-3b": {
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF"
            "/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        ),
        "size_mb": 2050,
        "description": (
            "Qwen2.5-3B-Instruct Q4_K_M – ultra-light fallback, "
            "multilingual, acceptable JSON output. ~2 GB download."
        ),
        "n_ctx": 4096,
        "sha256": None,
    },
}

DEFAULT_MODEL = "qwen2.5-7b"


def get_models_dir() -> Path:
    """Return the models directory, respecting the ``AI_ENGINE_MODELS_DIR``
    env variable.  Creates the directory if it does not exist.
    """
    from ai_engine.config import get_settings  # local import avoids circular deps

    d = Path(get_settings().models_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(model_name: str | None = None) -> Path:
    """Return the full path to a model's GGUF file on disk.

    Args:
        model_name: Key in :data:`MODELS` (defaults to :data:`DEFAULT_MODEL`).

    Raises:
        FileNotFoundError: If the model has not been downloaded yet.
    """
    name = model_name or DEFAULT_MODEL
    info = MODELS.get(name)
    if info is None:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODELS)}"
        )
    path = get_models_dir() / info["filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            f"Run `python -m ai_engine.llm.model_manager download {name}` first."
        )
    return path


def download_model(
    model_name: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download a GGUF model from Hugging Face.

    Args:
        model_name: Key in :data:`MODELS`.
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded file.
    """
    name = model_name or DEFAULT_MODEL
    info = MODELS.get(name)
    if info is None:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODELS)}"
        )

    dest = get_models_dir() / info["filename"]

    if dest.exists() and not force:
        logger.info("Model already exists at %s (use force=True to re-download)", dest)
        return dest

    url = info["url"]
    size_mb = info["size_mb"]
    logger.info("Downloading %s (~%d MB) from %s …", name, size_mb, url)

    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        logger.info(
                            "  %d%% (%d MB / %d MB)",
                            pct,
                            downloaded // (1024 * 1024),
                            total // (1024 * 1024),
                        )
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    # Integrity check (optional)
    expected = info.get("sha256")
    if expected:
        logger.info("Verifying SHA-256 …")
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        actual = h.hexdigest()
        if actual != expected:
            tmp.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch! Expected {expected}, got {actual}"
            )

    shutil.move(str(tmp), str(dest))
    logger.info("Model saved to %s", dest)
    return dest


def list_models() -> list[dict[str, Any]]:
    """Return metadata for all registered models."""
    result = []
    models_dir = get_models_dir()
    for key, info in MODELS.items():
        entry = {
            "name": key,
            "filename": info["filename"],
            "size_mb": info["size_mb"],
            "description": info["description"],
            "downloaded": (models_dir / info["filename"]).exists(),
        }
        result.append(entry)
    return result


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def _cli() -> None:
    """Minimal CLI: ``python -m ai_engine.llm.model_manager <command> [model]``"""
    import sys

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m ai_engine.llm.model_manager <command> [model_name]")
        print()
        print("Commands:")
        print("  list                    List available models and download status")
        print("  download [model_name]   Download a model (default: phi-3.5-mini)")
        print("  path [model_name]       Print the path to a downloaded model")
        return

    cmd = args[0]
    name = args[1] if len(args) > 1 else None

    if cmd == "list":
        for m in list_models():
            status = "✓" if m["downloaded"] else "✗"
            print(f"  [{status}] {m['name']:20s}  {m['description']}")
    elif cmd == "download":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        download_model(name)
    elif cmd == "path":
        print(model_path(name))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
