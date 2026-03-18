"""ai-engine – AI Engine for RAG, LLM and educational game generation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-engine")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
