"""Shared test configuration for ai-engine tests."""

from __future__ import annotations

import pytest


def _clear_all_settings_caches() -> None:
    """Clear get_settings LRU cache across all known import paths.

    Due to sys.path containing both the source directory and the installed
    package, ``ai_engine.config.get_settings`` may exist as two distinct
    ``lru_cache``-wrapped functions.  We clear every copy we can find so
    that environment-variable mutations in tests always take effect.
    """
    import sys

    seen_ids: set[int] = set()
    for _name, mod in list(sys.modules.items()):
        try:
            gs = getattr(mod, "get_settings", None)
        except Exception:
            continue
        if gs is None:
            continue
        try:
            has_clear = hasattr(gs, "cache_clear")
        except Exception:
            continue
        if has_clear and id(gs) not in seen_ids:
            seen_ids.add(id(gs))
            gs.cache_clear()


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the get_settings LRU cache before and after each test.

    This ensures that tests which manipulate environment variables
    via monkeypatch receive fresh Settings instances.
    """
    _clear_all_settings_caches()
    yield
    _clear_all_settings_caches()
