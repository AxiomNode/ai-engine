"""API Key authentication middleware for ai-engine FastAPI applications.

When the environment variable ``AI_ENGINE_API_KEY`` is set every incoming
request must carry a matching ``X-API-Key`` header.  When the variable is
**not** set the middleware is transparent — the application behaves as if it
were open (useful for local development and testing).

Usage::

    from fastapi import FastAPI
    from ai_engine.api.middleware import add_api_key_middleware

    app = FastAPI()
    add_api_key_middleware(app)

Responses:

- **401 Unauthorized** – ``X-API-Key`` header is absent.
- **403 Forbidden**    – ``X-API-Key`` header is present but incorrect.
- **pass-through**     – Header matches configured key, or no key is configured.
"""

from __future__ import annotations

from ai_engine.config import get_settings

try:
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for API key middleware.  "
        "Install it with:  pip install ai-engine[api]"
    ) from _err

_HEADER_NAME = "x-api-key"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces a shared API key.

    The key is read from ``AI_ENGINE_API_KEY`` at construction time so a
    single instance can be used for the lifetime of the application.

    Args:
        app: The ASGI application to wrap.
        api_key: The expected key value.  If ``None`` or empty, all requests
            are passed through without authentication.
    """

    def __init__(self, app, api_key: str | None = None) -> None:  # type: ignore[override]
        super().__init__(app)
        self._api_key: str | None = api_key or None

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        """Check the X-API-Key header before forwarding the request.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that forwards the request to the next handler.

        Returns:
            A 401/403 JSON response when authentication fails, or the normal
            application response when it succeeds (or no key is configured).
        """
        if self._api_key is None:
            # No key configured — pass through.
            return await call_next(request)

        provided = request.headers.get(_HEADER_NAME)
        if provided is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header."},
            )
        if provided != self._api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )
        return await call_next(request)


def add_api_key_middleware(app: FastAPI) -> None:
    """Attach :class:`APIKeyMiddleware` to *app* using the current environment.

    Reads ``AI_ENGINE_API_KEY`` from the process environment at call time.
    If the variable is unset or empty, no middleware is added and the
    application remains open.

    Args:
        app: The :class:`~fastapi.FastAPI` instance to protect.

    Example::

        app = FastAPI()
        add_api_key_middleware(app)   # reads AI_ENGINE_API_KEY from env
    """
    api_key = get_settings().api_key or None
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
