"""Rate limiting and generation capacity control for the ai-engine API."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    pass


class _FixedWindowRateLimiter:
    """Thread-safe fixed-window limiter keyed by client identity."""

    _CLEANUP_THRESHOLD = 10_000

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, identity: str) -> bool:
        """Return True when request is allowed, else False."""
        now = time.time()
        with self._lock:
            window_start, count = self._buckets.get(identity, (now, 0))
            if now - window_start >= self._window_seconds:
                self._buckets[identity] = (now, 1)
                self._maybe_cleanup(now)
                return True

            if count >= self._max_requests:
                return False

            self._buckets[identity] = (window_start, count + 1)
            return True

    def _maybe_cleanup(self, now: float) -> None:
        """Evict expired buckets when the map grows too large (called under lock)."""
        if len(self._buckets) < self._CLEANUP_THRESHOLD:
            return
        expired = [
            k
            for k, (ws, _) in self._buckets.items()
            if now - ws >= self._window_seconds
        ]
        for k in expired:
            del self._buckets[k]


class _GenerationCapacityLimiter:
    """Bound actively executing generation requests and a small waiting queue."""

    def __init__(self, max_in_flight: int, max_queue_size: int) -> None:
        self._max_in_flight = max_in_flight
        self._max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._lock = threading.Lock()
        self._active = 0
        self._interactive_queued = 0

    async def acquire(self, caller_tier: str = "interactive") -> bool:
        """Reserve execution capacity using a tier-aware admission policy."""
        normalized_tier = (
            caller_tier.strip().lower()
            if isinstance(caller_tier, str)
            else "interactive"
        )
        if normalized_tier == "background":
            return await self._acquire_background()
        return await self._acquire_interactive()

    async def _acquire_background(self) -> bool:
        """Allow background callers only when a slot is immediately available."""
        with self._lock:
            if self._active >= self._max_in_flight or self._interactive_queued > 0:
                return False
            self._active += 1

        try:
            await self._semaphore.acquire()
        except BaseException:
            with self._lock:
                if self._active > 0:
                    self._active -= 1
            raise
        return True

    async def _acquire_interactive(self) -> bool:
        """Reserve execution capacity or queue briefly for interactive callers."""
        immediate_slot = False
        with self._lock:
            if self._active < self._max_in_flight and self._interactive_queued == 0:
                self._active += 1
                immediate_slot = True
            elif self._interactive_queued >= self._max_queue_size:
                return False
            else:
                self._interactive_queued += 1

        try:
            await self._semaphore.acquire()
        except BaseException:
            with self._lock:
                if immediate_slot and self._active > 0:
                    self._active -= 1
                elif not immediate_slot and self._interactive_queued > 0:
                    self._interactive_queued -= 1
            raise

        if not immediate_slot:
            with self._lock:
                if self._interactive_queued > 0:
                    self._interactive_queued -= 1
                self._active += 1
        return True

    def release(self) -> None:
        """Release one in-flight execution slot."""
        self._semaphore.release()
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def stats(self) -> dict[str, int]:
        """Return current limiter counters for diagnostics."""
        with self._lock:
            return {
                "max_in_flight": self._max_in_flight,
                "max_queue_size": self._max_queue_size,
                "active": self._active,
                "queued": self._interactive_queued,
                "interactive_queued": self._interactive_queued,
                "background_queue_size": 0,
            }


def _get_rate_limiter(request: Request) -> _FixedWindowRateLimiter | None:
    """Return generation rate limiter from app state when enabled."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    return limiter if isinstance(limiter, _FixedWindowRateLimiter) else None


def _get_generation_capacity_limiter(
    request: Request,
) -> _GenerationCapacityLimiter | None:
    """Return generation capacity limiter from app state when configured."""
    limiter = getattr(request.app.state, "generation_capacity_limiter", None)
    return limiter if isinstance(limiter, _GenerationCapacityLimiter) else None


def _resolve_generation_caller_tier(request: Request) -> str:
    """Classify generation callers into interactive or background capacity tiers."""
    auth_scope = getattr(request.state, "auth_scope", "")
    if auth_scope == "games":
        return "background"
    return "interactive"


def _resolve_rate_limit_identity(request: Request, *, trust_proxy: bool = False) -> str:
    """Build a stable limiter identity using API key first, then client IP.

    When trust_proxy is True, reads X-Forwarded-For to identify the real client
    behind a reverse proxy instead of using the proxy's IP directly.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"

    if trust_proxy:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            real_ip = forwarded_for.split(",")[0].strip()
            if real_ip:
                return f"ip:{real_ip}"

    client_host = request.client.host if request.client is not None else "unknown"
    return f"ip:{client_host}"


def _enforce_generation_rate_limit(
    request: Request, *, trust_proxy: bool = False
) -> None:
    """Raise HTTP 429 when generation rate limit is exceeded."""
    limiter = _get_rate_limiter(request)
    if limiter is None:
        return

    identity = _resolve_rate_limit_identity(request, trust_proxy=trust_proxy)
    if not limiter.allow(identity):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")


async def _acquire_generation_capacity(
    request: Request,
) -> _GenerationCapacityLimiter | None:
    """Reserve a generation capacity slot or reject early when the queue is full."""
    limiter = _get_generation_capacity_limiter(request)
    if limiter is None:
        return None
    admitted = await limiter.acquire(_resolve_generation_caller_tier(request))
    if not admitted:
        raise HTTPException(
            status_code=503,
            detail="Generation service is busy. Please retry shortly.",
        )
    return limiter


def get_capacity_limiter_stats(request: Request) -> dict[str, Any]:
    """Return capacity limiter counters for health/diagnostics endpoints."""
    capacity_limiter = _get_generation_capacity_limiter(request)
    if capacity_limiter is None:
        return {
            "status": "unavailable",
            "max_in_flight": 0,
            "max_queue_size": 0,
            "active": 0,
            "queued": 0,
            "interactive_queued": 0,
            "background_queue_size": 0,
        }
    return {"status": "ready", **capacity_limiter.stats()}
