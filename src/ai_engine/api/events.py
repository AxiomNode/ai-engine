"""Observability event helpers for the ai-engine generation API."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

import httpx
from fastapi import HTTPException, Request

from ai_engine.api.schemas import GenerateRequest
from ai_engine.observability.collector import StatsCollector

logger = logging.getLogger(__name__)


def _get_collector(request: Request) -> StatsCollector:
    """Retrieve the StatsCollector from app state."""
    return request.app.state.collector  # type: ignore[no-any-return]


def _generation_failure_metadata(
    req: GenerateRequest,
    *,
    correlation_id: str,
    distribution_version: str,
    effective_max_tokens: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized metadata for failed generation events."""
    metadata: dict[str, Any] = {
        "event_type": "generation",
        "cache_hit": False,
        "cache_layer": "none",
        "game_type": req.game_type,
        "language": "en",
        "requested_max_tokens": req.max_tokens,
        "effective_max_tokens": effective_max_tokens or req.max_tokens,
        "difficulty_percentage": req.difficulty_percentage,
        "use_cache": req.use_cache,
        "force_refresh": req.force_refresh,
        "query_chars": len(req.resolved_topic),
        "correlation_id": correlation_id,
        "distribution_version": distribution_version,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


async def _publish_event_to_stats(request: Request, payload: dict[str, Any]) -> None:
    """Push observability event to ai-stats without breaking generation flow."""
    stats_url = getattr(request.app.state, "stats_url", None)
    if not isinstance(stats_url, str) or not stats_url.strip():
        return

    endpoint = f"{stats_url.rstrip('/')}/events"
    headers: dict[str, str] = {}
    stats_api_key = getattr(request.app.state, "stats_api_key", None)
    if isinstance(stats_api_key, str) and stats_api_key.strip():
        headers["X-API-Key"] = stats_api_key.strip()

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(endpoint, json=payload, headers=headers)
    except Exception:
        logger.warning("Failed to push observability event to ai-stats", exc_info=True)


async def _record_observability_event(
    request: Request,
    *,
    prompt: str,
    response: str,
    latency_ms: float,
    max_tokens: int,
    json_mode: bool,
    success: bool,
    game_type: str,
    metadata: dict[str, Any],
    error: str | None = None,
    prompt_version: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Record event locally and forward it to ai-stats."""
    _get_collector(request).record_call(
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        max_tokens=max_tokens,
        json_mode=json_mode,
        success=success,
        game_type=game_type,
        error=error,
        metadata=metadata,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    await _publish_event_to_stats(
        request,
        {
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "success": success,
            "game_type": game_type,
            "error": error,
            "metadata": metadata,
            "prompt_version": prompt_version,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )


async def _handle_generation_failure(
    exc: Exception,
    *,
    request: Request,
    req: GenerateRequest,
    correlation_id: str,
    distribution_version: str,
    effective_max_tokens: int,
    elapsed_ms: float,
    log_label: str,
) -> NoReturn:
    """Record failure observability event and raise the appropriate HTTPException.

    Centralises the duplicated error handling between _execute_generate and
    _execute_generate_sdk. Always raises — either HTTPException or the original exc.
    """
    if isinstance(exc, ValueError):
        await _record_observability_event(
            request,
            prompt=req.resolved_topic,
            response="",
            latency_ms=elapsed_ms,
            max_tokens=req.max_tokens,
            json_mode=True,
            success=False,
            game_type=req.game_type,
            error=str(exc),
            metadata=_generation_failure_metadata(
                req,
                correlation_id=correlation_id,
                distribution_version=distribution_version,
                effective_max_tokens=effective_max_tokens,
            ),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
        status_code = 504 if isinstance(exc, httpx.TimeoutException) else 503
        detail = (
            "Upstream LLM request timed out."
            if isinstance(exc, httpx.TimeoutException)
            else "Upstream LLM request failed."
        )
        verb = "timeout" if isinstance(exc, httpx.TimeoutException) else "request error"
        failure_metadata = _generation_failure_metadata(
            req,
            correlation_id=correlation_id,
            distribution_version=distribution_version,
            effective_max_tokens=effective_max_tokens,
            extra_metadata={
                **(getattr(exc, "generation_metrics", {}) or {}),
                "upstream_service": "llama",
                "error_type": exc.__class__.__name__,
            },
        )
        logger.warning(
            "%s upstream %s correlation_id=%s game_type=%s max_tokens=%s",
            log_label,
            verb,
            correlation_id,
            req.game_type,
            req.max_tokens,
            exc_info=True,
        )
        await _record_observability_event(
            request,
            prompt=req.resolved_topic,
            response="",
            latency_ms=float(failure_metadata.get("total_latency_ms", 0.0)),
            max_tokens=effective_max_tokens,
            json_mode=True,
            success=False,
            game_type=req.game_type,
            error=str(exc),
            metadata=failure_metadata,
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc

    # Generic / unexpected exception — record and re-raise as-is.
    failure_metadata = _generation_failure_metadata(
        req,
        correlation_id=correlation_id,
        distribution_version=distribution_version,
        effective_max_tokens=effective_max_tokens,
        extra_metadata=getattr(exc, "generation_metrics", None),
    )
    logger.exception(
        "%s request failed correlation_id=%s game_type=%s max_tokens=%s",
        log_label,
        correlation_id,
        req.game_type,
        req.max_tokens,
    )
    await _record_observability_event(
        request,
        prompt=req.resolved_topic,
        response="",
        latency_ms=float(failure_metadata.get("total_latency_ms", 0.0)),
        max_tokens=effective_max_tokens,
        json_mode=True,
        success=False,
        game_type=req.game_type,
        error=str(exc),
        metadata=failure_metadata,
    )
    raise exc
