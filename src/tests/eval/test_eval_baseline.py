"""Pytest entry point for the LLMOps eval harness (ADR 0009).

Runs the baseline dataset and asserts soft thresholds. Failure here is a
regression signal in either the prompt templates, the generator pipeline
or the schema validation layer.
"""

from __future__ import annotations

import pytest

from tests.eval.runner import (
    DEFAULT_DATASET,
    DEFAULT_REPORT,
    run_eval,
    write_report,
)

# Soft thresholds for the canned-LLM baseline. Real-LLM gating thresholds
# will be introduced once a stable provider is wired in (ADR 0009 step 5).
MIN_SUCCESS_RATE = 1.0
MAX_LATENCY_P95_MS = 2000.0


@pytest.mark.eval
def test_eval_baseline() -> None:
    report = run_eval(DEFAULT_DATASET)
    write_report(report, DEFAULT_REPORT)

    assert report.total > 0, "baseline dataset is empty"
    assert (
        report.success_rate >= MIN_SUCCESS_RATE
    ), f"eval success rate {report.success_rate:.2%} below threshold {MIN_SUCCESS_RATE:.2%}"
    assert (
        report.latency_p95_ms <= MAX_LATENCY_P95_MS
    ), f"eval latency p95 {report.latency_p95_ms:.1f}ms exceeds {MAX_LATENCY_P95_MS}ms"
