"""LLMOps eval harness for ai-engine (ADR 0009).

This package contains the evaluation dataset and runner used to detect
regressions in the educational game generation pipeline.

The harness exercises :class:`ai_engine.games.generator.GameGenerator`
end-to-end against a set of canned LLM responses (no real LLM call), so
it can run in CI without GPU/CPU cost. It validates:

- JSON parsing and schema validation succeed.
- The requested ``game_type`` and ``num_questions`` are honoured.
- The generated title is non-empty.
- Per-call latency stays under a soft threshold.

Future iterations may swap the canned LLM for a real provider behind a
flag and gate CI on the same metrics (see ADR 0009 step 5).
"""
