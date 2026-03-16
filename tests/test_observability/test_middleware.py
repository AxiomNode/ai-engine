"""Tests for the observability middleware (TrackedLlamaClient, TrackedGameGenerator)."""

from __future__ import annotations

from ai_engine.observability.collector import StatsCollector
from ai_engine.observability.middleware import TrackedGameGenerator, TrackedLlamaClient

# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


class FakeLLM:
    """Minimal LLM stub for testing."""

    default_max_tokens: int = 128
    json_mode: bool = False

    def __init__(self, response: str = "ok", *, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self, prompt: str, max_tokens: int | None = None, **kwargs: object
    ) -> str:
        """Return a canned response or raise."""
        self.calls.append((prompt, max_tokens))
        if self.fail:
            raise RuntimeError("LLM exploded")
        return self.response


class FakeGenerator:
    """Minimal GameGenerator stub for testing."""

    default_max_tokens: int = 512

    def __init__(self, result: object = None, *, fail: bool = False) -> None:
        self.result = result or {"game_type": "quiz", "game": {}}
        self.fail = fail

    def generate(self, *args: object, **kwargs: object) -> object:
        """Return a canned result or raise."""
        if self.fail:
            raise ValueError("bad json")
        return self.result

    def generate_raw(self, *args: object, **kwargs: object) -> object:
        """Return a raw dict or raise."""
        if self.fail:
            raise ValueError("bad json raw")
        return self.result


# ------------------------------------------------------------------
# TrackedLlamaClient
# ------------------------------------------------------------------


class TestTrackedLlamaClient:
    """Tests for the TrackedLlamaClient wrapper."""

    def test_records_successful_call(self) -> None:
        """A successful generate() records an event."""
        collector = StatsCollector()
        llm = FakeLLM(response="hello world")
        tracked = TrackedLlamaClient(llm, collector)

        result = tracked.generate("prompt", max_tokens=64)

        assert result == "hello world"
        assert len(collector) == 1
        s = collector.summary()
        assert s["successful_calls"] == 1

    def test_records_failed_call(self) -> None:
        """A failed generate() records the error and re-raises."""
        collector = StatsCollector()
        llm = FakeLLM(fail=True)
        tracked = TrackedLlamaClient(llm, collector)

        try:
            tracked.generate("prompt")
            assert False, "Should raise"
        except RuntimeError:
            pass

        assert len(collector) == 1
        h = collector.history()
        assert h[0]["success"] is False
        assert "exploded" in h[0]["error"]

    def test_forwards_attributes(self) -> None:
        """Attribute access is forwarded to the inner client."""
        llm = FakeLLM()
        tracked = TrackedLlamaClient(llm, StatsCollector())
        assert tracked.default_max_tokens == 128

    def test_json_mode_detected(self) -> None:
        """json_mode flag is captured from kwargs or inner client."""
        collector = StatsCollector()
        llm = FakeLLM()
        llm.json_mode = True
        tracked = TrackedLlamaClient(llm, collector)

        tracked.generate("p")
        h = collector.history()
        assert h[0]["json_mode"] is True


# ------------------------------------------------------------------
# TrackedGameGenerator
# ------------------------------------------------------------------


class TestTrackedGameGenerator:
    """Tests for the TrackedGameGenerator wrapper."""

    def test_records_generate(self) -> None:
        """A successful generate() records a game event."""
        collector = StatsCollector()
        gen = FakeGenerator(result="envelope")
        tracked = TrackedGameGenerator(gen, collector)

        result = tracked.generate("water cycle", topic="Science", game_type="quiz")
        assert result == "envelope"
        assert len(collector) == 1
        h = collector.history()
        assert h[0]["game_type"] == "quiz"

    def test_records_generate_failure(self) -> None:
        """A failed generate() records the error."""
        collector = StatsCollector()
        gen = FakeGenerator(fail=True)
        tracked = TrackedGameGenerator(gen, collector)

        try:
            tracked.generate("q", topic="t", game_type="quiz")
            assert False, "Should raise"
        except ValueError:
            pass

        h = collector.history()
        assert h[0]["success"] is False
        assert h[0]["game_type"] == "quiz"

    def test_records_generate_raw(self) -> None:
        """generate_raw() is also tracked."""
        collector = StatsCollector()
        gen = FakeGenerator(result={"data": 1})
        tracked = TrackedGameGenerator(gen, collector)

        result = tracked.generate_raw("q", topic="t", game_type="true_false")
        assert result == {"data": 1}
        h = collector.history()
        assert h[0]["game_type"] == "true_false"

    def test_forwards_attributes(self) -> None:
        """Attribute access is forwarded to the inner generator."""
        gen = FakeGenerator()
        tracked = TrackedGameGenerator(gen, StatsCollector())
        assert tracked.default_max_tokens == 512
