"""Trace accounting tests — the numbers the orchestrator reports about itself."""

from __future__ import annotations

import pytest

from orchestrator.trace import Trace


def test_wallclock_does_not_double_count_nested_calls():
    """A timed span encloses the llm_call events emitted inside it and both carry
    duration_ms, so summing every event counted the same elapsed time twice —
    `summary()` reported roughly 2x the real wall-clock."""
    t = Trace()
    t.start("writer", kind="summary")
    t.emit_llm("writer", model="m", tokens_in=10, tokens_out=5, cost_usd=0.01, duration_ms=2000)
    t.emit_llm("writer", model="m", tokens_in=10, tokens_out=5, cost_usd=0.01, duration_ms=2000)
    t.end()

    span = next(e for e in t.events if e.kind == "summary")
    # The nested calls are inside the span, so only the span itself counts.
    assert t.total_duration_ms() == span.duration_ms
    assert t.total_duration_ms() < 4000


def test_top_level_llm_call_still_counts():
    t = Trace()
    t.emit_llm("solo", model="m", tokens_in=1, tokens_out=1, cost_usd=0.0, duration_ms=1500)
    assert t.total_duration_ms() == 1500


def test_cost_and_tokens_still_sum_over_every_event():
    t = Trace()
    t.start("writer", kind="summary")
    t.emit_llm("writer", model="m", tokens_in=10, tokens_out=5, cost_usd=0.01, duration_ms=10)
    t.end()
    assert t.total_cost_usd() == pytest.approx(0.01)
    assert t.total_tokens() == (10, 5)


def test_unpaired_end_does_not_pop_someone_elses_span():
    """`end()` popped unconditionally; a stray call consumed a live frame and the
    next real `end()` mis-attributed its duration to the wrong actor."""
    t = Trace()
    t.end()  # nothing open — must be a no-op, not an IndexError
    assert t.events == []

    t.start("planner", kind="summary")
    t.end()
    assert [e.actor for e in t.events] == ["planner"]


def test_span_depth_is_recorded():
    t = Trace()
    t.start("outer", kind="summary")
    t.emit("tool_call", "outer")
    t.start("inner", kind="summary")
    t.emit("tool_call", "inner")
    t.end()
    t.end()
    by_actor = {(e.actor, e.kind): e.depth for e in t.events}
    assert by_actor[("outer", "tool_call")] == 1
    assert by_actor[("inner", "tool_call")] == 2
    assert by_actor[("inner", "summary")] == 1
    assert by_actor[("outer", "summary")] == 0
