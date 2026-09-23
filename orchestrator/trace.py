"""Structured trace events. Every step in the orchestrator emits one of these.

Design choice: we use plain dataclasses + a list, not OpenTelemetry. OT is the
right tool when you have a service mesh; for a single-process agent loop it is
overkill. We can export to OT later with a 30-line adapter — keeping the core
zero-dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

EventKind = Literal["handoff_in", "handoff_out", "llm_call", "tool_call", "workspace_write", "error", "summary"]


@dataclass
class TraceEvent:
    kind: EventKind
    actor: str                            # which agent / orchestrator step
    ts: datetime
    duration_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    #: Span nesting depth at which this event was recorded. 0 means top-level.
    #: Needed to total durations without counting a span and its children twice.
    depth: int = 0


class Trace:
    """Append-only log of events for one orchestrator run."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._stack: list[tuple[str, EventKind, float]] = []

    def emit(self, kind: EventKind, actor: str, **payload: Any) -> None:
        self.events.append(TraceEvent(
            kind=kind, actor=actor,
            ts=datetime.now(UTC),
            payload=payload,
            depth=len(self._stack),
        ))

    def emit_llm(self, actor: str, *, model: str, tokens_in: int, tokens_out: int, cost_usd: float | None, duration_ms: int) -> None:
        self.events.append(TraceEvent(
            kind="llm_call", actor=actor,
            ts=datetime.now(UTC),
            duration_ms=duration_ms,
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd,
            payload={"model": model},
            depth=len(self._stack),
        ))

    def start(self, actor: str, kind: EventKind) -> None:
        """Mark the beginning of a timed span — pair with `end`."""
        self._stack.append((actor, kind, time.perf_counter()))

    def end(self, **payload: Any) -> None:
        if not self._stack:
            # Unpaired end. Silently closing nothing beats popping a frame that
            # belongs to a different span and mis-attributing its duration.
            return
        actor, kind, t0 = self._stack.pop()
        dur = int((time.perf_counter() - t0) * 1000)
        self.events.append(TraceEvent(
            kind=kind, actor=actor,
            ts=datetime.now(UTC),
            duration_ms=dur, payload=payload,
            depth=len(self._stack),
        ))

    # ---- summarisation ----

    def total_cost_usd(self) -> float:
        return sum(e.cost_usd or 0.0 for e in self.events)

    def total_tokens(self) -> tuple[int, int]:
        return (sum(e.tokens_in for e in self.events), sum(e.tokens_out for e in self.events))

    def total_duration_ms(self) -> int:
        """Wall-clock for the run, in ms.

        Only top-level events count. A timed span encloses the llm_call/tool_call
        events emitted inside it and *both* carry duration_ms, so summing every
        event counted the same elapsed time twice: a 7-call run taking 14s
        reported `wallclock=28.0s`, and the overstatement grew with nesting depth.
        """
        return sum(e.duration_ms for e in self.events if e.depth == 0)

    def summary(self) -> str:
        llm_calls = [e for e in self.events if e.kind == "llm_call"]
        tool_calls = [e for e in self.events if e.kind == "tool_call"]
        ti, to = self.total_tokens()
        return (
            f"events={len(self.events)} "
            f"llm_calls={len(llm_calls)} "
            f"tool_calls={len(tool_calls)} "
            f"tokens={ti}+{to} "
            f"cost=${self.total_cost_usd():.4f} "
            f"wallclock={self.total_duration_ms()/1000:.1f}s"
        )

    def for_actor(self, actor: str) -> list[TraceEvent]:
        return [e for e in self.events if e.actor == actor]