"""Provider-portable LLM client.

Why not just use the provider SDK directly: agents need to be swappable across
providers. Production reality: a planner agent runs on Sonnet, the critic runs
on Opus, the researcher runs on Haiku. A thin shared interface lets each agent
declare its own model without the agent code knowing the provider.

The interface is deliberately tiny: one method, `chat(messages, ...) -> Response`.
We do NOT abstract over tool-use schemas — those genuinely differ and agents
that use tools are written per-provider. Don't fight the SDK on tools.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


# Per-1M-token USD pricing. Update quarterly.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5":   (3.00, 15.00),
    "claude-sonnet-4-7":   (3.00, 15.00),
    "claude-haiku-4-5":    (0.80, 4.00),
    "claude-opus-4-7":     (15.00, 75.00),
    "claude-opus-4-8":     (15.00, 75.00),
    "gpt-4o":              (2.50, 10.00),
    "gpt-4o-mini":         (0.15, 0.60),
}


@dataclass
class Response:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    cost_usd: float | None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


def _cost(model: str, ti: int, to: int) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    return (ti / 1_000_000) * p[0] + (to / 1_000_000) * p[1]


class LLM:
    """Thin shared client. Picks the provider based on model name prefix."""

    def __init__(self, model: str):
        self.model = model
        self.provider = "anthropic" if model.startswith("claude-") else "openai"

    def chat(self, messages: list[dict], *, system: str = "", max_tokens: int = 1024, temperature: float = 0.4) -> Response:
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            return self._anthropic(messages, system, max_tokens, temperature, t0)
        return self._openai(messages, system, max_tokens, temperature, t0)

    def _anthropic(self, messages, system, max_tokens, temperature, t0) -> Response:
        from anthropic import Anthropic  # lazy import — only required if used
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens,
            "temperature": temperature, "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        ti = resp.usage.input_tokens
        to = resp.usage.output_tokens
        return Response(
            content=resp.content[0].text,
            model=self.model,
            tokens_in=ti, tokens_out=to,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=_cost(self.model, ti, to),
        )

    def _openai(self, messages, system, max_tokens, temperature, t0) -> Response:
        from openai import OpenAI  # lazy import
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages
        resp = client.chat.completions.create(
            model=self.model, messages=full_messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        ti = resp.usage.prompt_tokens
        to = resp.usage.completion_tokens
        return Response(
            content=resp.choices[0].message.content or "",
            model=self.model,
            tokens_in=ti, tokens_out=to,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=_cost(self.model, ti, to),
        )