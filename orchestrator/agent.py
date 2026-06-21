"""Base Agent class. Subclass per role.

Agents are intentionally thin: model choice + system prompt + tool registry +
the `run` method that turns a typed handoff into a typed result. Everything
else (loop control, retries, workspace) is the orchestrator's job.
"""

from __future__ import annotations

from typing import Any

from .core import AgentResult, Handoff, Workspace
from .llm import LLM
from .tools import ToolRegistry
from .trace import Trace


class Agent:
    """Subclass and override `run(handoff, workspace, trace) -> AgentResult`."""

    role: str = "agent"
    model: str = "claude-sonnet-4-5"
    system_prompt: str = ""

    def __init__(self, *, model: str | None = None, tools: list | None = None, system_prompt: str | None = None) -> None:
        if model is not None:
            self.model = model
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.llm = LLM(self.model)
        self.tools = ToolRegistry(tools or [])

    def run(self, handoff: Handoff, *, workspace: Workspace, trace: Trace) -> AgentResult:
        raise NotImplementedError

    # ---- helper used by subclasses ----
    def call_llm(self, *, messages: list[dict], trace: Trace, max_tokens: int = 1024, temperature: float = 0.4) -> str:
        resp = self.llm.chat(messages, system=self.system_prompt, max_tokens=max_tokens, temperature=temperature)
        trace.emit_llm(self.role, model=resp.model, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd, duration_ms=resp.duration_ms)
        return resp.content