"""Orchestrator + Workspace + Handoff + Result.

Design choices:
  - Handoff/Result are Pydantic models that each agent declares. Type-safe.
  - Workspace is a dict-of-namespaces shared between agents. Keeps token cost down
    because we don't replay conversation history; we just hand the relevant slot
    to the next agent.
  - The orchestrator's `run` method is a plain loop — read it, modify it. No DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .trace import Trace


class Handoff(BaseModel):
    """Base for agent inputs. Subclass per agent role to pin a strict schema."""
    pass


class Result(BaseModel):
    """Base for agent outputs. Subclass per agent role."""
    pass


@dataclass
class AgentResult:
    """What an agent returns to the orchestrator: typed result + workspace writes."""
    result: Result
    workspace_writes: dict[str, Any] = field(default_factory=dict)


class Workspace:
    """Key-value shared memory for agents in one run.

    Reasoning for key-value over a message-log:
      - Cheaper: don't replay the whole conversation to every agent
      - Inspectable: each slot is a named, typed artifact
      - Composable: the writer reads ws["sources"] without knowing who put it there
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})
        self._history: list[tuple[str, str, Any]] = []  # (actor, key, value) for trace

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any, *, actor: str = "unknown") -> None:
        self._data[key] = value
        self._history.append((actor, key, value))

    def update(self, writes: dict[str, Any], *, actor: str = "unknown") -> None:
        for k, v in writes.items():
            self.set(k, v, actor=actor)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

    def history(self) -> list[tuple[str, str, Any]]:
        return list(self._history)


class Orchestrator:
    """Runs a sequence of agents with structured handoffs and a shared workspace.

    The default `run` method implements a simple linear pipeline:
        planner -> researcher -> writer -> critic (loop if needed) -> done

    For a different topology, subclass and override `run`. The loop is 30 lines
    of plain Python; that is on purpose.
    """

    def __init__(self, agents: dict[str, Any], workspace: Workspace | None = None, max_total_iterations: int = 10):
        self.agents = agents
        self.workspace = workspace or Workspace()
        self.trace = Trace()
        self.max_total_iterations = max_total_iterations

    def step(self, agent_name: str, handoff: Handoff) -> AgentResult:
        """Run one agent. Validates input + output against its declared schemas."""
        agent = self.agents[agent_name]
        self.trace.emit("handoff_in", agent_name, handoff_type=type(handoff).__name__)
        self.trace.start(agent_name, kind="summary")
        try:
            result = agent.run(handoff, workspace=self.workspace, trace=self.trace)
        except Exception as e:
            self.trace.emit("error", agent_name, error=str(e)[:300])
            raise
        # Validate result is the right type for this agent
        if not isinstance(result, AgentResult):
            raise TypeError(f"{agent_name}.run() must return AgentResult, got {type(result).__name__}")
        self.workspace.update(result.workspace_writes, actor=agent_name)
        self.trace.end(result_type=type(result.result).__name__)
        self.trace.emit("handoff_out", agent_name, result_type=type(result.result).__name__)
        return result

    def run(self, question: str) -> Any:
        """Default linear pipeline. Override for fan-out or branching."""
        from .agents import PlanInput, ResearchInput, WriteInput, CriticInput

        # 1. Plan
        plan = self.step("planner", PlanInput(question=question)).result

        # 2. Research each sub-question
        self.step("researcher", ResearchInput(sub_questions=plan.sub_questions))

        # 3. Write + critique loop
        max_revisions = self.agents.get("critic").max_revisions if "critic" in self.agents else 2
        for revision in range(max_revisions + 1):
            self.step("writer", WriteInput(question=question, revision=revision))
            critic_result = self.step("critic", CriticInput(question=question, revision=revision)).result
            if critic_result.accept or revision >= max_revisions:
                break

        # 4. Final
        return _Final(
            final_answer=self.workspace.get("draft", ""),
            sources=self.workspace.get("sources", []),
            critic_score=self.workspace.get("critic_score"),
            trace=self.trace,
        )


@dataclass
class _Final:
    final_answer: str
    sources: list
    critic_score: float | None
    trace: Trace