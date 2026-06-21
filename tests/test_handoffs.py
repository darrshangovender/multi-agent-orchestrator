"""Basic tests using a stub LLM so we never hit a real API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator import Orchestrator, Workspace
from orchestrator.agents import PlanInput, PlanOutput
from orchestrator.core import AgentResult


def test_workspace_set_get():
    ws = Workspace()
    ws.set("k", "v", actor="test")
    assert ws.get("k") == "v"
    assert ws.get("missing", "default") == "default"


def test_workspace_history_logs_actor():
    ws = Workspace()
    ws.update({"a": 1, "b": 2}, actor="planner")
    history = ws.history()
    assert len(history) == 2
    assert all(h[0] == "planner" for h in history)


def test_handoff_validates_input():
    # PlanInput requires `question`
    with pytest.raises(ValidationError):
        PlanInput()  # type: ignore[call-arg]


def test_plan_output_defaults():
    p = PlanOutput()
    assert p.sub_questions == []
    assert p.reasoning == ""


def test_orchestrator_rejects_non_agent_result():
    """The orchestrator step() must reject anything that is not AgentResult."""
    class BadAgent:
        def run(self, *a, **k):
            return "not an agent result"
    orch = Orchestrator(agents={"bad": BadAgent()})
    with pytest.raises(TypeError, match="AgentResult"):
        orch.step("bad", PlanInput(question="x"))