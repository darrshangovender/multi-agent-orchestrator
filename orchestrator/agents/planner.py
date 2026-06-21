"""Planner: decomposes a question into ordered sub-questions."""

from __future__ import annotations

import json
import re

from pydantic import Field

from ..agent import Agent
from ..core import AgentResult, Handoff, Result, Workspace
from ..trace import Trace


class PlanInput(Handoff):
    question: str


class PlanOutput(Result):
    sub_questions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class Planner(Agent):
    role = "planner"
    system_prompt = (
        "You decompose research questions into ordered sub-questions. "
        "Output strict JSON: {\"reasoning\": \"...\", \"sub_questions\": [\"q1\", \"q2\", ...]}. "
        "3-6 sub-questions. Each is specifically searchable. No filler."
    )

    def run(self, handoff: PlanInput, *, workspace: Workspace, trace: Trace) -> AgentResult:
        text = self.call_llm(
            messages=[{"role": "user", "content": handoff.question}],
            trace=trace, max_tokens=512, temperature=0.3,
        )
        data = _extract_json(text)
        result = PlanOutput(**data)
        return AgentResult(result=result, workspace_writes={"sub_questions": result.sub_questions, "plan_reasoning": result.reasoning})


def _extract_json(text: str) -> dict:
    """Be forgiving: try strict parse, then extract first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in planner output: {text[:200]!r}")
    return json.loads(m.group(0))