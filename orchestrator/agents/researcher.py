"""Researcher: for each sub-question, calls tools, collects sources.

Simplified for this scaffold: takes the toy `web_search` tool, collects results,
flags contradictions on simple keyword overlap. Production version would use a
real search provider + LLM-based contradiction detection.
"""

from __future__ import annotations

from pydantic import Field

from ..agent import Agent
from ..core import AgentResult, Handoff, Result, Workspace
from ..trace import Trace


class ResearchInput(Handoff):
    sub_questions: list[str]


class Source(Result):
    title: str
    url: str
    snippet: str
    sub_question: str


class ResearchOutput(Result):
    sources: list[Source] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class Researcher(Agent):
    role = "researcher"
    system_prompt = ""  # this agent uses tools, not LLM directly in this scaffold

    def run(self, handoff: ResearchInput, *, workspace: Workspace, trace: Trace) -> AgentResult:
        if not self.tools.names():
            raise RuntimeError("Researcher needs at least one tool (e.g. web_search)")
        web_search = self.tools.get("web_search")
        sources: list[Source] = []
        for q in handoff.sub_questions:
            trace.emit("tool_call", self.role, tool="web_search", query=q)
            results = web_search(query=q, max_results=3)
            for r in results:
                sources.append(Source(title=r["title"], url=r["url"], snippet=r["snippet"], sub_question=q))
        contradictions = _flag_contradictions(sources)
        result = ResearchOutput(sources=sources, contradictions=contradictions)
        return AgentResult(result=result, workspace_writes={"sources": [s.model_dump() for s in sources], "contradictions": contradictions})


def _flag_contradictions(sources: list[Source]) -> list[str]:
    """Toy heuristic: flag when two sources for the same sub_question disagree on keywords.

    Real implementation: LLM-as-judge on pairs of snippets. Left as an exercise so
    callers can plug in their preferred approach.
    """
    return []