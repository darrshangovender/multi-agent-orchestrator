"""Writer: produces a sourced draft from the workspace.

Reads `sources` and `sub_questions` from the workspace, generates a draft with
inline citation markers like [1], [2]. The critic uses those markers to score.
"""

from __future__ import annotations

from pydantic import Field

from ..agent import Agent
from ..core import AgentResult, Handoff, Result, Workspace
from ..trace import Trace


class WriteInput(Handoff):
    question: str
    revision: int = 0  # 0 = first draft, N = N'th revision


class WriteOutput(Result):
    draft: str = ""
    citation_count: int = 0


WRITER_SYSTEM = """\
You are a technical writer. You will receive:
  - A question
  - A list of sources (with [n] markers)
  - Optional revision feedback from a critic

Produce a 250-350 word answer that:
  - Cites sources inline as [1], [2], etc.
  - Has at least one citation per factual claim
  - Is direct and specific. No filler.

If revision feedback is provided, address each point explicitly. Do not silently ignore feedback.
"""


class Writer(Agent):
    role = "writer"
    system_prompt = WRITER_SYSTEM

    def run(self, handoff: WriteInput, *, workspace: Workspace, trace: Trace) -> AgentResult:
        sources = workspace.get("sources", [])
        feedback = workspace.get("critic_feedback", "")
        # Format sources for the prompt
        sources_text = "\n".join(
            f"[{i+1}] {s['title']} — {s['url']}\n    {s['snippet']}"
            for i, s in enumerate(sources)
        )
        user = f"Question: {handoff.question}\n\nSources:\n{sources_text}\n"
        if handoff.revision > 0 and feedback:
            user += f"\nRevision {handoff.revision} feedback from critic:\n{feedback}\n"
        draft = self.call_llm(
            messages=[{"role": "user", "content": user}],
            trace=trace, max_tokens=800, temperature=0.5,
        )
        citation_count = draft.count("[")  # crude but works for [n] markers
        result = WriteOutput(draft=draft, citation_count=citation_count)
        return AgentResult(result=result, workspace_writes={"draft": draft, "draft_revision": handoff.revision, "citation_count": citation_count})