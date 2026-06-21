"""Critic: scores the draft, decides accept-vs-revise.

The critic is intentionally strict on a fixed rubric:
  - Specific, no filler           (0-2)
  - Citation per factual claim    (0-3)
  - Addresses the question        (0-3)
  - Tight 250-350 words           (0-2)

Total /10. >= 8.0 to accept. The orchestrator caps revisions at `max_revisions`.
"""

from __future__ import annotations

import json
import re

from pydantic import Field

from ..agent import Agent
from ..core import AgentResult, Handoff, Result, Workspace
from ..trace import Trace


class CriticInput(Handoff):
    question: str
    revision: int = 0


class CriticOutput(Result):
    score: float = 0.0
    accept: bool = False
    feedback: str = ""


CRITIC_SYSTEM = """\
You are a strict editor scoring a draft answer. Rubric:
  - specific_no_filler:  0-2 (specificity, no padding)
  - citation_coverage:   0-3 (every factual claim has a citation)
  - addresses_question:  0-3 (directly answers what was asked)
  - tight_length:        0-2 (250-350 words, no fluff)

Output strict JSON:
{
  "score": <float 0..10>,
  "feedback": "<one short paragraph, actionable, no praise>",
  "accept": <bool — true iff score >= 8.0>
}
"""


class Critic(Agent):
    role = "critic"
    system_prompt = CRITIC_SYSTEM

    def __init__(self, *, max_revisions: int = 2, **kw):
        super().__init__(**kw)
        self.max_revisions = max_revisions

    def run(self, handoff: CriticInput, *, workspace: Workspace, trace: Trace) -> AgentResult:
        draft = workspace.get("draft", "")
        if not draft:
            raise RuntimeError("Critic: no draft in workspace")
        text = self.call_llm(
            messages=[{"role": "user", "content": f"Question: {handoff.question}\n\nDraft:\n{draft}\n"}],
            trace=trace, max_tokens=400, temperature=0.2,
        )
        data = _extract_json(text)
        result = CriticOutput(**data)
        return AgentResult(result=result, workspace_writes={
            "critic_score": result.score,
            "critic_feedback": result.feedback,
            "critic_accept": result.accept,
        })


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in critic output: {text[:200]!r}")
    return json.loads(m.group(0))