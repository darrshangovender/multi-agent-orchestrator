"""Demo: research-writer-critic pipeline answers a question.

Requires either ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.

Run:
    uv run python examples/research_demo.py
"""

from __future__ import annotations

from orchestrator import Orchestrator, Workspace
from orchestrator.agents import Planner, Researcher, Writer, Critic
from orchestrator.tools import web_search


def main() -> None:
    ws = Workspace()
    orch = Orchestrator(
        agents={
            "planner":    Planner(model="claude-sonnet-4-5"),
            "researcher": Researcher(tools=[web_search]),
            "writer":     Writer(model="claude-sonnet-4-5"),
            "critic":     Critic(model="claude-opus-4-7", max_revisions=2),
        },
        workspace=ws,
    )
    question = "What is the practical difference between RAG and fine-tuning for adapting an LLM to a domain?"
    final = orch.run(question)

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final.final_answer)
    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, s in enumerate(final.sources, 1):
        print(f"[{i}] {s['title']} — {s['url']}")
    print()
    print("=" * 70)
    print("TRACE SUMMARY")
    print("=" * 70)
    print(final.trace.summary())


if __name__ == "__main__":
    main()