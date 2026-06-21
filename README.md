<div align="center">

# multi-agent-orchestrator — production-grade multi-agent system with full observability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-CC785C)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.7+-E92063?logo=pydantic&logoColor=white)](https://pydantic.dev)
[![Status](https://img.shields.io/badge/Status-Working%20code-blue)](#)

</div>

---

> A small but production-shaped framework for **role-based multi-agent systems**. Agents have explicit roles, talk to each other through **structured Pydantic handoffs** (never freeform text), share a workspace memory, register tools, and emit a full trace of every step. Ships with a research-writer-critic example that produces a sourced answer with no orchestrator-level glue code.

**Why this exists.** Most "agent frameworks" are either too magic (LangGraph hides the control flow) or too thin (you re-implement the orchestration loop every project). This library exposes the loop as code you can read, while giving you the parts that actually deserve abstraction: structured handoffs, workspace memory, tool registration, and tracing. Two hundred lines covers 80% of the agent systems I've shipped.

---

## The core loop

```
┌──────────────────────────────────────────────────────────────┐
│  Orchestrator                                                 │
│                                                               │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│   │ Planner │ ─▶ │Researchr│ ─▶ │ Writer  │ ─▶ │ Critic  │  │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│        │              │              │              │        │
│        └──────────────┴──── Workspace ─────────────┘        │
│                                                               │
│                       ┌──── Trace ────┐                      │
│                       │ every step    │                      │
│                       │ logged        │                      │
│                       └───────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

Each agent receives a typed **Handoff** from the orchestrator, returns a typed **Result**, and writes intermediate observations to the **Workspace**. The orchestrator decides what runs next based on the result schema, not on parsing freeform text.

## Example: research-writer-critic

```python
from orchestrator import Orchestrator, Workspace
from orchestrator.agents import Planner, Researcher, Writer, Critic
from orchestrator.tools import web_search

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

result = orch.run(
    "Write a 300-word answer with sources to: "
    "What is the difference between RAG and fine-tuning?"
)
print(result.final_answer)
print(result.trace.summary())
```

Output (abbreviated):

```
[planner]    decomposed into 4 sub-questions
[researcher] gathered 6 sources, 2 contradictions flagged
[writer]     drafted 312 words, 4 citations
[critic]     rev 1 → score 7.5/10, "tighten the 'when to fine-tune' paragraph"
[writer]     rev 2 → 287 words, 5 citations
[critic]     rev 2 → score 9.0/10, accept
[orch]       total 7 LLM calls, 14.2s, $0.18
```

## What makes this different from "just call the API in a loop"

1. **Structured handoffs only.** Every agent declares its input + output type as Pydantic models. The orchestrator validates at the boundary — invalid output fails fast, doesn't silently propagate. No "the LLM returned something weird and we kept going".
2. **Workspace, not message history.** Agents read and write to a shared key-value workspace. Saves tokens (don't replay the whole conversation), enforces a real schema (sources, drafts, scores live in named slots), and makes debugging trivial — every workspace state is a snapshot.
3. **Trace, not logs.** Every step (handoff in, tool call, LLM call, handoff out) is a structured trace event with timestamps, tokens, cost. Replay the run, summarize cost, export for dashboards.
4. **Max-iteration safety.** Every agent loop has a hard cap. No "ran 47 times costing $400 because the critic kept rejecting drafts".
5. **Provider portable.** Same agent code runs on Anthropic, OpenAI, or any provider with a chat-completions-shaped API. Picks model per agent role.

## Repo structure

```
.
├── orchestrator/
│   ├── __init__.py
│   ├── core.py            # Orchestrator + Workspace + Handoff + Result
│   ├── agent.py           # Agent base class
│   ├── trace.py           # Structured trace events + summary
│   ├── llm.py             # provider-portable LLM client
│   ├── tools.py           # @tool decorator + tool registry
│   └── agents/
│       ├── __init__.py
│       ├── planner.py     # Decomposes the question into sub-tasks
│       ├── researcher.py  # Uses tools, gathers sources, flags contradictions
│       ├── writer.py      # Produces draft from workspace
│       └── critic.py      # Scores + decides revise vs accept
├── examples/
│   └── research_demo.py
├── tests/
│   └── test_handoffs.py
└── pyproject.toml
```

## Why no `pip install langgraph`

LangGraph is excellent, but it hides the orchestration as a graph DSL. You don't get to read the loop. For shipping into someone else's codebase you want exactly the opposite: a loop they can step through with a debugger and modify with a normal PR. This library is that loop, exposed.

## Status

- [x] Orchestrator + Workspace + structured Handoff/Result
- [x] Agent base class with system prompt + tool binding
- [x] Provider-portable LLM client (Anthropic + OpenAI)
- [x] Tool decorator + tool registry
- [x] Full trace: handoffs, LLM calls, tool calls, cost, latency
- [x] Planner / Researcher / Writer / Critic agents
- [x] Max-iteration safety per agent
- [ ] Parallel agent execution (next: fan-out for independent sub-tasks)
- [ ] Streaming handoffs (intermediate updates during long tools)

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)