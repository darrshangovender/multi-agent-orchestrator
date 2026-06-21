"""multi-agent-orchestrator — production-grade multi-agent system with full observability."""

from .core import Orchestrator, Workspace, Handoff, Result, AgentResult
from .agent import Agent
from .trace import Trace, TraceEvent
from .tools import tool, ToolRegistry

__version__ = "0.1.0"
__all__ = [
    "Orchestrator", "Workspace", "Handoff", "Result", "AgentResult",
    "Agent", "Trace", "TraceEvent", "tool", "ToolRegistry",
]