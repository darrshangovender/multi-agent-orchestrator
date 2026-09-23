"""multi-agent-orchestrator — production-grade multi-agent system with full observability."""

from .agent import Agent
from .core import AgentResult, Handoff, Orchestrator, Result, Workspace
from .tools import ToolRegistry, tool
from .trace import Trace, TraceEvent

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentResult",
    "Handoff",
    "Orchestrator",
    "Result",
    "ToolRegistry",
    "Trace",
    "TraceEvent",
    "Workspace",
    "tool",
]