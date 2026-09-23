"""The four built-in agents: Planner, Researcher, Writer, Critic.

Plus the typed Handoff/Result models they declare.
"""

from .critic import Critic, CriticInput, CriticOutput
from .planner import PlanInput, Planner, PlanOutput
from .researcher import Researcher, ResearchInput, ResearchOutput
from .writer import WriteInput, WriteOutput, Writer

__all__ = [
    "Critic",
    "CriticInput",
    "CriticOutput",
    "PlanInput",
    "PlanOutput",
    "Planner",
    "ResearchInput",
    "ResearchOutput",
    "Researcher",
    "WriteInput",
    "WriteOutput",
    "Writer",
]