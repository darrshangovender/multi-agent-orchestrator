"""The four built-in agents: Planner, Researcher, Writer, Critic.

Plus the typed Handoff/Result models they declare.
"""

from .planner import Planner, PlanInput, PlanOutput
from .researcher import Researcher, ResearchInput, ResearchOutput
from .writer import Writer, WriteInput, WriteOutput
from .critic import Critic, CriticInput, CriticOutput

__all__ = [
    "Planner", "PlanInput", "PlanOutput",
    "Researcher", "ResearchInput", "ResearchOutput",
    "Writer", "WriteInput", "WriteOutput",
    "Critic", "CriticInput", "CriticOutput",
]