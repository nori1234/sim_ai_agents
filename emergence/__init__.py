"""Emergence World — an autonomous multi-agent city simulation.

A small, dependency-free engine that drops a population of AI agents into a
virtual town with resources, governance and the freedom to cooperate or harm
one another, then runs them for many simulated days and measures what kind of
society emerges.

The agent "brain" (decision policy) is pluggable: an offline, deterministic
``HeuristicBrain`` ships in the box so the whole simulation runs with nothing
but the Python standard library, and an ``LLMBrain`` adapter can drive agents
with a real model (Llama via any OpenAI-compatible endpoint, or Anthropic).
"""

from .world import World, Facility
from .agent import Agent
from .simulation import Simulation, SimulationConfig
from .metrics import Metrics

__all__ = [
    "World",
    "Facility",
    "Agent",
    "Simulation",
    "SimulationConfig",
    "Metrics",
]

__version__ = "0.1.0"
