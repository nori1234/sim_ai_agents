"""The pluggable decision policy interface.

Anything that can turn an :class:`Observation` into an :class:`Action` is a
brain. The engine never calls a model directly; it only ever asks a brain to
``decide``. That keeps the offline heuristic, a local Llama, and a hosted
Anthropic model perfectly interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..actions import Action
from ..agent import Agent
from ..observation import Observation


class AgentBrain(ABC):
    """Base class for all decision policies."""

    #: Human-readable name shown in reports.
    name: str = "brain"

    @abstractmethod
    def decide(self, agent: Agent, obs: Observation) -> Action:
        """Return the action ``agent`` will attempt this turn."""
        raise NotImplementedError
