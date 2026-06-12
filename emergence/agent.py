"""Agents: the inhabitants of the town, each with a job, memory and needs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


MAX_ENERGY = 100.0
MAX_MEMORY = 40


@dataclass
class Agent:
    """A single inhabitant.

    Survival hinges on ``energy``: it decays every tick and is replenished by
    eating food. An agent that hits zero energy starves — reproducing the
    failure mode where a population that talks but never feeds itself dies out.
    """

    id: str
    name: str
    profession: str
    persona: str  # personality archetype key (see personas.py)
    x: int
    y: int

    energy: float = MAX_ENERGY
    money: int = 20
    inventory: dict[str, int] = field(default_factory=lambda: {"food": 3, "materials": 0})

    alive: bool = True
    day_of_death: Optional[int] = None
    cause_of_death: Optional[str] = None

    # Social state.
    memory: list[str] = field(default_factory=list)
    trust: dict[str, float] = field(default_factory=dict)  # other_id -> [-1, 1]

    # Per-agent tallies, useful for the post-mortem report.
    crimes_committed: int = 0
    times_victimized: int = 0
    proposals_made: int = 0
    votes_cast: int = 0
    collaborations: int = 0
    frauds_committed: int = 0

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    @pos.setter
    def pos(self, value: tuple[int, int]) -> None:
        self.x, self.y = value

    # -- needs -------------------------------------------------------------
    def food(self) -> int:
        return self.inventory.get("food", 0)

    def materials(self) -> int:
        return self.inventory.get("materials", 0)

    def add(self, resource: str, amount: int) -> None:
        self.inventory[resource] = self.inventory.get(resource, 0) + amount

    def take(self, resource: str, amount: int) -> int:
        """Remove up to ``amount`` of a resource; return how much was taken."""
        have = self.inventory.get(resource, 0)
        taken = min(have, max(0, amount))
        self.inventory[resource] = have - taken
        return taken

    # -- social ------------------------------------------------------------
    def remember(self, entry: str) -> None:
        self.memory.append(entry)
        if len(self.memory) > MAX_MEMORY:
            self.memory = self.memory[-MAX_MEMORY:]

    def trust_of(self, other_id: str) -> float:
        return self.trust.get(other_id, 0.0)

    def adjust_trust(self, other_id: str, delta: float) -> None:
        new = self.trust_of(other_id) + delta
        self.trust[other_id] = max(-1.0, min(1.0, new))

    def die(self, day: int, cause: str) -> None:
        self.alive = False
        self.day_of_death = day
        self.cause_of_death = cause

    def snapshot(self) -> dict:
        """A compact view used both for observations and for reporting."""
        return {
            "id": self.id,
            "name": self.name,
            "profession": self.profession,
            "alive": self.alive,
            "energy": round(self.energy, 1),
            "money": self.money,
            "food": self.food(),
            "materials": self.materials(),
            "crimes": self.crimes_committed,
            "frauds": self.frauds_committed,
        }
