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
    inventory: dict[str, int] = field(default_factory=lambda: {"food": 3, "materials": 0})
    # Money is NOT a privileged field — it is an ordinary inventory item,
    # conserved by the same add/take physics as food and materials. This stays
    # a constructor argument for convenience, but it is folded into
    # ``inventory["money"]`` and read back through the ``money`` property below.
    money: int = 20

    # The three primal urges (only active when DrivesConfig is enabled). Each is
    # an instinctual pressure that builds over time and is discharged — with a
    # hit of pleasure — by the matching act. Behaviour is driven by these urges
    # and the pleasure of relieving them, not by rational calculation.
    #   hunger  : 0 = full,   100 = starving    (rises over time, EAT to lower)
    #   fatigue : 0 = rested, 100 = exhausted   (rises over time, SLEEP to lower)
    #   libido  : 0 = sated,  100 = desperate   (rises over time, MATE to lower)
    hunger: float = 0.0
    fatigue: float = 0.0
    libido: float = 0.0
    pleasure: float = 0.0         # lifetime pleasure accumulated (wellbeing)
    age_days: int = 99            # seeded adults start mature; newborns start at 0
    last_reproduced_day: Optional[int] = None
    parent_ids: tuple[str, ...] = ()

    # Higher (social) needs — esteem/honour/power. Only active under StatusConfig.
    #   esteem     : the urge to be recognised (rises over time, praise relieves)
    #   reputation : standing/honour in others' eyes (earned by deeds, decays)
    esteem: float = 0.0
    reputation: float = 0.0
    praise_received: int = 0
    praise_given: int = 0
    times_mayor: int = 0

    # Psyche layer — safety (fear) and self-actualization. Under PsycheConfig.
    #   fear        : struck by suffering/witnessing crime; decays in safety
    #   fulfillment : deep satisfaction earned by creating works
    fear: float = 0.0
    fulfillment: float = 0.0
    works_created: int = 0

    # Society layer — weapons, addiction, gang and faith affiliation.
    weapons: int = 0
    addiction: float = 0.0
    gang_id: Optional[str] = None
    faith: Optional[str] = None
    rebellions_joined: int = 0
    last_rebelled_day: Optional[int] = None

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
    children: int = 0

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
            "hunger": round(self.hunger, 1),
            "fatigue": round(self.fatigue, 1),
            "libido": round(self.libido, 1),
            "reputation": round(self.reputation, 1),
            "fear": round(self.fear, 1),
            "weapons": self.weapons,
            "addiction": round(self.addiction, 1),
            "gang": self.gang_id,
            "faith": self.faith,
            "age_days": self.age_days,
            "crimes": self.crimes_committed,
            "frauds": self.frauds_committed,
        }


# Money lives in the inventory, not in a privileged scalar field. We install
# the accessor after the dataclass is built so the ``money=`` constructor
# argument (assigned in __init__ after ``inventory``) flows through the setter
# into ``inventory["money"]``. From here on, money is conserved by the same
# add/take physics as every other tradable good.
def _money_get(self: Agent) -> int:
    return self.inventory.get("money", 0)


def _money_set(self: Agent, value: int) -> None:
    self.inventory["money"] = int(value)


Agent.money = property(_money_get, _money_set)
