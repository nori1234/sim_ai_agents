"""The underworld and culture layer: weapons, drugs, gangs, faith — and the
emergent *roles* facilities take on when these forces move through the town.

Everything here is opt-in via :data:`SocietyConfig.enabled` (default ``False``)
and interlocks with the rest of the simulation:

* **Weapons & rebellion** — agents craft arms at a workshop (which becomes a
  *weapons factory*). Arms make violence deadlier and embolden the discontented;
  when enough armed malcontents share a grievance they *rebel* and can depose
  the mayor.
* **Drugs** — a narcotic can be produced and dealt. A dose is a jolt of energy
  and pleasure, but it builds *addiction*; addicts then chase the next hit over
  food and safety, and suffer withdrawal. Drug spots become *dens*.
* **Gangs** — alienated, aggressive agents band into gangs: fierce loyalty
  within, hostility without. Gangs claim facilities as *turf*, run rackets, and
  feud with rivals.
* **Religion** — a high-standing agent may found a faith and win converts.
  Worship at a *temple* eases fear and grants belonging, binding the faithful —
  a force for cohesion that can also harden into in-group/out-group lines.

Facilities thus acquire emergent **roles** (weapons factory, drug den, gang
turf, temple) on top of their built-in function.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SocietyConfig:
    enabled: bool = False
    # Sub-systems (all on when the layer is enabled; toggle individually if wanted).
    weapons: bool = True
    drugs: bool = True
    gangs: bool = True
    religion: bool = True

    # -- weapons & rebellion (武器・反乱) -------------------------------
    weapon_material_cost: int = 1
    weapon_attack_bonus: float = 14.0     # extra damage an armed attacker deals
    rebellion_discontent: float = 55.0    # personal discontent needed to revolt
    rebellion_min_rebels: int = 3         # armed malcontents needed to topple power
    rebellion_cooldown_days: int = 4

    # -- drugs (薬物) ----------------------------------------------------
    drug_material_cost: int = 1
    drug_price: int = 4
    drug_pleasure: float = 7.0            # the hit (banked as pleasure)
    drug_energy_spike: float = 20.0
    addiction_per_dose: float = 24.0
    addiction_decay_per_tick: float = 2.0
    withdrawal_threshold: float = 45.0    # above this, craving + harm set in
    withdrawal_energy_penalty: float = 4.0

    # -- gangs (ギャング) ------------------------------------------------
    gang_form_min_aggression: float = 0.4
    gang_loyalty: float = 0.4             # trust granted among gang-mates
    gang_rival_hostility: float = 0.5     # distrust toward rival gangs
    gang_join_radius: int = 6

    # -- religion (宗教) -------------------------------------------------
    faith_min_reputation: float = 5.0     # standing needed to found a faith
    conversion_radius: int = 5
    worship_fear_relief: float = 28.0
    worship_pleasure: float = 3.0         # communion / belonging
    worship_esteem_relief: float = 18.0


@dataclass
class Gang:
    id: str
    name: str
    leader: str                       # agent id
    members: list[str] = field(default_factory=list)
    turf: list[str] = field(default_factory=list)  # facility names held
    founded_day: int = 0

    def size(self) -> int:
        return len(self.members)


@dataclass
class Religion:
    id: str
    name: str
    prophet: str                      # founding agent id
    members: list[str] = field(default_factory=list)
    founded_day: int = 0

    def size(self) -> int:
        return len(self.members)


# A small bank of names for colour.
GANG_NAMES = [
    "Iron Vipers", "Ash Wolves", "Red Talons", "Gravel Kings", "Night Crows",
    "Salt Brotherhood", "Broken Anvil", "Dust Serpents",
]
FAITH_NAMES = [
    "Order of the Granary", "Church of the Open Hand", "The Quiet Light",
    "Disciples of the Plaza", "The Harvest Covenant", "Way of the Builder",
]


def discontent(agent, *, oppressed: bool, fear_weight: float = 0.5) -> float:
    """A 0..100 measure of how aggrieved an agent is — the fuel for rebellion.

    Rises with fear and addiction, falls with pleasure/standing, and jumps when
    the agent is shut out of power (e.g. ruled by an oligarchy, fined, taxed).
    """
    score = fear_weight * agent.fear
    score += 0.3 * agent.addiction
    score += max(0.0, 30.0 - agent.reputation)        # the powerless chafe
    score -= 0.05 * agent.pleasure                     # the content endure
    if oppressed:
        score += 25.0
    return max(0.0, min(100.0, score))
