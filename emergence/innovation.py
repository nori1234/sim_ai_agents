"""Emergent innovation & productivity growth (技術革新).

Before this, "technology" was a fixed historical script: ``development.py``
unlocks facilities in a scripted prerequisite order, but productivity itself
never changed — a workshop's ``tools`` recipe, a farm's yield, gather rates
were constants forever. There was no invention, no learning-by-doing, no
diffusion of technique.

This lets productivity rise from what agents *do*, not a timeline:

* **Skill / learning-by-doing** — a single per-agent scalar (``Agent.skill``,
  0..``skill_cap``) that rises a little with each gather/craft and scales its
  yield up — human capital that compounds with practice, not a scripted
  unlock.
* **Invention as discovery** — an experienced crafter occasionally discovers
  a better recipe, drawn from a small, bounded, hand-authored pool per good
  (never code-generated — the same stance as law-as-norm, #37). A discovery
  replaces the simulation's own working copy of the recipe (``Simulation.
  recipes``), so it diffuses to every crafter from then on, and is written to
  the town library (where one exists) so it persists as a record — and can be
  lost the way any book can (#22's rot).

Opt-in via :data:`InnovationConfig.enabled` (default ``False``); ``agent.skill``
sits at 0 and nothing here fires when off, so the four-society baseline is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InnovationConfig:
    enabled: bool = False

    skill_gain_per_use: float = 0.02   # skill gained per successful gather/craft
    skill_cap: float = 1.0             # skill saturates here (fully experienced)
    skill_yield_bonus: float = 0.6     # max yield multiplier bonus at full skill
    discovery_chance: float = 0.03     # per craft at/above discovery_skill_min
    discovery_skill_min: float = 0.6   # skill needed before discovery can occur


# A bounded, predefined pool of better recipes a skilled crafter may discover
# for a good -- drawn from, never generated. output good -> ordered list of
# (inputs, required_facility) variants, each an improvement worth adopting.
DISCOVERY_POOL: dict[str, list[tuple[dict, "str | None"]]] = {
    "tools": [
        ({"materials": 1}, "workshop"),   # a technique that halves the material cost
    ],
}


def skill_yield_mult(skill: float, cfg: InnovationConfig) -> float:
    """1.0 with no skill, rising toward ``1 + skill_yield_bonus`` at full skill."""
    if not cfg.enabled:
        return 1.0
    return 1.0 + cfg.skill_yield_bonus * max(0.0, min(skill, cfg.skill_cap))
