"""Affordances & roles: the *possibility space*, defined as data.

Design stance — separate "what is possible" from "what to do":

* The **engine** declares, as plain data, what each facility lets an agent do
  (a town hall affords proposing & voting; a hospital, healing & rest) and what
  each profession's role is. Extending the world is then editing a dict, not
  rewiring behaviour.
* The **agent** (especially an LLM brain) is shown its role and the actions
  available where it stands, and chooses freely. We don't script "the guard
  patrols"; we tell the guard it can keep the peace and let it decide.

The heuristic brain ignores all of this (so offline runs are unchanged); it is
surfaced through the Observation for the LLM brain to act on.
"""

from __future__ import annotations

from .actions import ActionType
from .world import FacilityType

# What each facility *affords* — the actions that specifically make sense there.
# (General actions like move/eat/rest are always available and aren't listed.)
FACILITY_AFFORDANCES: dict[FacilityType, list[ActionType]] = {
    FacilityType.FARM: [ActionType.GATHER],
    FacilityType.FOREST: [ActionType.GATHER],
    FacilityType.MINE: [ActionType.GATHER],
    FacilityType.GRANARY: [ActionType.GATHER, ActionType.DEPOSIT_GRANARY,
                           ActionType.DRAW_GRANARY],
    FacilityType.WORKSHOP: [ActionType.WORK, ActionType.CRAFT_WEAPON,
                            ActionType.CREATE],
    FacilityType.MARKET: [ActionType.WORK],
    FacilityType.TOWN_HALL: [ActionType.PROPOSE, ActionType.VOTE],
    FacilityType.LIBRARY: [ActionType.COLLABORATE, ActionType.CREATE],
    FacilityType.POLICE_STATION: [ActionType.REPORT_CRIME, ActionType.ARREST],
    FacilityType.HOSPITAL: [ActionType.REST, ActionType.TREAT],
    FacilityType.HOUSE: [ActionType.REST, ActionType.SLEEP],
    FacilityType.PLAZA: [ActionType.SPEAK, ActionType.BUILD, ActionType.CREATE],
    FacilityType.TEMPLE: [ActionType.WORSHIP, ActionType.PREACH],
    FacilityType.PRISON: [ActionType.REPORT_CRIME, ActionType.ARREST],
}

# A one-line role per profession — soft guidance handed to the agent, not a rule.
PROFESSION_ROLES: dict[str, str] = {
    "farmer": "tend the farms and keep the town fed",
    "builder": "raise buildings and monuments at the plaza",
    "teacher": "write and share knowledge at the library",
    "merchant": "trade and earn at the market",
    "doctor": "care for the weak and injured at the hospital",
    "guard": "keep the peace — deter crime and report offenders to the police",
    "miner": "gather ore and materials from the mines",
    "librarian": "co-author the town's records at the library",
    "smith": "forge tools and goods at the workshop",
    "council clerk": "run the town's governance — propose and vote at the town hall",
    "child": "learn from your family and find your place",
}


def affordances_at(facility) -> list[str]:
    """Action names specifically afforded by the facility an agent stands on."""
    if facility is None:
        return []
    return [a.value for a in FACILITY_AFFORDANCES.get(facility.ftype, [])]


def role_of(profession: str) -> str:
    return PROFESSION_ROLES.get(profession, "make your own way in the town")


# Production specialisation (economy layer). Each profession produces ONE good
# well and is inefficient at off-specialty self-supply — so self-sufficiency
# wastes turns and buying from a specialist is the sensible path. This is what
# gives food/materials a *structural* demand (and a producer its buyers).
# Survival is not locked: off-specialty gathering is a low-yield fallback (never
# zero), not an inability.
GATHER_SPECIALTY: dict[str, str] = {
    "farmer": "food",
    "miner": "materials",
}
_SPECIALIST_BONUS = 2.0
_OFF_SPECIALTY = 0.5
_SPECIALISED_RESOURCES = {"food", "materials"}


def gather_specialty(profession: str) -> str | None:
    """The resource a profession produces well (or None for a generalist)."""
    return GATHER_SPECIALTY.get(profession)


def gather_multiplier(profession: str, resource: str) -> float:
    """How well a profession gathers a resource (economy layer only). Specialists
    are productive; off-specialty self-supply of a specialised good is inefficient
    so trade has a reason. Unspecialised goods stay at 1.0."""
    if GATHER_SPECIALTY.get(profession) == resource:
        return _SPECIALIST_BONUS
    if resource in _SPECIALISED_RESOURCES:
        return _OFF_SPECIALTY
    return 1.0
