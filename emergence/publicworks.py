"""Public works: the civic loop that lets the town build itself.

Closes a loop across the layers already present:

    a condition arises (crime, hunger, sickness)
        -> someone proposes a public work ("build a prison")
        -> the council votes (existing governance)
        -> if it passes AND the treasury can afford it
        -> a builder erects the facility, paid for by the state
        -> the new facility changes the world (a prison deters crime), which
           feeds back into what gets proposed next.

The state's money (the treasury) is filled by a small daily civic levy. Opt-in
via ``Simulation.public_works`` (default off), so the baseline town is unchanged.
"""

from __future__ import annotations

from .world import FacilityType

# What the council may commission, keyed by words an agent (or LLM) might use.
BUILDABLE: dict[str, FacilityType] = {
    "prison": FacilityType.PRISON,
    "jail": FacilityType.PRISON,
    "police": FacilityType.POLICE_STATION,
    "police_station": FacilityType.POLICE_STATION,
    "watch": FacilityType.POLICE_STATION,
    "granary": FacilityType.GRANARY,
    "hospital": FacilityType.HOSPITAL,
    "clinic": FacilityType.HOSPITAL,
    "farm": FacilityType.FARM,
    "workshop": FacilityType.WORKSHOP,
    "library": FacilityType.LIBRARY,
    "market": FacilityType.MARKET,
}

PUBLIC_WORKS_COST = 20        # what the treasury pays to erect one facility
CIVIC_LEVY_PER_AGENT = 2      # each citizen's daily contribution to the treasury

# Facilities that deter crime (so building them actually improves safety).
DETERRENT_FACILITIES = {FacilityType.POLICE_STATION, FacilityType.PRISON}


def parse_build(text: str) -> FacilityType | None:
    """Map free proposal text (or an explicit keyword) to a buildable facility."""
    t = (text or "").lower()
    for keyword, ftype in BUILDABLE.items():
        if keyword in t:
            return ftype
    return None


def proposed_work_for_conditions(*, recent_crimes: int, food_scarce: bool,
                                 sick: bool) -> str | None:
    """A simple condition -> public-work mapping the heuristic council uses.

    (An LLM council isn't limited to this — it can propose whatever it judges
    the town needs.)
    """
    if recent_crimes >= 6:
        return "prison"
    if recent_crimes >= 3:
        return "police_station"
    if food_scarce:
        return "granary"
    if sick:
        return "hospital"
    return None
