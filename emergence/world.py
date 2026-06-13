"""The town: a grid of tiles populated with facilities the agents can use."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FacilityType(str, Enum):
    """Kinds of buildings in the town and the affordance each provides."""

    FARM = "farm"  # gather food
    FOREST = "forest"  # gather materials (wood)
    MINE = "mine"  # gather materials (ore)
    WORKSHOP = "workshop"  # convert materials -> money (work)
    MARKET = "market"  # trade resources for money
    BANK = "bank"  # store / hold money
    TOWN_HALL = "town_hall"  # propose rules and vote
    LIBRARY = "library"  # write reports, collaborate
    POLICE_STATION = "police_station"  # report crime, deter offenders
    HOSPITAL = "hospital"  # restore energy faster
    HOUSE = "house"  # rest / restore energy
    PLAZA = "plaza"  # gather, speak publicly
    MONUMENT = "monument"  # agent-built landmark (collaboration trophy)
    GRANARY = "granary"  # shared food store
    TEMPLE = "temple"  # place of worship (society layer)


# Where each facility lets an agent gather, and what it yields.
GATHER_YIELD: dict[FacilityType, tuple[str, int]] = {
    FacilityType.FARM: ("food", 3),
    FacilityType.FOREST: ("materials", 2),
    FacilityType.MINE: ("materials", 3),
    FacilityType.GRANARY: ("food", 2),
}

# Facilities where an agent can "work" for money.
WORKPLACES = {FacilityType.WORKSHOP, FacilityType.MARKET}


@dataclass
class Facility:
    """A single building at a fixed location in the town."""

    name: str
    ftype: FacilityType
    x: int
    y: int
    # Built facilities (e.g. monuments) record who collaborated on them.
    builders: list[str] = field(default_factory=list)
    built_on_day: Optional[int] = None
    # Emergent roles a facility takes on (society layer): "weapons_factory",
    # "drug_den", "gang_turf", "temple". ``controller`` names the holding gang.
    roles: set[str] = field(default_factory=set)
    controller: Optional[str] = None  # gang id holding this as turf

    def add_role(self, role: str) -> None:
        self.roles.add(role)

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def can_gather(self) -> bool:
        return self.ftype in GATHER_YIELD

    def gather_yield(self) -> Optional[tuple[str, int]]:
        return GATHER_YIELD.get(self.ftype)

    def is_workplace(self) -> bool:
        return self.ftype in WORKPLACES


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Grid distance allowing diagonal moves (king moves)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class World:
    """The shared environment: a grid, the facilities on it, and town-wide
    state such as the public event log and shared stores."""

    def __init__(self, width: int = 24, height: int = 24):
        self.width = width
        self.height = height
        self.facilities: list[Facility] = []
        self.day: int = 1
        self.tick: int = 0
        # Shared food store any agent may draw from (a commons that can be
        # cooperated over or plundered).
        self.granary_food: int = 30
        self.events: list[dict] = []

    # -- construction ------------------------------------------------------
    def add_facility(self, facility: Facility) -> Facility:
        self.facilities.append(facility)
        return facility

    def facilities_of(self, ftype: FacilityType) -> list[Facility]:
        return [f for f in self.facilities if f.ftype == ftype]

    def facility_at(self, pos: tuple[int, int]) -> Optional[Facility]:
        for f in self.facilities:
            if f.pos == pos:
                return f
        return None

    def nearest(
        self, pos: tuple[int, int], ftype: FacilityType
    ) -> Optional[Facility]:
        candidates = self.facilities_of(ftype)
        if not candidates:
            return None
        return min(candidates, key=lambda f: chebyshev(pos, f.pos))

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def step_towards(
        self, pos: tuple[int, int], target: tuple[int, int]
    ) -> tuple[int, int]:
        """One king-move step from ``pos`` toward ``target``."""
        dx = _sign(target[0] - pos[0])
        dy = _sign(target[1] - pos[1])
        nxt = (pos[0] + dx, pos[1] + dy)
        return nxt if self.in_bounds(nxt) else pos

    # -- logging -----------------------------------------------------------
    def log(self, kind: str, **data) -> dict:
        entry = {"day": self.day, "tick": self.tick, "kind": kind, **data}
        self.events.append(entry)
        return entry


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


# Default town layout: 40+ facilities, echoing the "library, police station,
# 40 facilities" setup from the Emergence World experiment.
DEFAULT_LAYOUT: list[tuple[str, FacilityType, int, int]] = [
    ("North Farm", FacilityType.FARM, 3, 3),
    ("South Farm", FacilityType.FARM, 20, 19),
    ("East Farm", FacilityType.FARM, 21, 6),
    ("Greenwood", FacilityType.FOREST, 2, 12),
    ("Old Forest", FacilityType.FOREST, 18, 14),
    ("Iron Mine", FacilityType.MINE, 5, 21),
    ("Copper Mine", FacilityType.MINE, 22, 2),
    ("Central Workshop", FacilityType.WORKSHOP, 11, 11),
    ("North Workshop", FacilityType.WORKSHOP, 8, 4),
    ("Dockside Workshop", FacilityType.WORKSHOP, 17, 20),
    ("Grand Market", FacilityType.MARKET, 12, 8),
    ("Riverside Market", FacilityType.MARKET, 6, 16),
    ("First Bank", FacilityType.BANK, 13, 12),
    ("Town Hall", FacilityType.TOWN_HALL, 12, 12),
    ("Public Library", FacilityType.LIBRARY, 10, 13),
    ("Police Station", FacilityType.POLICE_STATION, 14, 10),
    ("City Hospital", FacilityType.HOSPITAL, 9, 9),
    ("Central Plaza", FacilityType.PLAZA, 12, 11),
    ("West Granary", FacilityType.GRANARY, 7, 11),
    ("East Granary", FacilityType.GRANARY, 16, 12),
    # Housing district.
    ("House 1", FacilityType.HOUSE, 4, 7),
    ("House 2", FacilityType.HOUSE, 5, 8),
    ("House 3", FacilityType.HOUSE, 6, 7),
    ("House 4", FacilityType.HOUSE, 7, 8),
    ("House 5", FacilityType.HOUSE, 18, 7),
    ("House 6", FacilityType.HOUSE, 19, 8),
    ("House 7", FacilityType.HOUSE, 20, 9),
    ("House 8", FacilityType.HOUSE, 17, 9),
    ("House 9", FacilityType.HOUSE, 9, 18),
    ("House 10", FacilityType.HOUSE, 10, 19),
    # A handful of extra civic buildings to clear "40+ facilities".
    ("School", FacilityType.LIBRARY, 8, 14),
    ("Archive", FacilityType.LIBRARY, 15, 14),
    ("North Plaza", FacilityType.PLAZA, 9, 5),
    ("South Plaza", FacilityType.PLAZA, 15, 18),
    ("Clinic", FacilityType.HOSPITAL, 16, 6),
    ("Watchpost", FacilityType.POLICE_STATION, 6, 19),
    ("Reserve Bank", FacilityType.BANK, 19, 13),
    ("Trade Post", FacilityType.MARKET, 21, 16),
    ("Lumber Camp", FacilityType.FOREST, 13, 22),
    ("Quarry", FacilityType.MINE, 3, 17),
    ("Annex Workshop", FacilityType.WORKSHOP, 22, 11),
    ("Town Granary", FacilityType.GRANARY, 12, 16),
]


def build_default_world() -> World:
    """A 24x24 town with the default 40+ facility layout."""
    world = World(24, 24)
    for name, ftype, x, y in DEFAULT_LAYOUT:
        world.add_facility(Facility(name=name, ftype=ftype, x=x, y=y))
    return world
