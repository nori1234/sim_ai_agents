"""Historical development: institutions appear in a plausible order, and the
town's prosperity is measured the way history suggests it compounds.

The idea (the user's): how facilities come to be, and how a society prospers,
follows patterns we can read off world / Japanese history — agriculture yields a
*surplus*, surplus enables *storage* and *specialization*, specialization enables
*trade*, trade and population call for *governance* and *law*, security and
surplus free people for *knowledge* and *culture*. So instead of a town that
starts with everything, a **founding** town starts at subsistence and must build
itself up *in order*, each institution gated on its historical prerequisites.

Opt-in (``Simulation.development`` / the founding scenario); the rich default
town is unchanged.
"""

from __future__ import annotations

from .world import Facility, FacilityType, World

# A sparse subsistence start: farms to eat, a wood/ore source, homes, a square.
# Everything else (granary, market, town hall, police, library, ...) must be
# *built* by the town as it develops.
FOUNDING_LAYOUT = [
    ("North Field", FacilityType.FARM, 5, 5),
    ("South Field", FacilityType.FARM, 18, 18),
    ("East Field", FacilityType.FARM, 19, 6),
    ("Greenwood", FacilityType.FOREST, 4, 14),
    ("Old Quarry", FacilityType.MINE, 7, 20),
    ("Common Square", FacilityType.PLAZA, 12, 12),
    ("House 1", FacilityType.HOUSE, 9, 9),
    ("House 2", FacilityType.HOUSE, 10, 10),
    ("House 3", FacilityType.HOUSE, 14, 11),
    ("House 4", FacilityType.HOUSE, 11, 14),
]

# What must already exist before a facility can plausibly be built. The chain
# encodes a historical sequence: storage after farming, trade after production,
# the state after surplus, law after the state, knowledge after security.
PREREQUISITES: dict[FacilityType, list[FacilityType]] = {
    FacilityType.GRANARY: [FacilityType.FARM],
    FacilityType.WORKSHOP: [FacilityType.MINE],
    FacilityType.MARKET: [FacilityType.WORKSHOP, FacilityType.GRANARY],
    FacilityType.BANK: [FacilityType.MARKET],
    FacilityType.TOWN_HALL: [FacilityType.GRANARY],
    FacilityType.LIBRARY: [FacilityType.TOWN_HALL],
    FacilityType.HOSPITAL: [FacilityType.TOWN_HALL],
    FacilityType.POLICE_STATION: [FacilityType.TOWN_HALL],
    FacilityType.PRISON: [FacilityType.POLICE_STATION],
    FacilityType.MONUMENT: [FacilityType.TOWN_HALL],
    FacilityType.TEMPLE: [],          # religion/culture comes early
}


def founding_world() -> World:
    """A 24x24 frontier with only subsistence facilities — a town to grow."""
    world = World(24, 24)
    for name, ftype, x, y in FOUNDING_LAYOUT:
        world.add_facility(Facility(name=name, ftype=ftype, x=x, y=y))
    return world


def _has(world: World, ftype: FacilityType) -> bool:
    return bool(world.facilities_of(ftype))


def can_build(ftype: FacilityType, world: World) -> bool:
    """True if every historical prerequisite for ``ftype`` already stands."""
    return all(_has(world, req) for req in PREREQUISITES.get(ftype, []))


def next_public_work(sim) -> str | None:
    """The institution a developing town should build next, read off its state
    in roughly historical order. Drives the heuristic council; an LLM council is
    told to reason from real history instead and may choose differently."""
    world = sim.world
    pop = max(1, sum(1 for a in sim.agents if a.alive))
    crimes = sim.metrics.crimes_total
    food = sum(a.food() for a in sim.agents if a.alive) + world.granary_food

    def missing(ft):
        return not _has(world, ft) and can_build(ft, world)

    # 1. Storage once there's farming (and especially if food is tight).
    if missing(FacilityType.GRANARY):
        return "granary"
    # 2. Specialization -> a workshop, then trade -> a market.
    if missing(FacilityType.WORKSHOP):
        return "workshop"
    if missing(FacilityType.MARKET):
        return "market"
    # 3. The state, once there's a surplus to organise.
    if missing(FacilityType.TOWN_HALL):
        return "town_hall"
    # 4. Law and order, once the state exists and disorder appears.
    if crimes >= 3 and missing(FacilityType.POLICE_STATION):
        return "police_station"
    if crimes >= 8 and _has(world, FacilityType.POLICE_STATION) \
            and not _has(world, FacilityType.PRISON):
        return "prison"
    # 5. Knowledge & care, once people are secure and numerous.
    if missing(FacilityType.LIBRARY):
        return "library"
    if pop >= 8 and missing(FacilityType.HOSPITAL):
        return "hospital"
    if missing(FacilityType.BANK):
        return "bank"
    return None


def prosperity(sim) -> float:
    """A 0–100 index combining the drivers history rewards: food security,
    order, wealth/trade, knowledge, and built infrastructure."""
    world = sim.world
    agents = [a for a in sim.agents if a.alive]
    pop = max(1, len(agents))

    def clamp(v):
        return max(0.0, min(1.0, v))

    food = (sum(a.food() for a in agents) + world.granary_food) / (pop * 4)
    order = 1.0 - clamp(sim.metrics.crimes_total / (pop * 8))
    wealth = sum(a.money for a in agents) / (pop * 40)
    libraries = len(world.facilities_of(FacilityType.LIBRARY))
    knowledge = (sim.metrics.works_created + 2 * libraries) / (pop)
    types_present = len({f.ftype for f in world.facilities})
    infra = types_present / len(FacilityType)

    score = (clamp(food) + order + clamp(wealth) + clamp(knowledge) + infra) / 5.0
    return round(score * 100, 1)
