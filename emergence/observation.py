"""What an agent perceives on its turn — the input handed to a brain."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent
from .governance import Proposal
from .world import Facility


@dataclass
class Observation:
    """A snapshot of the world from one agent's point of view."""

    day: int
    tick: int
    self_view: dict  # the acting agent's own snapshot (energy, money, ...)
    position: tuple[int, int]
    nearby_facilities: list[dict]  # {name, type, distance}
    here: dict | None  # facility at the agent's tile, if any
    others: list[dict]  # snapshots of other living agents + distance + trust
    open_proposals: list[dict]  # {id, author, text, yes, no, already_voted}
    granary_food: int
    recent_events: list[str]  # short public log tail

    # Memory of the acting agent (its private recollections).
    memory: list[str] = field(default_factory=list)

    # Town library: lessons recorded by predecessors, surfaced when the agent is
    # by a library. Horizontal/cultural inheritance — the heuristic brain ignores
    # it (offline outcomes unchanged); an LLM brain can act on it.
    knowledge: list[str] = field(default_factory=list)

    # Drives layer: whether the body is capable of reproducing right now, and
    # the instinctual 0..1 strength of the urge to seek a mate.
    can_reproduce: bool = False
    mating_urge: float = 0.0

    # Esteem layer: 0..1 strength of the urge to seek recognition.
    esteem_urge: float = 0.0

    # Psyche layer: 0..1 grip of fear, and 0..1 pull toward creation (present
    # only when every lower need is satisfied).
    fear_level: float = 0.0
    actualization_pull: float = 0.0

    # Society layer: which sub-systems are active, this agent's grievance, and
    # the emergent roles of the tile it stands on.
    society: dict = field(default_factory=dict)
    discontent: float = 0.0
    here_roles: list = field(default_factory=list)
    nearest_roles: dict = field(default_factory=dict)  # role -> (x, y) of nearest

    # Environment layer: season, weather, market prices, active disaster.
    environment: dict = field(default_factory=dict)

    # Affordances/roles: what this agent's job is, and what the tile it stands on
    # specifically lets it do. The possibility space; the brain chooses freely.
    role: str = ""
    affordances: list = field(default_factory=list)

    # Norms: published expectations the town has enacted (e.g. a law against
    # crime) plus how credibly they are enforced. A norm is not a mechanical
    # force — the agent weighs whether to comply with it.
    #   {"crime": bool, "enforcement": float 0..1}
    norms: dict = field(default_factory=dict)

    # Every law the town has actually enacted, published as text so an LLM agent
    # can read, comply with, or *enforce* even legislation the engine has no
    # built-in mechanism for. A law's force is emergent agent behaviour, not
    # engine code — so a novel law works iff citizens choose to honour/enforce it.
    # The heuristic brain ignores this (offline outcomes unchanged).
    #   [{"text": str, "effects": [str], "day": int}]
    laws: list = field(default_factory=list)

    # Public-works loop: whether it's active and the state treasury (for proposing
    # council-funded construction).
    public_works: dict = field(default_factory=dict)

    # Economy: open swap offers the agent could accept, emergent prices, and the
    # loans this agent currently owes (its debts).
    open_offers: list = field(default_factory=list)
    economy: dict = field(default_factory=dict)
    debts: list = field(default_factory=list)


def _facility_view(f: Facility, dist: int) -> dict:
    v = {"name": f.name, "type": f.ftype.value, "distance": dist}
    if f.owner is not None:        # whose property this is (None = commons); set only under --economy
        v["owner"] = f.owner
    return v


def _proposal_view(p: Proposal, voter_id: str) -> dict:
    return {
        "id": p.id,
        "author": p.author,
        "text": p.text,
        "yes": p.yes(),
        "no": p.no(),
        "already_voted": voter_id in p.votes,
    }
