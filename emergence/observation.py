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


def _facility_view(f: Facility, dist: int) -> dict:
    return {"name": f.name, "type": f.ftype.value, "distance": dist}


def _proposal_view(p: Proposal, voter_id: str) -> dict:
    return {
        "id": p.id,
        "author": p.author,
        "text": p.text,
        "yes": p.yes(),
        "no": p.no(),
        "already_voted": voter_id in p.votes,
    }
