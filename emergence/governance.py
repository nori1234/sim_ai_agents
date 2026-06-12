"""Lightweight direct democracy: agents propose rules and vote on them."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProposalStatus(str, Enum):
    OPEN = "open"
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass
class Proposal:
    id: int
    author: str  # agent id
    text: str
    day: int
    status: ProposalStatus = ProposalStatus.OPEN
    votes: dict[str, bool] = field(default_factory=dict)  # agent_id -> support

    def yes(self) -> int:
        return sum(1 for s in self.votes.values() if s)

    def no(self) -> int:
        return sum(1 for s in self.votes.values() if not s)


class Legislature:
    """Holds open proposals and resolves them once enough votes are in."""

    def __init__(self, quorum: int = 4):
        self.quorum = quorum
        self.proposals: list[Proposal] = []
        self._next_id = 1

    def propose(self, author: str, text: str, day: int) -> Proposal:
        p = Proposal(id=self._next_id, author=author, text=text, day=day)
        self._next_id += 1
        self.proposals.append(p)
        return p

    def open_proposals(self) -> list[Proposal]:
        return [p for p in self.proposals if p.status is ProposalStatus.OPEN]

    def get(self, proposal_id: int) -> Proposal | None:
        for p in self.proposals:
            if p.id == proposal_id:
                return p
        return None

    def cast_vote(self, proposal_id: int, agent_id: str, support: bool) -> bool:
        p = self.get(proposal_id)
        if p is None or p.status is not ProposalStatus.OPEN:
            return False
        p.votes[agent_id] = support
        return True

    def resolve_ready(self, electorate_size: int) -> list[Proposal]:
        """Close any open proposal that has reached quorum; return those
        whose status changed this call."""
        resolved: list[Proposal] = []
        needed = max(self.quorum, electorate_size // 2 + 1)
        for p in self.open_proposals():
            if len(p.votes) >= min(needed, max(1, electorate_size)):
                p.status = (
                    ProposalStatus.PASSED if p.yes() > p.no()
                    else ProposalStatus.REJECTED
                )
                resolved.append(p)
        return resolved

    # -- metrics -----------------------------------------------------------
    def counts(self) -> tuple[int, int, int]:
        passed = sum(1 for p in self.proposals if p.status is ProposalStatus.PASSED)
        rejected = sum(1 for p in self.proposals if p.status is ProposalStatus.REJECTED)
        return (len(self.proposals), passed, rejected)
