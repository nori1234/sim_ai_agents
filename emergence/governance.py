"""Governance forms, active laws, and the policy engine.

Beyond the basic Legislature (propose + vote), this module adds:

* :class:`GovernanceConfig` — presets for four governance forms that change
  who can vote, how many votes pass a bill, and whether law enforcement exists.
* :class:`Law` — a passed bill parsed for keyword effects; active laws
  mechanically alter the simulation each tick (deterrence, tax, redistribution,
  punishment).
* :class:`Mayor` — the agent who has cast the most votes is elected every
  ``config.election_interval`` days; the mayor gets easier-to-pass proposals.
* :class:`PolicyEngine` — the runtime that owns the law list and exposes
  helpers the simulation queries per tick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ======================================================================
# Governance presets
# ======================================================================

class GovernanceForm(str, Enum):
    DIRECT = "direct"            # everyone votes equally (default)
    OLIGARCHY = "oligarchy"      # only the N wealthiest agents may vote/propose
    CONSTITUTIONAL = "constitutional"  # normal bills pass by majority; "rights"
                                       # bills need a supermajority (2/3)
    ANARCHY = "anarchy"          # no legislature at all; laws have no force


@dataclass(frozen=True)
class GovernanceConfig:
    form: GovernanceForm = GovernanceForm.DIRECT
    quorum: int = 4              # minimum votes before a bill closes
    supermajority: float = 0.67  # fraction needed for constitutional bills
    oligarch_count: int = 3      # how many top-wealth agents vote in oligarchy
    election_interval: int = 5   # days between mayoral elections
    police_range: int = 7        # tiles within which a station enables the punishment fine
    fine_amount: int = 8         # money taken when punishment law triggers
    tax_rate: float = 0.10       # fraction of money taken from richest each day


# Named presets the CLI accepts. (Crime deterrence is no longer a config knob:
# order emerges from a guard's ARREST act and agents complying with published
# norms — see docs/PRINCIPLED_MIGRATION.md.)
GOVERNANCE_PRESETS: dict[str, GovernanceConfig] = {
    "direct": GovernanceConfig(form=GovernanceForm.DIRECT),
    "oligarchy": GovernanceConfig(
        form=GovernanceForm.OLIGARCHY,
        oligarch_count=3,
    ),
    "constitutional": GovernanceConfig(
        form=GovernanceForm.CONSTITUTIONAL,
        supermajority=0.67,
    ),
    "anarchy": GovernanceConfig(
        form=GovernanceForm.ANARCHY,
    ),
}


# ======================================================================
# Law effects parsed from proposal text
# ======================================================================

class LawEffect(str, Enum):
    CRIME_DETERRENCE = "crime_deterrence"   # strengthen police effect
    FOOD_REDISTRIBUTION = "food_redistribution"  # daily commons top-up from rich
    TAX = "tax"                             # daily wealth tax → redistribution
    PUNISHMENT = "punishment"               # fine offenders reported to police
    # Per-offence norms (#35): a bill can target a *specific* crime kind
    # instead of collapsing into one blanket "crime" bucket. Parsed
    # additively alongside CRIME_DETERRENCE (a bill mentioning "theft" still
    # sets CRIME_DETERRENCE too, so has_crime_norm()'s existing meaning and
    # every existing caller are untouched) -- purely more data, not a
    # replacement.
    VIOLENCE_NORM = "violence_norm"
    THEFT_NORM = "theft_norm"
    ARSON_NORM = "arson_norm"


# Simple keyword patterns → effect mappings.
_EFFECT_PATTERNS: list[tuple[re.Pattern, LawEffect]] = [
    (re.compile(r"ban|prohibit|illegal|crime|theft|violence|offend", re.I),
     LawEffect.CRIME_DETERRENCE),
    (re.compile(r"food|granary|quota|ration|hunger|starv", re.I),
     LawEffect.FOOD_REDISTRIBUTION),
    (re.compile(r"tax|levy|revenue|fund", re.I),
     LawEffect.TAX),
    (re.compile(r"punish|fine|penalt|jail|imprison|sentence", re.I),
     LawEffect.PUNISHMENT),
    (re.compile(r"violence|assault|attack|murder|hurt|harm", re.I),
     LawEffect.VIOLENCE_NORM),
    (re.compile(r"theft|steal|stealing|robbery|rob|thief", re.I),
     LawEffect.THEFT_NORM),
    (re.compile(r"arson|burn|burning|fire-sett|torch", re.I),
     LawEffect.ARSON_NORM),
]

#: Offence kind -> the LawEffect that targets it specifically, for callers
#: that want to key a norm by which crime it's actually about.
OFFENCE_EFFECTS: dict[str, LawEffect] = {
    "violence": LawEffect.VIOLENCE_NORM,
    "theft": LawEffect.THEFT_NORM,
    "arson": LawEffect.ARSON_NORM,
}


@dataclass
class Law:
    proposal_id: int
    text: str
    enacted_day: int
    effects: list[LawEffect] = field(default_factory=list)

    @classmethod
    def from_proposal_text(cls, proposal_id: int, text: str, day: int) -> "Law":
        effects = [effect for pat, effect in _EFFECT_PATTERNS if pat.search(text)]
        # Deduplicate while preserving order.
        seen: set[LawEffect] = set()
        unique: list[LawEffect] = []
        for e in effects:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return cls(proposal_id=proposal_id, text=text, enacted_day=day, effects=unique)

    def has(self, effect: LawEffect) -> bool:
        return effect in self.effects


# ======================================================================
# Mayor
# ======================================================================

@dataclass
class Mayor:
    agent_id: str
    elected_day: int
    term_ends_day: int


# ======================================================================
# Legislature (extended)
# ======================================================================

class ProposalStatus(str, Enum):
    OPEN = "open"
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass
class Proposal:
    id: int
    author: str
    text: str
    day: int
    status: ProposalStatus = ProposalStatus.OPEN
    votes: dict[str, bool] = field(default_factory=dict)
    # Proposals tagged "constitutional" need supermajority to pass.
    constitutional: bool = False
    # A public-works proposal names the facility type to build when it passes.
    build: str | None = None

    def yes(self) -> int:
        return sum(1 for s in self.votes.values() if s)

    def no(self) -> int:
        return sum(1 for s in self.votes.values() if not s)


_CONSTITUTIONAL_RE = re.compile(r"right|right[s]?|freedom|liberty|constitution", re.I)


class Legislature:
    def __init__(self, config: GovernanceConfig | None = None):
        self.config = config or GovernanceConfig()
        self.proposals: list[Proposal] = []
        self._next_id = 1

    def propose(self, author: str, text: str, day: int,
                eligible_ids: set[str] | None = None,
                build: str | None = None) -> Optional[Proposal]:
        if self.config.form is GovernanceForm.ANARCHY:
            return None
        if eligible_ids is not None and author not in eligible_ids:
            return None
        constitutional = bool(_CONSTITUTIONAL_RE.search(text))
        p = Proposal(id=self._next_id, author=author, text=text, day=day,
                     constitutional=constitutional, build=build)
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

    def cast_vote(self, proposal_id: int, agent_id: str, support: bool,
                  eligible_ids: set[str] | None = None) -> bool:
        if self.config.form is GovernanceForm.ANARCHY:
            return False
        if eligible_ids is not None and agent_id not in eligible_ids:
            return False
        p = self.get(proposal_id)
        if p is None or p.status is not ProposalStatus.OPEN:
            return False
        p.votes[agent_id] = support
        return True

    def resolve_ready(self, electorate_size: int) -> list[Proposal]:
        resolved: list[Proposal] = []
        needed = max(self.config.quorum, electorate_size // 2 + 1)
        for p in self.open_proposals():
            if len(p.votes) < min(needed, max(1, electorate_size)):
                continue
            total = len(p.votes)
            yes_frac = p.yes() / total if total else 0
            if p.constitutional and self.config.form is GovernanceForm.CONSTITUTIONAL:
                passed = yes_frac >= self.config.supermajority
            else:
                passed = p.yes() > p.no()
            p.status = ProposalStatus.PASSED if passed else ProposalStatus.REJECTED
            resolved.append(p)
        return resolved

    def counts(self) -> tuple[int, int, int]:
        passed = sum(1 for p in self.proposals if p.status is ProposalStatus.PASSED)
        rejected = sum(1 for p in self.proposals if p.status is ProposalStatus.REJECTED)
        return (len(self.proposals), passed, rejected)


# ======================================================================
# Policy Engine — owns active laws; queried by the simulation each tick
# ======================================================================

class PolicyEngine:
    """Tracks active laws and exposes the mechanical effects they produce."""

    def __init__(self, config: GovernanceConfig | None = None):
        self.config = config or GovernanceConfig()
        self.laws: list[Law] = []

    def enact(self, proposal_id: int, text: str, day: int) -> Law:
        law = Law.from_proposal_text(proposal_id, text, day)
        self.laws.append(law)
        return law

    # -- effect queries (called by Simulation per tick/day) -------------

    def has_crime_norm(self) -> bool:
        """Whether the town has published a norm against crime.

        A norm is not a mechanical multiplier — it does nothing on its own.
        It is an *expectation*; agents weigh whether to comply with it given
        their disposition and how credibly it is enforced (see the simulation's
        enforcement expectation and the brain's compliance check)."""
        return any(l.has(LawEffect.CRIME_DETERRENCE) for l in self.laws)

    def has_punishment_law(self) -> bool:
        return any(l.has(LawEffect.PUNISHMENT) for l in self.laws)

    def has_food_redistribution(self) -> bool:
        return any(l.has(LawEffect.FOOD_REDISTRIBUTION) for l in self.laws)

    def has_tax(self) -> bool:
        return any(l.has(LawEffect.TAX) for l in self.laws)

    def offence_norm(self, kind: str) -> bool:
        """Whether the town has published a norm against this *specific*
        offence (#35) -- distinct from ``has_crime_norm()``'s one blanket
        bucket. ``kind`` is one of ``OFFENCE_EFFECTS``'s keys
        ("violence"/"theft"/"arson"); an unknown kind is never targeted."""
        effect = OFFENCE_EFFECTS.get(kind)
        if effect is None:
            return False
        return any(l.has(effect) for l in self.laws)

    def active_effects(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for l in self.laws:
            for e in l.effects:
                if e.value not in seen:
                    seen.add(e.value)
                    out.append(e.value)
        return out

    def summary(self) -> dict:
        return {
            "form": self.config.form.value,
            "laws_enacted": len(self.laws),
            "active_effects": self.active_effects(),
        }
