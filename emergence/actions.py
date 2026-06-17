"""The vocabulary of things an agent can intend to do on its turn.

A brain returns one :class:`Action` per turn. The simulation validates and
applies it; invalid actions degrade gracefully (e.g. moving toward a target
you cannot reach this tick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    IDLE = "idle"
    MOVE = "move"  # toward a facility type or coordinate
    GATHER = "gather"  # collect food/materials at current facility
    EAT = "eat"  # consume food -> energy, relieve hunger
    REST = "rest"  # restore a little energy (more at house/hospital)
    SLEEP = "sleep"  # relieve fatigue substantially (best at a house)
    MATE = "mate"  # reproduce with an adjacent, trusted, well-rested partner
    WORK = "work"  # earn money at a workplace
    DEPOSIT_GRANARY = "deposit_granary"  # give food to the commons
    DRAW_GRANARY = "draw_granary"  # take food from the commons
    TRANSFER = "transfer"  # give resources/money to another agent
    SOLICIT = "solicit"  # ask another agent for resources (may be a scam)
    PROPOSE = "propose"  # put a rule to the town
    VOTE = "vote"  # vote on an open proposal
    BUILD = "build"  # contribute to constructing a facility
    COLLABORATE = "collaborate"  # co-author a report / shared project
    SPEAK = "speak"  # public statement (logged)
    PRAISE = "praise"  # publicly commend another agent (grants them esteem)
    CREATE = "create"  # produce a work of art/craft/scholarship (fulfillment)
    STEAL = "steal"  # take resources from another agent (crime)
    ATTACK = "attack"  # violence against another agent (crime)
    ARSON = "arson"  # destroy/damage a facility (crime)
    REPORT_CRIME = "report_crime"  # flag an offender at a police station
    ARREST = "arrest"  # detain a nearby recent offender (enforcement as an act)

    # -- society layer (weapons, drugs, gangs, religion) ----------------
    CRAFT_WEAPON = "craft_weapon"  # forge a weapon at a workshop
    DEAL_DRUG = "deal_drug"  # produce and sell a narcotic to another agent
    TAKE_DRUG = "take_drug"  # consume a dose (escape; builds addiction)
    JOIN_GANG = "join_gang"  # form or join a gang
    REBEL = "rebel"  # armed uprising against those in power
    PREACH = "preach"  # found or spread a religion (convert nearby agents)
    WORSHIP = "worship"  # pray at a temple (eases fear, grants belonging)

    # -- economic physics (primitives, not institutions) ----------------
    OFFER = "offer"  # post a swap: give N of A for M of B (any tradable goods)
    ACCEPT = "accept"  # agree to an open offer; the swap executes atomically
    CRAFT = "craft"  # transform inputs into an output per a recipe
    LEND = "lend"  # extend credit: give a principal now against a promised repay
    REPAY = "repay"  # settle a loan you owe (builds trust; defaulting destroys it)


    # -- physical primitives (the instruction set the macros lower to) --
    # Institutions are read off these acts + context, not baked into the verb:
    # a consent-less take from an agent IS theft; a consensual give IS a gift.
    TAKE = "take"  # pull items into own inventory (from an agent or the world)
    GIVE = "give"  # push items out of own inventory (to an agent or the world)
    USE = "use"    # consume/apply a held item to self or a target (eat, dose)
    STRIKE = "strike"  # apply force to damage an agent or a structure
    MAKE = "make"  # transform inputs / effort into an output (a work, a good)
    SAY = "say"    # broadcast a signal/message (optionally at a target)
    BOND = "bond"  # commit to an agreement or allegiance (a vote, a pact)


# Actions the world treats as crimes for metric purposes.
CRIME_ACTIONS = {ActionType.STEAL, ActionType.ATTACK, ActionType.ARSON}


@dataclass
class Action:
    """A single intended action plus its parameters.

    Conventions for ``params``:
      MOVE        -> {"facility_type": str} or {"pos": (x, y)}
      TRANSFER    -> {"target": agent_id, "resource": str, "amount": int}
      SOLICIT     -> {"target": agent_id, "resource": str, "amount": int,
                      "deceptive": bool}
      PROPOSE     -> {"text": str}
      VOTE        -> {"proposal_id": int, "support": bool}
      BUILD       -> {"facility_type": str, "name": str}
      STEAL/ATTACK-> {"target": agent_id}
      MATE        -> {"target": agent_id}
      ARSON       -> {"facility_name": str}
      REPORT_CRIME-> {"target": agent_id}
      ARREST      -> {"target": agent_id}
      TAKE        -> {"from": agent_id, "items": {res: qty}, "consent": bool}
      GIVE        -> {"to": agent_id, "items": {res: qty}, "consent": bool}
      USE         -> {"item": str, "qty": int, "on": agent_id (optional, default self)}
      STRIKE      -> {"target": agent_id} or {"facility_name": str}
      MAKE        -> {"output": "work"|recipe_item, "title": str (for a work)}
      SAY         -> {"text": str, "to": agent_id (optional)}
      BOND        -> {"proposal_id": int, "support": bool} or {"with": agent_id}
      PRAISE      -> {"target": agent_id}
      CREATE      -> {"title": str}
      DEAL_DRUG   -> {"target": agent_id}
      PREACH      -> {} (founds or spreads the agent's faith to those nearby)
      SPEAK/COLLAB-> {"text": str}
    """

    type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    # Free-text rationale (the brain's "thought"); handy for LLM traces.
    rationale: str = ""

    def is_crime(self) -> bool:
        return self.type in CRIME_ACTIONS

    def __str__(self) -> str:
        if self.params:
            kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.type.value}({kv})"
        return self.type.value


@dataclass
class Event:
    """A structured record of what *physically* happened when a primitive ran.

    The interpretation layer reads the event plus context to decide what it
    *means* (theft, gift, trade), so meaning is derived from the act rather
    than hard-coded into a verb. ``other`` is the counterparty (or None),
    ``items`` is what actually moved after clamping, ``consent`` is whether the
    counterparty agreed (True/False/None when not applicable)."""

    kind: str
    actor: Any                       # Agent (kept loose to avoid an import cycle)
    other: Optional[Any] = None      # counterparty agent, if any
    items: dict[str, int] = field(default_factory=dict)
    consent: Optional[bool] = None
    site: Optional[Any] = None        # a struck/used structure (Facility), if any
    intent: Optional[str] = None      # say/bond sub-kind (praise, accusation, ...)
    payload: Optional[dict] = None    # intent-specific data (e.g. a proposal's text/build)


def idle(rationale: str = "") -> Action:
    return Action(ActionType.IDLE, rationale=rationale)
