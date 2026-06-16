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


def idle(rationale: str = "") -> Action:
    return Action(ActionType.IDLE, rationale=rationale)
