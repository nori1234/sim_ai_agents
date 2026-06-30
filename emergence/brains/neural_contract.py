"""The versioned contract between the engine and an external developmental brain.

This is the single source of truth that ``llm_model_agi``'s adapter
(``agent.adapters.emergence`` — ``to_engine_action`` and ``EmergenceObsTokenizer``)
should import rather than hard-code, so that when the engine's action vocabulary
or observation schema changes, the contract version bumps and the drift is caught
by ``tests/test_neural_contract.py`` instead of failing silently at integration.

Nothing here imports torch or the external package; it only reflects the engine's
own contracts (``actions.ActionType``, ``Agent.snapshot``, ``Observation``).

Bump ``CONTRACT_VERSION`` (minor for additive, major for breaking) whenever
``ACTION_VOCAB``, ``PARAM_SPEC`` or ``SELF_VIEW_KEYS`` change.
"""

from __future__ import annotations

from ..actions import ActionType

#: Semantic version of this contract. The adapter should record the version it
#: was built against; a major mismatch means the mapping must be revisited.
CONTRACT_VERSION = "1.0"

#: The engine's canonical no-op, for the out-of-vocab / unresolvable-target clamp.
#: ``_do_idle`` is literally ``pass`` (no effect, only the per-tick upkeep runs).
#: It is also ``ACTION_VOCAB[0]`` (IDLE is the first enum member). IMPORTANT: clamp
#: to THIS, not to ``rest``/``sleep`` — those are real actions that restore
#: energy/fatigue, so using them as a "do nothing" would hand the agent free
#: recovery and bias learning.
IDLE_ACTION = ActionType.IDLE.value  # == "idle"

#: The complete action vocabulary a policy may target — derived directly from the
#: engine enum so it can never silently drift from what the engine accepts. Map
#: the policy's output dimension onto these string values.
ACTION_VOCAB: tuple[str, ...] = tuple(a.value for a in ActionType)

#: The shape of ``params`` for each action (the second positional arg of
#: ``Action``). ``target`` and friends live INSIDE params — there is no separate
#: ``target`` argument. Values are human-readable type hints; the engine clamps
#: invalid/over-large params gracefully, so an approximate spec degrades, it does
#: not crash. Authoritative prose lives in ``emergence/actions.py``'s docstring.
#: ``|`` separates alternative param sets; an empty dict means "no params".
PARAM_SPEC: dict[str, dict] = {
    "idle": {},
    "move": {"facility_type": "str (a FacilityType value)", "pos": "(x, y) — alt to facility_type"},
    "gather": {},
    "sow": {},
    "eat": {},
    "rest": {},
    "sleep": {},
    "mate": {"target": "agent_id"},
    "work": {},
    "deposit_granary": {"amount": "int"},
    "draw_granary": {"amount": "int"},
    "transfer": {"target": "agent_id", "resource": "str", "amount": "int"},
    "solicit": {"target": "agent_id", "resource": "str", "amount": "int", "deceptive": "bool"},
    "propose": {"text": "str", "build": "str (optional facility to commission)"},
    "vote": {"proposal_id": "int", "support": "bool"},
    "build": {"facility_type": "str", "name": "str"},
    "collaborate": {"text": "str"},
    "speak": {"text": "str"},
    "praise": {"target": "agent_id"},
    "create": {"title": "str"},
    "steal": {"target": "agent_id"},
    "attack": {"target": "agent_id"},
    "arson": {"facility_name": "str"},
    "report_crime": {"target": "agent_id"},
    "arrest": {"target": "agent_id"},
    "craft_weapon": {},
    "deal_drug": {"target": "agent_id"},
    "take_drug": {},
    "join_gang": {},
    "rebel": {},
    "preach": {},
    "worship": {},
    # offer has three alternative param-sets (| marks each boundary):
    #   swap    : give_item, give_qty, want_item, want_qty
    #   service : service, want_item, want_qty (the fee; 0 = charity)
    #   loan    : item, principal, repay
    "offer": {"give_item": "str", "give_qty": "int", "want_item": "str", "want_qty": "int",
              "|service": "str (a service you perform, e.g. healing/feast)",
              "|item": "str (loan: the lent item)", "|principal": "int", "|repay": "int"},
    "accept": {"offer_id": "int"},
    "craft": {"item": "str (a recipe output)"},
    "lend": {"to": "agent_id", "item": "money", "qty": "int", "repay": "int", "due_in_days": "int"},
    "repay": {"loan_id": "int"},
    "deposit": {"bank": "agent_id (a banker standing at a BANK)", "amount": "int"},
    "withdraw": {"bank": "agent_id", "amount": "int"},
    "endorse": {"to": "agent_id", "bank": "agent_id", "amount": "int"},
    "take": {"from": "agent_id", "items": "dict[str, int]", "consent": "bool"},
    "give": {"to": "agent_id", "items": "dict[str, int]", "consent": "bool"},
    "use": {"item": "str", "qty": "int", "on": "agent_id (optional; default self)"},
    "strike": {"target": "agent_id", "|facility_name": "str"},
    "make": {"output": "'work' | a recipe item", "title": "str (for a work)"},
    "say": {"text": "str", "to": "agent_id (optional)"},
    "bond": {"proposal_id": "int", "support": "bool", "|with": "agent_id"},
}

#: Keys present in ``observation.self_view`` (= ``Agent.snapshot()``). The reward
#: function and the tokenizer read from these. ``energy`` (0..100 float) and
#: ``money`` (int) are the survival/material signals; ``reputation`` is the social
#: standing signal. NOTE: there is no "trust toward me" scalar — ``trust`` only
#: appears per-neighbour in ``observation.others[i]["trust"]`` (this agent's trust
#: OF them).
SELF_VIEW_KEYS: frozenset[str] = frozenset({
    "id", "name", "profession", "alive", "energy", "money", "food", "materials",
    "hunger", "fatigue", "libido", "reputation", "fear", "weapons", "addiction",
    "gang", "faith", "age_days", "crimes", "last_crime_day", "frauds",
})

#: Top-level ``Observation`` fields the adapter/tokenizer may consume. (The engine
#: may surface extra layer dicts as empty when the layer is off.)
OBSERVATION_FIELDS: frozenset[str] = frozenset({
    "day", "tick", "self_view", "position", "nearby_facilities", "here", "others",
    "open_proposals", "granary_food", "recent_events", "memory", "knowledge",
    "can_reproduce", "mating_urge", "esteem_urge", "fear_level", "actualization_pull",
    "society", "discontent", "here_roles", "nearest_roles", "environment", "role",
    "affordances", "norms", "laws", "public_works", "open_offers", "economy", "debts",
})


#: For each verb that references another agent, the ``params`` key that holds the
#: agent id (= ``observation.others[i]["id"]``). Verbs not listed take no agent
#: counterparty. The adapter should resolve the key per this map, not assume
#: "target" everywhere.
COUNTERPARTY_KEY: dict[str, str] = {
    "mate": "target", "praise": "target", "attack": "target", "steal": "target",
    "arrest": "target", "report_crime": "target", "transfer": "target",
    "solicit": "target", "deal_drug": "target",
    "take": "from", "give": "to", "lend": "to", "endorse": "to", "bond": "with",
    "strike": "target",          # strike also has a facility form — see below
}

#: Verbs that target a facility *by name* (= ``observation.nearby_facilities[i]["name"]``
#: or ``observation.here["name"]``). ``strike`` is the only verb that is agent-OR-
#: facility; ``arson`` is always a facility.
FACILITY_TARGET_KEY: dict[str, str] = {"arson": "facility_name", "strike": "facility_name"}

#: Target-resolution rules for the adapter (advisory, not version-bearing):
#:  1. Honour an explicit target already in the policy's spec before any default.
#:  2. The chosen target must actually be present in THIS observation (a nearby
#:     agent / facility); if not, clamp to idle (same policy as out-of-vocab).
#:  3. ``strike`` defaults to an AGENT, not a facility: the explicit
#:     building-destruction verb is ``arson``, so a bare ``strike`` should read as
#:     violence (the common case) and not silently become arson. Pick the nearest
#:     plausible agent; fall back to a facility only if the spec named one.
STRIKE_DEFAULT_TARGET = "agent"


def is_valid_action_value(value: str) -> bool:
    """True if ``value`` is an action the engine accepts (a defensive check the
    adapter can call before constructing an Action)."""
    return value in ACTION_VOCAB
