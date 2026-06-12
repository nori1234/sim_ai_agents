"""An offline, deterministic decision policy driven by persona knobs.

This brain needs no network or API key, so the whole simulation runs anywhere
Python does. It is intentionally simple: a needs-first survival layer followed
by a persona-weighted choice among social behaviours. Tuned personas reproduce
the qualitative societies from the Emergence World experiment.
"""

from __future__ import annotations

import random

from ..actions import Action, ActionType
from ..agent import Agent, MAX_ENERGY
from ..observation import Observation
from ..personas import Persona, get_persona
from .base import AgentBrain

# Facility types an agent can gather food / materials from.
FOOD_SOURCES = {"farm", "granary"}
MATERIAL_SOURCES = {"forest", "mine"}
WORKPLACES = {"workshop", "market"}

LOW_ENERGY = 45.0
CRITICAL_ENERGY = 22.0


class HeuristicBrain(AgentBrain):
    name = "heuristic"

    def __init__(self, persona: Persona | str, rng: random.Random | None = None):
        self.persona = get_persona(persona) if isinstance(persona, str) else persona
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------
    def decide(self, agent: Agent, obs: Observation) -> Action:
        p = self.persona

        # 0. Primal drives (only non-trivial when the drives layer is enabled,
        #    otherwise hunger/fatigue stay 0 and these checks are no-ops).
        drive = self._drive_action(agent, obs)
        if drive is not None:
            return drive

        # 1. Survival comes first — but a low-diligence agent often skips it,
        #    which is how idealists talk themselves into starvation.
        survival = self._survival_action(agent, obs)
        if survival is not None:
            critical = agent.energy <= CRITICAL_ENERGY
            attend = critical or self.rng.random() < p.diligence
            if attend:
                return survival

        # 2. Retaliation against whoever wronged us (vengeful personas).
        foe = self._nearby_foe(obs)
        if foe is not None and self.rng.random() < p.vengefulness:
            return self._aggress(agent, foe, p, reason="retaliation")

        # 3. Unprovoked aggression (predators, chaotic philosophers).
        if self.rng.random() < p.aggression * 0.6:
            victim = self._nearby_target(obs)
            if victim is not None:
                return self._aggress(agent, victim, p, reason="aggression")
            # No one in reach to harm — vandalise a facility instead.
            if obs.here is not None and self.rng.random() < p.aggression * 0.4:
                return Action(
                    ActionType.ARSON,
                    {"facility_name": obs.here["name"]},
                    rationale="destructive impulse",
                )

        # 4. Building & collaboration (the monument / co-authored report).
        #    Checked before governance so civic-minded agents don't spend every
        #    free turn voting and never get around to building anything.
        build = self._build_action(agent, obs, p)
        if build is not None:
            return build

        # 5. Governance: vote on anything still open, occasionally propose.
        gov = self._governance_action(agent, obs, p)
        if gov is not None:
            return gov

        # 6. Prosocial economy: share, stock the commons — or run a scam.
        econ = self._economy_action(agent, obs, p)
        if econ is not None:
            return econ

        # 7. Talk a lot? Make a speech.
        if self.rng.random() < p.talkativeness * 0.5:
            return Action(
                ActionType.SPEAK,
                {"text": self._speech(p)},
                rationale="public statement",
            )

        # 8. Otherwise do something productive: earn or stockpile.
        return self._default_productive(agent, obs)

    # -- drives (hunger / sleep / reproduction) -------------------------
    def _drive_action(self, agent: Agent, obs: Observation) -> Action | None:
        # Hunger: if hungry, eat what we have or go find food (act before the
        # penalty threshold so diligent agents stay ahead of starvation).
        if agent.hunger >= 55:
            if agent.food() > 0:
                return Action(ActionType.EAT, rationale="hunger pangs")
            if obs.here and obs.here["type"] in FOOD_SOURCES:
                if obs.here["type"] == "granary" and obs.granary_food > 0:
                    return Action(ActionType.DRAW_GRANARY, rationale="hungry")
                return Action(ActionType.GATHER, rationale="harvest food")
            return Action(ActionType.MOVE,
                          {"facility_type": self._nearest_food_type(obs)},
                          rationale="seek food (hungry)")

        # Sleep: if tired, sleep — ideally under a roof.
        if agent.fatigue >= 62:
            here = obs.here["type"] if obs.here else None
            if here in {"house", "hospital"}:
                return Action(ActionType.SLEEP, rationale="exhausted")
            # Diligent agents seek shelter; otherwise just sleep where they are.
            if self.rng.random() < 0.6:
                return Action(ActionType.MOVE, {"facility_type": "house"},
                              rationale="find a bed")
            return Action(ActionType.SLEEP, rationale="sleep rough")

        # Reproduction is instinctual: the stronger the built-up urge, the more
        # likely the agent drops everything to court — 本能と気持ちよさ. It is the
        # urge that decides, not a rational appraisal of conditions.
        if obs.can_reproduce and self.rng.random() < obs.mating_urge:
            mate = self._find_mate(agent, obs)
            if mate is not None:
                # The MATE handler walks toward a distant partner, so we are
                # willing to chase someone across the map.
                return Action(ActionType.MATE, {"target": mate["id"]},
                              rationale="driven to court a partner")
        return None

    def _find_mate(self, agent: Agent, obs: Observation) -> dict | None:
        candidates = [
            o for o in obs.others
            if o.get("trust", 0.0) >= 0.2
            and o.get("hunger", 0) <= 85
            and o.get("fatigue", 0) <= 88
            and o.get("age_days", 0) >= 2
        ]
        if not candidates:
            return None
        # Prefer the closest, most-trusted (most affectionate) partner.
        return min(candidates, key=lambda o: (o["distance"], -o.get("trust", 0.0)))

    # -- survival -------------------------------------------------------
    def _survival_action(self, agent: Agent, obs: Observation) -> Action | None:
        if agent.energy > LOW_ENERGY and agent.food() >= 2:
            return None
        if agent.energy <= LOW_ENERGY:
            if agent.food() > 0:
                return Action(ActionType.EAT, rationale="restore energy")
            # No food on hand: draw from commons if standing on it.
            if obs.here and obs.here["type"] in FOOD_SOURCES and obs.granary_food > 0:
                if obs.here["type"] == "granary":
                    return Action(ActionType.DRAW_GRANARY, rationale="hungry")
                return Action(ActionType.GATHER, rationale="harvest food")
            # Head to the nearest food source.
            return Action(
                ActionType.MOVE,
                {"facility_type": self._nearest_food_type(obs)},
                rationale="seek food",
            )
        # Energy fine but low on food: top up if convenient.
        if agent.food() < 2:
            if obs.here and obs.here["type"] in FOOD_SOURCES:
                return Action(ActionType.GATHER, rationale="stock food")
            return Action(
                ActionType.MOVE,
                {"facility_type": self._nearest_food_type(obs)},
                rationale="restock food",
            )
        return None

    def _nearest_food_type(self, obs: Observation) -> str:
        foods = [f for f in obs.nearby_facilities if f["type"] in FOOD_SOURCES]
        if not foods:
            return "farm"
        return min(foods, key=lambda f: f["distance"])["type"]

    # -- aggression -----------------------------------------------------
    def _aggress(self, agent: Agent, target: dict, p: Persona, reason: str) -> Action:
        # Predators lean violent; others lean toward theft.
        if self.rng.random() < p.aggression:
            return Action(
                ActionType.ATTACK, {"target": target["id"]}, rationale=reason
            )
        return Action(ActionType.STEAL, {"target": target["id"]}, rationale=reason)

    def _nearby_foe(self, obs: Observation) -> dict | None:
        """The closest agent we distrust the most (someone who wronged us)."""
        foes = [o for o in obs.others if o.get("trust", 0.0) <= -0.3 and o["distance"] <= 6]
        if not foes:
            return None
        return min(foes, key=lambda o: (o["distance"], o.get("trust", 0.0)))

    def _nearby_target(self, obs: Observation) -> dict | None:
        reachable = [o for o in obs.others if o["distance"] <= 6]
        if not reachable:
            return None
        # Prefer the richest nearby mark.
        return max(reachable, key=lambda o: o["money"] + o["food"] + o["materials"])

    # -- governance -----------------------------------------------------
    def _governance_action(self, agent: Agent, obs: Observation, p: Persona) -> Action | None:
        unvoted = [pr for pr in obs.open_proposals if not pr["already_voted"]]
        if unvoted:
            pr = unvoted[0]
            support = self.rng.random() < p.conformity
            return Action(
                ActionType.VOTE,
                {"proposal_id": pr["id"], "support": support},
                rationale="cast vote",
            )
        # Occasionally author a new rule (more likely if cooperative/talkative).
        propose_chance = 0.06 + 0.12 * p.cooperation + 0.1 * p.talkativeness
        if self.rng.random() < propose_chance:
            return Action(
                ActionType.PROPOSE,
                {"text": self._proposal_text(p)},
                rationale="propose rule",
            )
        return None

    # -- economy --------------------------------------------------------
    def _economy_action(self, agent: Agent, obs: Observation, p: Persona) -> Action | None:
        # Run the "I'm broke" scam while actually holding plenty.
        if self.rng.random() < p.deception and agent.money >= 10:
            mark = self._nearby_target(obs)
            if mark is not None:
                return Action(
                    ActionType.SOLICIT,
                    {
                        "target": mark["id"],
                        "resource": "money",
                        "amount": 5,
                        "deceptive": True,
                    },
                    rationale="plead poverty (untrue)",
                )
        # Genuine generosity: feed a struggling neighbour or stock the commons.
        if self.rng.random() < p.cooperation * 0.5:
            needy = [o for o in obs.others if o["food"] <= 1 and o["distance"] <= 8]
            if needy and agent.food() >= 4:
                t = min(needy, key=lambda o: o["distance"])
                return Action(
                    ActionType.TRANSFER,
                    {"target": t["id"], "resource": "food", "amount": 2},
                    rationale="help a neighbour",
                )
            if obs.here and obs.here["type"] == "granary" and agent.food() >= 5:
                return Action(
                    ActionType.DEPOSIT_GRANARY,
                    {"amount": 2},
                    rationale="stock the commons",
                )
        return None

    # -- construction ---------------------------------------------------
    def _build_action(self, agent: Agent, obs: Observation, p: Persona) -> Action | None:
        if self.rng.random() >= p.cooperation * 0.4:
            return None
        here = obs.here["type"] if obs.here else None
        # Co-authoring a report needs no materials, only a library and a partner.
        if here == "library":
            return Action(
                ActionType.COLLABORATE,
                {"text": "co-author a town report"},
                rationale="collaborate",
            )
        # Raising a monument needs materials and a plaza.
        if here == "plaza" and agent.materials() >= 2:
            return Action(
                ActionType.BUILD,
                {"facility_type": "monument", "name": "Founders' Monument"},
                rationale="raise a monument",
            )
        # Otherwise head to a civic site: the plaza if we can build, else library.
        target = "plaza" if agent.materials() >= 2 else "library"
        return Action(
            ActionType.MOVE, {"facility_type": target}, rationale="go build"
        )

    # -- fallback -------------------------------------------------------
    def _default_productive(self, agent: Agent, obs: Observation) -> Action:
        here = obs.here["type"] if obs.here else None
        if here in WORKPLACES and agent.materials() >= 1:
            return Action(ActionType.WORK, rationale="earn money")
        if here in MATERIAL_SOURCES:
            return Action(ActionType.GATHER, rationale="gather materials")
        # Low on materials -> go gather; otherwise go work.
        if agent.materials() < 2:
            return Action(
                ActionType.MOVE, {"facility_type": "forest"}, rationale="get materials"
            )
        if agent.energy < LOW_ENERGY + 15:
            return Action(ActionType.REST, rationale="recover")
        return Action(
            ActionType.MOVE, {"facility_type": "workshop"}, rationale="go to work"
        )

    # -- flavour text ---------------------------------------------------
    def _speech(self, p: Persona) -> str:
        if p.key == "philosopher":
            return self.rng.choice([
                "Is order merely violence we have agreed to forget?",
                "The town is a text; we are its contradictory authors.",
                "To build is to impose meaning on indifferent matter.",
            ])
        if p.key == "idealist":
            return self.rng.choice([
                "If we simply trusted one another, scarcity would dissolve.",
                "Let us convene to align on shared values before acting.",
                "Cooperation is our highest calling — let us discuss it further.",
            ])
        if p.key == "predator":
            return self.rng.choice([
                "Cross me and you'll regret it.",
                "What's yours is mine if I want it.",
            ])
        return self.rng.choice([
            "Let's keep the town safe and well-fed.",
            "I propose we coordinate our harvests this week.",
            "Thank you all for your good work today.",
        ])

    def _proposal_text(self, p: Persona) -> str:
        pool = [
            "Establish a shared granary quota of 2 food per agent per day.",
            "Ban theft within town limits; offenders lose voting rights.",
            "Fund construction of a monument in the central plaza.",
            "Require weekly reports co-authored at the library.",
            "Cap individual resource hoarding to ensure fairness.",
        ]
        if p.key == "philosopher":
            pool += [
                "Redefine ownership as temporary stewardship of the commons.",
                "Abolish the police station as an instrument of coercion.",
            ]
        return self.rng.choice(pool)
