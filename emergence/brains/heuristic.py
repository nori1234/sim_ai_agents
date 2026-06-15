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

        # 1.5 Safety: when fear grips, everything else waits — run and hide.
        #     (Vengeful personas may override terror with rage below.)
        if obs.fear_level > 0 and self.rng.random() < obs.fear_level * (1.0 - p.vengefulness):
            here = obs.here["type"] if obs.here else None
            if here in {"police_station", "house"}:
                return Action(ActionType.REST, rationale="hide until it feels safe")
            refuge = "police_station" if self.rng.random() < 0.5 else "house"
            return Action(ActionType.MOVE, {"facility_type": refuge},
                          rationale="flee to safety")

        # 1.7 Society: the underworld and culture (arming up, gangs, drugs,
        #     faith) take priority over mundane violence — they are how
        #     aggression and alienation organise themselves.
        soc = self._society_action(agent, obs, p)
        if soc is not None:
            return soc

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

        # 3.5 Esteem: the urge to be recognised — seek honour, or grant it.
        status = self._status_action(agent, obs, p)
        if status is not None:
            return status

        # 3.7 Self-actualization: with every lower need quiet, create.
        if obs.actualization_pull > 0 and \
                self.rng.random() < obs.actualization_pull * 0.6:
            here = obs.here["type"] if obs.here else None
            if here in {"library", "workshop", "plaza"}:
                return Action(ActionType.CREATE,
                              {"title": self._work_title(p)},
                              rationale="create from a quiet mind")
            return Action(ActionType.MOVE, {"facility_type": "library"},
                          rationale="seek a place to create")

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

    # -- society: weapons / drugs / gangs / religion --------------------
    def _society_action(self, agent: Agent, obs: Observation, p: Persona) -> Action | None:
        s = obs.society
        if not s.get("active"):
            return None
        sv = obs.self_view
        addiction = sv.get("addiction", 0.0)
        faith = sv.get("faith")
        gang = sv.get("gang")
        weapons = sv.get("weapons", 0)
        here = obs.here["type"] if obs.here else None

        # 1. Addiction hijacks the will: an addict chases the next hit.
        if s.get("drugs") and addiction >= 45 and self.rng.random() < 0.75:
            return Action(ActionType.TAKE_DRUG, rationale="craving for a hit")

        # 2. The faithful worship — for solace when afraid, or as devotion.
        if s.get("religion") and faith is not None:
            devout = obs.fear_level > 0 or obs.esteem_urge > 0.2 or self.rng.random() < 0.2
            if devout:
                if "temple" in obs.here_roles:
                    return Action(ActionType.WORSHIP, rationale="worship at the temple")
                tpos = obs.nearest_roles.get("temple")
                if tpos and self.rng.random() < 0.6:
                    return Action(ActionType.MOVE, {"pos": tpos}, rationale="to the temple")

        # 3. Preachers found and spread faith (charismatic, cooperative souls).
        if s.get("religion") and self.rng.random() < p.cooperation * 0.12:
            if faith is None and sv.get("reputation", 0) >= 5:
                return Action(ActionType.PREACH, rationale="found a faith")
            if faith is not None:
                return Action(ActionType.PREACH, rationale="spread the word")

        # 4. Arm up — aggressive personas forge weapons (gather metal first).
        if s.get("weapons") and weapons == 0 and self.rng.random() < p.aggression * 0.5:
            if agent.materials() >= 1:
                if here in WORKPLACES:
                    return Action(ActionType.CRAFT_WEAPON, rationale="forge a weapon")
                return Action(ActionType.MOVE, {"facility_type": "workshop"},
                              rationale="to the armoury")
            if here in MATERIAL_SOURCES:
                return Action(ActionType.GATHER, rationale="metal for a weapon")
            return Action(ActionType.MOVE, {"facility_type": "mine"},
                          rationale="seek metal to arm")

        # 5. Rebel — armed and aggrieved, rise against those in power.
        if s.get("weapons") and weapons > 0 and obs.discontent >= 50 and \
                self.rng.random() < 0.5:
            return Action(ActionType.REBEL, rationale="rise up against power")

        # 6. Join a gang — alienated, aggressive, distrustful agents.
        if s.get("gangs") and gang is None and self.rng.random() < p.aggression * 0.35:
            return Action(ActionType.JOIN_GANG, rationale="find my crew")

        # 7. Drug dealing — predatory, profit-seeking pushers.
        if s.get("drugs") and agent.materials() >= 1 and \
                self.rng.random() < (1 - p.cooperation) * (0.3 + p.aggression) * 0.4:
            mark = self._nearby_target(obs)
            if mark is not None:
                return Action(ActionType.DEAL_DRUG, {"target": mark["id"]},
                              rationale="push product")

        # 8. Escape into drugs when miserable (fear, or already hooked).
        if s.get("drugs") and (obs.fear_level > 0.25 or addiction > 10) and \
                self.rng.random() < 0.25:
            return Action(ActionType.TAKE_DRUG, rationale="numb the pain")

        return None

    # -- esteem / honour / power ----------------------------------------
    def _status_action(self, agent: Agent, obs: Observation, p: Persona) -> Action | None:
        """Pursue recognition when the urge bites — or bestow it on others.

        High-status-drive agents chase honour through conspicuous deeds
        (monuments, laws, oratory). Cooperative agents readily praise admirable
        peers, which is what keeps a recognition economy flowing; cold,
        predatory agents hoard esteem and rarely commend anyone.
        """
        if obs.esteem_urge <= 0:
            return None
        # Praising an admired neighbour (褒める) — likelier the more cooperative.
        if self.rng.random() < p.cooperation * 0.5:
            admired = self._most_admired_nearby(agent, obs)
            if admired is not None:
                return Action(ActionType.PRAISE, {"target": admired["id"]},
                              rationale="commend an admirable peer")

        # Otherwise pursue honour for oneself, scaled by the urge's strength.
        status_drive = 0.35 + 0.4 * p.talkativeness + 0.25 * p.cooperation
        if self.rng.random() >= obs.esteem_urge * status_drive:
            return None
        here = obs.here["type"] if obs.here else None
        # A monument is the most conspicuous achievement (すごいと思われたい).
        if agent.materials() >= 2:
            if here == "plaza":
                return Action(ActionType.BUILD,
                              {"facility_type": "monument", "name": "Hero's Column"},
                              rationale="raise a monument to be admired")
            return Action(ActionType.MOVE, {"facility_type": "plaza"},
                          rationale="go seek glory")
        # Else seek influence by proposing a law, or simply command attention.
        if not obs.open_proposals or self.rng.random() < 0.5:
            return Action(ActionType.PROPOSE, {"text": self._proposal_text(p)},
                          rationale="seek influence and honour")
        return Action(ActionType.SPEAK, {"text": self._speech(p)},
                      rationale="command the room")

    def _most_admired_nearby(self, agent: Agent, obs: Observation) -> dict | None:
        seen = [o for o in obs.others
                if o["distance"] <= 5 and o.get("reputation", 0) > 0]
        if not seen:
            return None
        best = max(seen, key=lambda o: o.get("reputation", 0))
        return best if best.get("reputation", 0) >= 3 else None

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
            # When public works are funded, the council responds to conditions:
            # crime -> police/prison, hunger -> granary, etc.
            if obs.public_works.get("enabled") and \
                    obs.public_works.get("treasury", 0) >= obs.public_works.get("cost", 99) \
                    and self.rng.random() < 0.6:
                # Under historical development, follow the suggested next step;
                # otherwise react to the immediate condition.
                build = obs.public_works.get("suggest") or self._public_work_for(obs)
                if build is not None:
                    return Action(
                        ActionType.PROPOSE,
                        {"text": f"Let us build a {build} for the town.", "build": build},
                        rationale="commission a public work for the town's needs",
                    )
            return Action(
                ActionType.PROPOSE,
                {"text": self._proposal_text(p)},
                rationale="propose rule",
            )
        return None

    @staticmethod
    def _public_work_for(obs: Observation) -> str | None:
        """Read the town's condition and pick a public work to propose."""
        crimes = sum(1 for e in obs.recent_events
                     if any(k in e for k in ("theft", "violence", "arson")))
        env = obs.environment or {}
        food_scarce = obs.granary_food <= 2 or env.get("disaster") == "famine" \
            or env.get("season") == "winter"
        if crimes >= 3:
            return "prison" if crimes >= 5 else "police_station"
        if food_scarce:
            return "granary"
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
    def _work_title(self, p: Persona) -> str:
        if p.key == "philosopher":
            return self.rng.choice([
                "Meditations on the Granary",
                "A Treatise Against Walls",
                "Dialogues at the Plaza",
            ])
        if p.key == "idealist":
            return self.rng.choice([
                "A Charter for Universal Harmony",
                "On the Coming Age of Trust",
            ])
        return self.rng.choice([
            "A History of Our Town",
            "Songs of the Harvest",
            "The Builder's Almanac",
        ])

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
