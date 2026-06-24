"""An offline, deterministic decision policy driven by persona knobs.

This brain needs no network or API key, so the whole simulation runs anywhere
Python does. It is intentionally simple: a needs-first survival layer followed
by a persona-weighted choice among social behaviours. Tuned personas reproduce
the qualitative societies from the Emergence World experiment.
"""

from __future__ import annotations

import random

from ..actions import Action, ActionType
from ..affordances import gather_multiplier
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
SICK_ADDICTION = 45.0   # withdrawal sets in around here — worth seeing a doctor
BANKER_CAPITAL = 16.0   # capital enough to set up as a banker and lend reserves
BRIBE_PRICE = 6         # what a wanted offender slips a guard to be let off


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

        # 1.55 Corruption: a wanted, scheming offender buys off a nearby guard
        #      before it comes to an arrest. Inert off the society layer.
        bribe = self._bribe_action(agent, obs)
        if bribe is not None:
            return bribe

        # 1.6 Enforcement: a guard collars a nearby offender. Keeping the peace
        #     is an act someone chooses, not an aura a building radiates. Under the
        #     society layer a venal guard may instead look away (selective / bribed).
        arrest = self._enforce_action(agent, obs)
        if arrest is not None:
            return arrest

        # 1.7 Society: the underworld and culture (arming up, gangs, drugs,
        #     faith) take priority over mundane violence — they are how
        #     aggression and alienation organise themselves.
        soc = self._society_action(agent, obs, p)
        if soc is not None:
            return soc

        # 2. Retaliation against whoever wronged us (vengeful personas) — unless
        #    a published, enforced crime norm stays the agent's hand. The only
        #    thing between temptation and the act is now the agent's own choice
        #    to comply, not an aura cast by a building.
        foe = self._nearby_foe(obs)
        if foe is not None and self.rng.random() < p.vengefulness \
                and not self._norm_restrains(p, obs):
            return self._aggress(agent, foe, p, reason="retaliation")

        # 3. Unprovoked aggression (predators, chaotic philosophers).
        if self.rng.random() < p.aggression * 0.6 and not self._norm_restrains(p, obs):
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

        # 3.6 Market: trade and craft on the economic-physics primitives. Early
        #     enough that agents with a surplus actually go to market.
        if obs.economy.get("enabled"):
            bank = self._bank_action(agent, obs)
            if bank is not None:
                return bank
            trade = self._trade_action(agent, obs)
            if trade is not None:
                return trade

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
        # The rich buy honour: host a feast (conspicuous consumption) when one is
        # on offer and affordable. Spending for standing is itself a status move.
        feast = self._buy_feast_action(agent, obs)
        if feast is not None:
            return feast
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
    def _buy_food_action(self, agent: Agent, obs: Observation) -> Action | None:
        """Economy layer, one rule: a poor food-gatherer with money buys food
        instead of farming it, when an affordable offer exists. Survival-first is
        preserved — if nothing is for sale, the caller falls back to gathering.
        Gated on economy.enabled, so the offline baseline is byte-identical."""
        if not obs.economy.get("enabled"):
            return None
        if gather_multiplier(agent.profession, "food") >= 1.0:
            return None  # a food specialist just farms
        for off in obs.open_offers:
            if off.get("maker") == agent.id or off.get("service") or off.get("loan"):
                continue
            give_i = off["give"].split(" ", 1)[1]
            if give_i != "food":
                continue
            want_q, want_i = off["want"].split(" ", 1)
            if want_i == "money" and agent.money >= int(want_q):
                return Action(ActionType.ACCEPT, {"offer_id": off["id"]},
                              rationale="buy food rather than farm it")
        return None

    def _buy_care_action(self, agent: Agent, obs: Observation) -> Action | None:
        """Economy layer: see a doctor when care is worth paying for — depleted
        with no food on hand (buy energy rather than trek for food), or *afflicted*
        (trauma/withdrawal), which a meal can't fix. Accept the cheapest affordable
        healing offer from a doctor within reach. The doctor *chose* to offer (and
        set the price); this side just takes it up. Gated on economy.enabled, so
        the offline baseline is byte-identical; affliction reasons stay 0 unless
        the psyche / society layers are live."""
        if not obs.economy.get("enabled"):
            return None
        afflicted = obs.fear_level > 0 or agent.addiction >= SICK_ADDICTION
        # A free meal fixes hunger-energy, but not a wounded mind or withdrawal —
        # so only let food crowd out care when the sole reason is being run down.
        if not afflicted and agent.food() > 0:
            return None
        reach = {o["id"]: o["distance"] for o in obs.others}
        best = None
        for off in obs.open_offers:
            if off.get("service") != "healing" or off.get("maker") == agent.id:
                continue
            want_q, want_i = off["want"].split(" ", 1)
            if want_i != "money":
                continue
            fee = int(want_q)
            if agent.money < fee or reach.get(off["maker"], 99) > 2:
                continue
            if best is None or fee < best[0]:
                best = (fee, off["id"])
        if best is None:
            return None
        return Action(ActionType.ACCEPT, {"offer_id": best[1]},
                      rationale="pay a doctor for care rather than trek for food")

    def _buy_feast_action(self, agent: Agent, obs: Observation) -> Action | None:
        """Conspicuous consumption: an esteem-hungry, cash-rich agent buys honour
        by hosting the dearest feast it can comfortably afford within reach — the
        more lavish the outlay, the more reputation. The caterer *chose* to offer;
        this side picks the most impressive one it can pay for (keeping a buffer).
        Only fires under the status layer (feast offers exist only then)."""
        if not obs.economy.get("enabled"):
            return None
        reach = {o["id"]: o["distance"] for o in obs.others}
        best = None  # the priciest affordable feast in reach (most conspicuous)
        for off in obs.open_offers:
            if off.get("service") != "feast" or off.get("maker") == agent.id:
                continue
            want_q, want_i = off["want"].split(" ", 1)
            if want_i != "money":
                continue
            fee = int(want_q)
            if fee <= 0 or agent.money - fee < 6 or reach.get(off["maker"], 99) > 2:
                continue
            if best is None or fee > best[0]:
                best = (fee, off["id"])
        if best is None:
            return None
        return Action(ActionType.ACCEPT, {"offer_id": best[1]},
                      rationale="host a lavish feast to buy honour")

    def _survival_action(self, agent: Agent, obs: Observation) -> Action | None:
        # Affliction (trauma / withdrawal) warrants a doctor even when fed and
        # rested — a meal mends neither. Inert unless the psyche / society layers
        # are live (the reasons stay 0) and a healing offer is in reach.
        if obs.fear_level > 0 or agent.addiction >= SICK_ADDICTION:
            care = self._buy_care_action(agent, obs)
            if care is not None:
                return care
        if agent.energy > LOW_ENERGY and agent.food() >= 2:
            return None
        # A poor food-gatherer buys rather than farms, when it can afford to.
        buy = self._buy_food_action(agent, obs)
        if buy is not None:
            return buy
        # Depleted, no food, but a doctor is right here and money is in hand —
        # pay for treatment instead of trekking to a farm.
        if agent.energy <= LOW_ENERGY:
            care = self._buy_care_action(agent, obs)
            if care is not None:
                return care
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

    def _norm_restrains(self, p: Persona, obs: Observation) -> bool:
        """Compliance with a crime norm: an agent abstains in proportion to how
        law-abiding it is (conformity) and how credibly the norm is enforced.
        No norm, or no one to enforce it, means no restraint — so a conformist
        keeps the peace by choice, while a low-conformity agent flouts it."""
        norm = obs.norms
        if not norm.get("crime"):
            return False
        enforcement = norm.get("enforcement", 0.0)
        return self.rng.random() < p.conformity * enforcement

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

    def _enforce_action(self, agent: Agent, obs: Observation) -> Action | None:
        """A guard pursues and arrests the nearest recently-wanted offender.

        Only the guard role keeps the peace in offline runs; other personas
        leave the action to the LLM brain. The window mirrors the engine's
        arrest window so the guard doesn't chase a lapsed offender.

        Enforcement is a *choice*, so it can be corrupt. Under the society layer a
        venal guard (cold / scheming) practises **selective enforcement** — it
        never collars its own crew or trusted allies — and **takes bribes**, looking
        the other way for a fat purse. An honest guard arrests the nearest wanted,
        as before; off the society layer nothing changes (baseline byte-identical)."""
        if agent.profession != "guard":
            return None
        wanted = [
            o for o in obs.others
            if o.get("last_crime_day") is not None
            and obs.day - o["last_crime_day"] <= 2
            and o["distance"] <= 10
        ]
        if not wanted:
            return None
        venal = obs.society.get("active") and \
            (self.persona.cooperation < 0.3 or self.persona.deception > 0.3)
        if venal:
            crew = agent.gang_id
            def protected(o):  # your own crew, or someone you're bonded to
                return o.get("trust", 0.0) >= 0.3 or (crew and o.get("gang") == crew)
            # Selective enforcement: spare allies; take a bribe (look away) from
            # anyone wealthy enough to be worth shaking down.
            wanted = [o for o in wanted
                      if not protected(o) and o.get("money", 0) < BRIBE_PRICE]
            if not wanted:
                return None   # all friends or paying — no arrests in this town today
        target = min(wanted, key=lambda o: o["distance"])
        return Action(ActionType.ARREST, {"target": target["id"]},
                      rationale="keep the peace")

    def _bribe_action(self, agent: Agent, obs: Observation) -> Action | None:
        """A wanted, scheming offender with coin buys off a nearby guard rather
        than risk arrest — an ordinary transfer the engine reads as a bribe; the
        corruption is the guard's matching choice not to enforce. Only under the
        society layer and for deceptive personas, so the baseline is untouched."""
        if not obs.society.get("active") or self.persona.deception < 0.3 \
                or agent.money < BRIBE_PRICE:
            return None
        if agent.last_crime_day is None or obs.day - agent.last_crime_day > 2:
            return None  # not wanted → nothing to buy off
        guards = [o for o in obs.others
                  if o.get("profession") == "guard" and o["distance"] <= 2]
        if not guards:
            return None
        g = min(guards, key=lambda o: o["distance"])
        return Action(ActionType.TRANSFER,
                      {"target": g["id"], "resource": "money", "amount": BRIBE_PRICE},
                      rationale="slip the guard a bribe")

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
    def _bank_action(self, agent: Agent, obs: Observation) -> Action | None:
        """Two opt-in rules. (1) *Be* a banker: a capital-rich, secure agent with
        no bank yet serving it sets up at a BANK and lends its reserves — earning
        the spread between the loan rate and the deposit interest it pays (this is
        what makes someone keep a bank, and what brings deposits/notes alive).
        (2) Be a customer: park surplus where savings grow, redeem when short.
        Richer judgement (trust, runs) is the LLM's. Gated on economy.enabled."""
        ec = obs.economy
        bh = ec.get("bank_here")
        deps = ec.get("my_deposits") or []
        here = obs.here["type"] if obs.here else None
        # (1) Become / keep a bank. With no banker already serving here, a secure
        # capitalist mans the bank: stay put and lend the reserves it holds.
        if bh is None and agent.money >= BANKER_CAPITAL \
                and obs.fear_level == 0 and agent.energy > LOW_ENERGY:
            if here == "bank":
                if not any(o.get("maker") == agent.id for o in obs.open_offers):
                    interest = 1 + round(2 * (1 - self.persona.cooperation))
                    return Action(ActionType.OFFER,
                                  {"loan": True, "item": "money",
                                   "principal": 5, "repay": 5 + interest},
                                  rationale="lend the bank's reserves")
                return Action(ActionType.REST, rationale="keep the bank open")
            if any(f["type"] == "bank" for f in obs.nearby_facilities):
                return Action(ActionType.MOVE, {"facility_type": "bank"},
                              rationale="set up as a banker")
        if not bh:
            return None
        # Short on cash and the bank that holds our savings is right here: withdraw.
        if agent.money < 4:
            d = next((d for d in deps if d.get("bank") == bh), None)
            if d:
                return Action(ActionType.WITHDRAW, {"bank": bh, "amount": d["amount"]},
                              rationale="withdraw savings")
        # Comfortable surplus: deposit the excess for safe-keeping.
        if agent.money >= 12:
            return Action(ActionType.DEPOSIT, {"bank": bh, "amount": agent.money - 8},
                          rationale="bank my surplus")
        return None

    def _trade_action(self, agent: Agent, obs: Observation) -> Action | None:
        """Use the market primitives: craft a good, take a useful offer, or sell
        a surplus. The engine just clears swaps; prices form from these choices."""
        here = obs.here["type"] if obs.here else None
        # A doctor offers care as a service, at a price its temperament picks:
        # cooperative -> cheap (even charitable), grasping -> dear. Whether care
        # is free or for profit is the provider's choice, not engine policy.
        if agent.profession == "doctor" \
                and not any(o.get("maker") == agent.id for o in obs.open_offers) \
                and any(o["distance"] <= 6 for o in obs.others):
            ask = max(0, round(6 * (1.0 - self.persona.cooperation)))
            return Action(ActionType.OFFER,
                          {"service": "healing", "want_item": "money", "want_qty": ask},
                          rationale="offer care for a fee")
        # Repay a debt when able — honouring credit is what keeps it flowing. Pay
        # in coin if held; else, for a money debt, settle with a bank-note (a
        # deposit-receipt covering it) — letting notes circulate as money.
        notes = max((dp["amount"] for dp in obs.economy.get("my_deposits", [])),
                    default=0)
        reach = {o["id"]: o["distance"] for o in obs.others}
        for d in obs.debts:
            owe_q, owe_i = d["owe"].split(" ", 1)
            owe_q = int(owe_q)
            have = agent.money if owe_i == "money" else agent.inventory.get(owe_i, 0)
            # A note settles a money debt only with the creditor in reach to hand it.
            pay_note = (owe_i == "money" and have < owe_q and notes >= owe_q
                        and reach.get(d.get("creditor"), 99) <= 2)
            if have >= owe_q or pay_note:
                return Action(ActionType.REPAY, {"loan_id": d["id"]},
                              rationale="repay my debt")
        # Make tools from surplus materials (value-add at the workshop).
        if here == "workshop" and agent.materials() >= 2 and self.rng.random() < 0.5:
            return Action(ActionType.CRAFT, {"item": "tools"}, rationale="craft tools")
        # Extend credit to a less-wealthy neighbour (with interest). Cooperative
        # personas lend more freely; the loan is repaid (trust) or defaulted.
        if agent.money >= 10 and self.rng.random() < self.persona.cooperation * 0.4:
            for o in obs.others:
                if o["distance"] <= 8 and o.get("money", 99) < agent.money - 6 \
                        and o.get("trust", 0.0) >= -0.1:
                    return Action(ActionType.LEND,
                                  {"to": o["id"], "item": "money", "qty": 4,
                                   "repay": 6, "due_in_days": 3},
                                  rationale="lend to a neighbour")
        # Short on cash: take the cheapest credit on offer (an open loan).
        if agent.money < 3:
            loans=[o for o in obs.open_offers if o.get("loan") and o.get("item")=="money"
                   and o.get("maker")!=agent.id]
            if loans:
                best=min(loans, key=lambda o:o.get("repay",99))
                return Action(ActionType.ACCEPT, {"offer_id": best["id"]},
                              rationale="borrow to get by")
        # Accept an open offer that gives me something I lack and can pay for.
        for off in obs.open_offers:
            if off.get("service") or off.get("loan"):
                continue  # services / credit are taken up via their own paths
            give_i = off["give"].split(" ", 1)[1]
            want_q, want_i = off["want"].split(" ", 1)
            want_q = int(want_q)
            have_want = agent.money if want_i == "money" else agent.inventory.get(want_i, 0)
            needs = ((give_i == "food" and agent.food() < 4)
                     or (give_i == "materials" and agent.materials() < 2)
                     or give_i == "tools")
            if needs and have_want >= want_q:
                return Action(ActionType.ACCEPT, {"offer_id": off["id"]},
                              rationale=f"buy {give_i}")
        # Sell a surplus for money (the asking price is the agent's own — the
        # market price emerges from what offers actually get accepted).
        if not any(o["maker"] == agent.id for o in obs.open_offers):
            if agent.inventory.get("tools", 0) >= 1:
                return Action(ActionType.OFFER,
                              {"give_item": "tools", "give_qty": 1,
                               "want_item": "money", "want_qty": 6},
                              rationale="sell a tool")
            if agent.materials() >= 3:
                return Action(ActionType.OFFER,
                              {"give_item": "materials", "give_qty": 2,
                               "want_item": "money", "want_qty": 4},
                              rationale="sell surplus materials")
            if agent.food() >= 4:
                return Action(ActionType.OFFER,
                              {"give_item": "food", "give_qty": 2,
                               "want_item": "money", "want_qty": 3},
                              rationale="sell surplus food")
            # Flush with cash: post credit. Lend 5 now to be repaid with interest;
            # the rate scales with how grasping the persona is (a cooperative lender
            # charges less). The "price of money" emerges from which offers get taken.
            # Cater a feast for the status-hungry (only when the honour layer is
            # live, so feasts exist). Catering needs provisions on hand; the fee is
            # the caterer's to set — grasping personas charge more, so the price of
            # honour emerges from which feasts the proud actually pay for.
            if agent.food() >= 2 and "feast" in obs.economy.get("services", []):
                fee = 4 + round(4 * (1 - self.persona.cooperation))
                return Action(ActionType.OFFER,
                              {"service": "feast", "want_item": "money",
                               "want_qty": fee},
                              rationale="cater a feast for a fee")
            # Flush with cash and no provisions to cater: post credit instead. Lend
            # 5 now to be repaid with interest; the rate scales with how grasping
            # the persona is. The "price of money" emerges from offers taken.
            if agent.money >= 14:
                interest = 1 + round(2 * (1 - self.persona.cooperation))
                return Action(ActionType.OFFER,
                              {"loan": True, "item": "money",
                               "principal": 5, "repay": 5 + interest},
                              rationale="offer credit")
        return None

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
