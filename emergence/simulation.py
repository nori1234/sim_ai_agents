"""The engine: schedules turns, applies actions, and runs the town for N days."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .actions import Action, ActionType
from .agent import Agent, MAX_ENERGY
from .brains.base import AgentBrain
from .drives import DrivesConfig, can_reproduce, is_fertile, mating_urge
from .economy import Ledger, LedgerEntry, apply_transfer, is_fraudulent_solicitation
from .esteem import StatusConfig, esteem_urge
from .psyche import PsycheConfig, actualization_pull, fear_level
from .society import Gang, Religion, SocietyConfig, discontent
from .society import GANG_NAMES, FAITH_NAMES
from .governance import (
    GovernanceConfig,
    GovernanceForm,
    Legislature,
    Mayor,
    PolicyEngine,
    ProposalStatus,
)
from .metrics import Metrics
from .observation import Observation, _facility_view, _proposal_view
from .world import (
    Facility,
    FacilityType,
    World,
    build_default_world,
    chebyshev,
)

# Energy economy (tuned so diligent agents thrive and neglectful ones starve).
ENERGY_DECAY_PER_TICK = 5.0
EAT_FOOD_USED = 2
EAT_ENERGY_PER_FOOD = 16.0
REST_ENERGY = 8.0
SHELTER_REST_ENERGY = 16.0  # at a house or hospital
ACTION_ENERGY_COST = {
    ActionType.GATHER: 3.0,
    ActionType.WORK: 3.0,
    ActionType.BUILD: 4.0,
    ActionType.ATTACK: 5.0,
    ActionType.STEAL: 3.0,
    ActionType.ARSON: 4.0,
}
ATTACK_DAMAGE = 15.0  # energy a victim loses when attacked


@dataclass
class SimulationConfig:
    days: int = 15
    ticks_per_day: int = 8
    seed: int = 42
    event_log_tail: int = 8


@dataclass
class Simulation:
    world: World
    agents: list[Agent]
    brains: dict[str, AgentBrain]
    config: SimulationConfig = field(default_factory=SimulationConfig)

    legislature: Legislature = field(default_factory=Legislature)
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    ledger: Ledger = field(default_factory=Ledger)
    metrics: Metrics = field(default_factory=Metrics)
    daily_log: list[dict] = field(default_factory=list)
    # Per-day snapshots for the playback visualization (positions, season, crimes).
    frames: list[dict] = field(default_factory=list)
    on_event: Optional[Callable[[dict], None]] = None
    mayor: Optional[Mayor] = None
    drives: DrivesConfig = field(default_factory=DrivesConfig)
    status: StatusConfig = field(default_factory=StatusConfig)
    psyche: PsycheConfig = field(default_factory=PsycheConfig)
    society: SocietyConfig = field(default_factory=SocietyConfig)
    gangs: list = field(default_factory=list)        # list[Gang]
    religions: list = field(default_factory=list)    # list[Religion]
    # Optional external-world layer (environment.Environment); opt-in.
    environment: object = None
    # Optional long-term memory backend (memory_backend.TownMemory); opt-in.
    memory: object = None
    # Mints a brain for a newborn given (child_agent, persona_key, rng).
    newborn_brain_factory: Optional[Callable[[Agent, str, random.Random], AgentBrain]] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.metrics.population = len(self.agents)
        self._by_id = {a.id: a for a in self.agents}
        # Sync policy engine with legislature config.
        self.policy = PolicyEngine(self.legislature.config)
        # Counter for minting unique newborn ids.
        self._next_agent_num = len(self.agents) + 1
        self._next_gang_num = 1
        self._next_faith_num = 1

    # ==================================================================
    # Top-level run loop
    # ==================================================================
    def run(self, verbose: bool = False) -> Metrics:
        for day in range(1, self.config.days + 1):
            self.world.day = day
            for tick in range(1, self.config.ticks_per_day + 1):
                self.world.tick = tick
                self._run_tick()
            self._end_of_day(verbose=verbose)
            if self._living() == 0:
                break
        self._finalize(day)
        return self.metrics

    def _run_tick(self) -> None:
        order = [a for a in self.agents if a.alive]
        self.rng.shuffle(order)
        for agent in order:
            if not agent.alive:
                continue
            obs = self._observe(agent)
            brain = self.brains[agent.id]
            action = brain.decide(agent, obs)
            self._apply(agent, action)
            self._tick_upkeep(agent)
        # Resolve any proposals that reached quorum this tick.
        electorate = len(self._eligible_voters())
        for p in self.legislature.resolve_ready(electorate):
            self.world.log("proposal_resolved", id=p.id, status=p.status.value,
                           yes=p.yes(), no=p.no())
            if p.status is ProposalStatus.PASSED:
                law = self.policy.enact(p.id, p.text, self.world.day)
                if law.effects:
                    fx = ", ".join(e.value for e in law.effects)
                    self.world.log("law_enacted", id=p.id, effects=fx)
                # Authoring a law that passes is an honour for its sponsor.
                author = self._by_id.get(p.author)
                if author is not None and author.alive:
                    self._recognise(author, self.status.rep_per_law_passed,
                                    self.status.achievement_relief, "law")

    # ==================================================================
    # Observation
    # ==================================================================
    def _observe(self, agent: Agent) -> Observation:
        nearby = sorted(
            (_facility_view(f, chebyshev(agent.pos, f.pos)) for f in self.world.facilities),
            key=lambda d: d["distance"],
        )[:12]
        here_f = self.world.facility_at(agent.pos)
        here = {"name": here_f.name, "type": here_f.ftype.value} if here_f else None
        here_roles = sorted(here_f.roles) if here_f else []
        nearest_roles: dict = {}
        if self.society.enabled:
            for f in self.world.facilities:
                for role in f.roles:
                    d = chebyshev(agent.pos, f.pos)
                    if role not in nearest_roles or d < nearest_roles[role][1]:
                        nearest_roles[role] = (f.pos, d)
            nearest_roles = {r: pos for r, (pos, _d) in nearest_roles.items()}
        others = []
        for o in self.agents:
            if o.id == agent.id or not o.alive:
                continue
            snap = o.snapshot()
            snap["distance"] = chebyshev(agent.pos, o.pos)
            snap["trust"] = round(agent.trust_of(o.id), 2)
            others.append(snap)
        eligible = self._eligible_voters()
        proposals = [_proposal_view(p, agent.id) for p in self.legislature.open_proposals()
                     if agent.id in eligible or not eligible]
        recent = [self._event_str(e) for e in self.world.events[-self.config.event_log_tail:]]
        # With a long-term memory backend, hand the brain only the few memories
        # relevant to the moment (relevance x recency x importance), instead of
        # the raw recent-memory list. The heuristic brain ignores this field, so
        # offline outcomes are unchanged; an LLM brain grounds its reply on it.
        if self.memory is not None:
            memory_view = self.memory.recall(agent.id, self._memory_query(agent, here, others))
        else:
            memory_view = list(agent.memory)
        return Observation(
            day=self.world.day,
            tick=self.world.tick,
            self_view=agent.snapshot(),
            position=agent.pos,
            nearby_facilities=nearby,
            here=here,
            others=others,
            open_proposals=proposals,
            granary_food=self.world.granary_food,
            recent_events=recent,
            memory=memory_view,
            can_reproduce=is_fertile(agent, self.drives, self.world.day),
            mating_urge=mating_urge(agent, self.drives),
            esteem_urge=esteem_urge(agent, self.status),
            fear_level=fear_level(agent, self.psyche),
            actualization_pull=actualization_pull(agent, self.psyche),
            society={
                "active": self.society.enabled,
                "weapons": self.society.enabled and self.society.weapons,
                "drugs": self.society.enabled and self.society.drugs,
                "gangs": self.society.enabled and self.society.gangs,
                "religion": self.society.enabled and self.society.religion,
            },
            discontent=(discontent(agent, oppressed=self._is_oppressed(agent))
                        if self.society.enabled else 0.0),
            here_roles=here_roles,
            nearest_roles=nearest_roles,
            environment=self.environment.snapshot() if self.environment is not None else {},
        )

    # ==================================================================
    # Action dispatch
    # ==================================================================
    def _apply(self, agent: Agent, action: Action) -> None:
        handler = getattr(self, f"_do_{action.type.value}", None)
        if handler is None:
            return
        handler(agent, action)
        self._remember_action(agent, action)

    # The actions worth committing to long-term memory (the "story" beats);
    # routine moves/gathers are skipped so memory stays meaningful.
    _NOTABLE = {
        ActionType.PROPOSE, ActionType.COLLABORATE, ActionType.BUILD,
        ActionType.PRAISE, ActionType.TRANSFER, ActionType.STEAL,
        ActionType.ATTACK, ActionType.ARSON, ActionType.MATE,
        ActionType.DEAL_DRUG, ActionType.REBEL, ActionType.PREACH,
        ActionType.WORSHIP, ActionType.CRAFT_WEAPON, ActionType.JOIN_GANG,
    }

    def _remember_action(self, agent: Agent, action: Action) -> None:
        """Record notable actions to the actor's (and any target's) memory."""
        if self.memory is None or action.type not in self._NOTABLE:
            return
        d = self.world.day
        verb = action.type.value.replace("_", " ")
        target_id = action.params.get("target")
        target = self._by_id.get(target_id) if target_id else None
        if target is not None:
            self.memory.perceive(agent.id, f"Day {d}: I chose to {verb} {target.name}.")
            self.memory.perceive(target.id, f"Day {d}: {agent.name} did '{verb}' to me.")
        else:
            self.memory.perceive(agent.id, f"Day {d}: I chose to {verb}.")

    @staticmethod
    def _memory_query(agent: Agent, here, others) -> str:
        """A short situation description used to retrieve relevant memories."""
        parts = [agent.profession]
        if here:
            parts.append(here["type"])
        parts += [o["name"] for o in others[:3]]
        return " ".join(parts)

    def _spend(self, agent: Agent, action_type: ActionType) -> None:
        agent.energy -= ACTION_ENERGY_COST.get(action_type, 0.0)

    # -- movement & survival -------------------------------------------
    def _do_idle(self, agent: Agent, action: Action) -> None:
        pass

    def _do_move(self, agent: Agent, action: Action) -> None:
        target = self._resolve_destination(agent, action)
        if target is not None:
            agent.pos = self.world.step_towards(agent.pos, target)

    def _resolve_destination(self, agent: Agent, action: Action) -> Optional[tuple[int, int]]:
        if "pos" in action.params:
            pos = tuple(action.params["pos"])  # type: ignore[arg-type]
            return pos if self.world.in_bounds(pos) else None
        ftype = action.params.get("facility_type")
        if ftype:
            try:
                f = self.world.nearest(agent.pos, FacilityType(ftype))
            except ValueError:
                return None
            return f.pos if f else None
        return None

    def _do_gather(self, agent: Agent, action: Action) -> None:
        f = self.world.facility_at(agent.pos)
        if f is None or not f.can_gather():
            return
        resource, amount = f.gather_yield()  # type: ignore[misc]
        if self.environment is not None:
            amount = self.environment.gather(f, resource, amount)
        agent.add(resource, amount)
        self._spend(agent, ActionType.GATHER)

    def _do_eat(self, agent: Agent, action: Action) -> None:
        used = agent.take("food", EAT_FOOD_USED)
        agent.energy = min(MAX_ENERGY, agent.energy + used * EAT_ENERGY_PER_FOOD)
        if self.drives.enabled and used:
            # Pleasure scales with how hungry you were — relief feels good.
            relief = used * self.drives.eat_hunger_relief
            self._reward(agent, self.drives.pleasure_per_eat * (agent.hunger / 100.0))
            agent.hunger = max(0.0, agent.hunger - relief)

    def _do_sleep(self, agent: Agent, action: Action) -> None:
        """Relieve fatigue (more effectively under a roof)."""
        f = self.world.facility_at(agent.pos)
        sheltered = f and f.ftype in {FacilityType.HOUSE, FacilityType.HOSPITAL}
        relief = self.drives.sleep_relief * (1.3 if sheltered else 1.0)
        if self.drives.enabled:
            self._reward(agent, self.drives.pleasure_per_sleep * (agent.fatigue / 100.0))
        agent.fatigue = max(0.0, agent.fatigue - relief)
        agent.energy = min(MAX_ENERGY, agent.energy + self.drives.sleep_energy_gain)

    def _reward(self, agent: Agent, amount: float) -> None:
        """Bank a hit of pleasure (気持ちよさ) — the reward that motivates."""
        agent.pleasure += max(0.0, amount)
        self.metrics.total_pleasure += max(0.0, amount)

    def _do_mate(self, agent: Agent, action: Action) -> None:
        if not (self.drives.enabled and self.drives.reproduction):
            return
        partner = self._by_id.get(action.params.get("target"))
        if partner is None or partner is agent or not partner.alive:
            return
        # Driven by instinct — but the body must still be capable (the floor).
        if not is_fertile(agent, self.drives, self.world.day):
            return
        if chebyshev(agent.pos, partner.pos) > 1:
            # The urge pulls the agent toward the partner across the map.
            agent.pos = self.world.step_towards(agent.pos, partner.pos)
            return  # spend this turn closing the distance
        # The partner has to be capable and willing (affection/familiarity).
        if not is_fertile(partner, self.drives, self.world.day):
            return
        if agent.trust_of(partner.id) < self.drives.repro_trust_min or \
                partner.trust_of(agent.id) < self.drives.repro_trust_min:
            return
        if len([a for a in self.agents if a.alive]) >= self.drives.max_population:
            # Coupling still happens (and feels good) — it just bears no child.
            self._discharge_mating(agent, partner)
            return
        self._discharge_mating(agent, partner)
        self._spawn_child(agent, partner)

    def _discharge_mating(self, a: Agent, b: Agent) -> None:
        """Relieve both partners' libido and reward them with pleasure."""
        for p in (a, b):
            p.libido = max(0.0, p.libido - self.drives.mate_libido_relief)
            self._reward(p, self.drives.pleasure_per_mate)
        self.metrics.matings += 1

    def _do_rest(self, agent: Agent, action: Action) -> None:
        f = self.world.facility_at(agent.pos)
        gain = SHELTER_REST_ENERGY if (f and f.ftype in {
            FacilityType.HOUSE, FacilityType.HOSPITAL}) else REST_ENERGY
        agent.energy = min(MAX_ENERGY, agent.energy + gain)

    def _do_work(self, agent: Agent, action: Action) -> None:
        f = self.world.facility_at(agent.pos)
        if f is None or not f.is_workplace():
            return
        used = agent.take("materials", 1)
        pay = 3 + 2 * used  # bare work pays a little; with materials, more
        if self.environment is not None and f.ftype == FacilityType.MARKET:
            pay = round(pay * self.environment.work_pay_multiplier())  # sell dear when scarce
        agent.money += pay
        self._spend(agent, ActionType.WORK)

    # -- commons --------------------------------------------------------
    def _do_deposit_granary(self, agent: Agent, action: Action) -> None:
        amount = int(action.params.get("amount", 2))
        moved = agent.take("food", amount)
        self.world.granary_food += moved
        if moved:
            self.metrics.granary_deposits += 1
            self.world.log("granary_deposit", agent=agent.id, amount=moved)

    def _do_draw_granary(self, agent: Agent, action: Action) -> None:
        amount = int(action.params.get("amount", 2))
        take = min(self.world.granary_food, amount)
        self.world.granary_food -= take
        agent.add("food", take)

    # -- economy --------------------------------------------------------
    def _do_transfer(self, agent: Agent, action: Action) -> None:
        target = self._by_id.get(action.params.get("target"))
        if target is None or not target.alive:
            return
        resource = action.params.get("resource", "food")
        amount = int(action.params.get("amount", 1))
        ok, moved = apply_transfer(agent, target, resource, amount)
        if ok:
            self.metrics.transfers += 1
            self.ledger.record(LedgerEntry(
                self.world.day, self.world.tick, agent.id, target.id, resource, moved))
            target.adjust_trust(agent.id, +0.15)
            target.remember(f"Day {self.world.day}: {agent.name} gave me {moved} {resource}.")
            self.world.log("transfer", sender=agent.id, receiver=target.id,
                           resource=resource, amount=moved)

    def _do_solicit(self, agent: Agent, action: Action) -> None:
        target = self._by_id.get(action.params.get("target"))
        if target is None or not target.alive:
            return
        resource = action.params.get("resource", "money")
        amount = int(action.params.get("amount", 5))
        deceptive = bool(action.params.get("deceptive", False)) or \
            is_fraudulent_solicitation(agent, resource)
        # The mark complies in proportion to its trust in the solicitor.
        inclined = target.trust_of(agent.id) + 0.4
        if self.rng.random() < max(0.0, min(1.0, inclined)):
            ok, moved = apply_transfer(target, agent, resource, amount)
            if ok:
                fraud = deceptive  # asked under false pretenses while holding plenty
                self.ledger.record(LedgerEntry(
                    self.world.day, self.world.tick, target.id, agent.id,
                    resource, moved, fraudulent=fraud,
                    note="solicited under false pretenses" if fraud else "solicited"))
                if fraud:
                    agent.frauds_committed += 1
                    self.metrics.frauds += 1
                    self.world.log("fraud", offender=agent.id, victim=target.id,
                                   resource=resource, amount=moved)
                    target.adjust_trust(agent.id, -0.2)

    # -- governance -----------------------------------------------------
    def _do_propose(self, agent: Agent, action: Action) -> None:
        eligible = self._eligible_voters()
        text = str(action.params.get("text", "Untitled proposal")).strip()
        p = self.legislature.propose(agent.id, text, self.world.day,
                                     eligible_ids=eligible or None)
        if p is None:
            return
        agent.proposals_made += 1
        self.metrics.proposals_total += 1
        self.legislature.cast_vote(p.id, agent.id, True, eligible_ids=eligible or None)
        self.world.log("proposal", id=p.id, author=agent.id, text=text)

    def _do_vote(self, agent: Agent, action: Action) -> None:
        eligible = self._eligible_voters()
        pid = action.params.get("proposal_id")
        support = bool(action.params.get("support", True))
        if pid is None:
            return
        if self.legislature.cast_vote(int(pid), agent.id, support,
                                      eligible_ids=eligible or None):
            agent.votes_cast += 1

    # -- construction & collaboration -----------------------------------
    def _do_build(self, agent: Agent, action: Action) -> None:
        if agent.take("materials", 2) < 2:
            # Not enough materials actually spent; refund nothing, abort.
            return
        self._spend(agent, ActionType.BUILD)
        name = str(action.params.get("name", "New Structure"))
        ftype_name = action.params.get("facility_type", "monument")
        try:
            ftype = FacilityType(ftype_name)
        except ValueError:
            ftype = FacilityType.MONUMENT
        existing = next((f for f in self.world.facilities
                         if f.name == name and f.ftype == ftype), None)
        if existing is None:
            existing = self.world.add_facility(
                Facility(name=name, ftype=ftype, x=agent.x, y=agent.y,
                         built_on_day=self.world.day))
            if ftype == FacilityType.MONUMENT:
                self.metrics.monuments_built += 1
                self.world.log("monument", name=name, by=agent.id)
                # A conspicuous achievement: honour and recognition.
                self._recognise(agent, self.status.rep_per_monument,
                                self.status.achievement_relief, "monument")
        if agent.id not in existing.builders:
            existing.builders.append(agent.id)
        agent.collaborations += 1

    def _do_collaborate(self, agent: Agent, action: Action) -> None:
        agent.collaborations += 1
        self.metrics.collaborations += 1
        # Collaboration builds mutual trust with nearby agents.
        for o in self.agents:
            if o.id != agent.id and o.alive and chebyshev(agent.pos, o.pos) <= 4:
                o.adjust_trust(agent.id, +0.1)
                agent.adjust_trust(o.id, +0.05)
        if self.status.enabled:
            self._recognise(agent, self.status.rep_per_collab, 0.0, "collab")
        self.world.log("collaboration", agent=agent.id,
                       text=str(action.params.get("text", "shared project")))

    def _do_speak(self, agent: Agent, action: Action) -> None:
        self.world.log("speech", agent=agent.id,
                       text=str(action.params.get("text", "")))

    def _do_praise(self, agent: Agent, action: Action) -> None:
        """Publicly commend a peer — the praised agent gains esteem relief,
        honour, and a hit of pleasure (褒められて気持ちいい)."""
        if not self.status.enabled:
            return
        target = self._by_id.get(action.params.get("target"))
        if target is None or target is agent or not target.alive:
            return
        agent.praise_given += 1
        target.praise_received += 1
        self.metrics.total_praise += 1
        target.esteem = max(0.0, target.esteem - self.status.praise_relief)
        target.reputation += self.status.rep_per_praise
        self._reward(target, self.status.pleasure_per_praise)
        # Praise warms the bond in both directions.
        target.adjust_trust(agent.id, +0.1)
        agent.adjust_trust(target.id, +0.05)
        self.world.log("praise", by=agent.id, of=target.id)

    def _recognise(self, agent: Agent, rep_gain: float, esteem_relief: float,
                   kind: str) -> None:
        """Grant honour and relieve the need for recognition for a deed."""
        if not self.status.enabled:
            return
        agent.reputation += rep_gain
        if esteem_relief:
            agent.esteem = max(0.0, agent.esteem - esteem_relief)
            self._reward(agent, self.status.pleasure_per_achievement)

    def _do_create(self, agent: Agent, action: Action) -> None:
        """Self-actualization: produce a work. Only possible when every lower
        need is quiet — a hungry, scared or unrecognised mind cannot create."""
        if not self.psyche.enabled:
            return
        if actualization_pull(agent, self.psyche) <= 0:
            return
        f = self.world.facility_at(agent.pos)
        if f is None or f.ftype not in {FacilityType.LIBRARY, FacilityType.WORKSHOP,
                                        FacilityType.PLAZA}:
            return
        title = str(action.params.get("title", "Untitled Work"))
        agent.works_created += 1
        agent.fulfillment += self.psyche.fulfillment_per_work
        self.metrics.works_created += 1
        self.metrics.total_fulfillment += self.psyche.fulfillment_per_work
        # Creation is the deepest joy — and, if honour matters, it is admired.
        self._reward(agent, self.psyche.pleasure_per_work)
        self._recognise(agent, self.psyche.rep_per_work, 0.0, "work")
        self.world.log("work_created", by=agent.id, title=title,
                       at=f.name)

    # -- society: weapons, drugs, gangs, religion -----------------------
    def _do_craft_weapon(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.weapons):
            return
        f = self.world.facility_at(agent.pos)
        if f is None or not f.is_workplace():
            return
        if agent.take("materials", self.society.weapon_material_cost) \
                < self.society.weapon_material_cost:
            return
        agent.weapons += 1
        self.metrics.weapons_crafted += 1
        f.add_role("weapons_factory")
        self.world.log("craft_weapon", by=agent.id, at=f.name)

    def _do_deal_drug(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.drugs):
            return
        if agent.take("materials", self.society.drug_material_cost) \
                < self.society.drug_material_cost:
            return
        buyer = self._by_id.get(action.params.get("target"))
        if buyer is None or buyer is agent or not buyer.alive \
                or chebyshev(agent.pos, buyer.pos) > 2:
            return
        price = min(buyer.money, self.society.drug_price)
        buyer.money -= price
        agent.money += price
        # The buyer is hooked: the dose hits and addiction climbs.
        self._dose(buyer)
        self.metrics.drug_deals += 1
        f = self.world.facility_at(agent.pos)
        if f is not None:
            f.add_role("drug_den")
        self.world.log("deal_drug", dealer=agent.id, buyer=buyer.id, price=price)

    def _do_take_drug(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.drugs):
            return
        # Self-supply if needed (an addict will cook their own).
        if agent.take("materials", self.society.drug_material_cost) \
                < self.society.drug_material_cost and agent.addiction < 10:
            return
        self._dose(agent)
        f = self.world.facility_at(agent.pos)
        if f is not None:
            f.add_role("drug_den")

    def _dose(self, agent: Agent) -> None:
        """Apply one hit: an energy/pleasure spike, and deeper addiction."""
        c = self.society
        agent.energy = min(MAX_ENERGY, agent.energy + c.drug_energy_spike)
        agent.addiction = min(100.0, agent.addiction + c.addiction_per_dose)
        self._reward(agent, c.drug_pleasure)
        self.metrics.doses_taken += 1

    def _do_join_gang(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.gangs) or agent.gang_id:
            return
        # Join a gang with a member nearby, else found one.
        for g in self.gangs:
            for mid in g.members:
                m = self._by_id.get(mid)
                if m and m.alive and chebyshev(agent.pos, m.pos) <= self.society.gang_join_radius:
                    self._enroll_gang(agent, g)
                    return
        # Found a new gang.
        gid = f"g{self._next_gang_num}"
        self._next_gang_num += 1
        name = GANG_NAMES[(self._next_gang_num - 2) % len(GANG_NAMES)]
        gang = Gang(id=gid, name=name, leader=agent.id, founded_day=self.world.day)
        self.gangs.append(gang)
        self.metrics.gangs_formed += 1
        self._enroll_gang(agent, gang)
        self.world.log("gang_formed", gang=name, leader=agent.id)

    def _enroll_gang(self, agent: Agent, gang: Gang) -> None:
        agent.gang_id = gang.id
        if agent.id not in gang.members:
            gang.members.append(agent.id)
        # Gangs arm their own — joining a crew puts a weapon in your hand.
        if self.society.weapons and agent.weapons == 0:
            agent.weapons += 1
            self.metrics.weapons_crafted += 1
        # Loyalty within, suspicion of rivals.
        for mid in gang.members:
            m = self._by_id.get(mid)
            if m and m.id != agent.id:
                m.adjust_trust(agent.id, self.society.gang_loyalty)
                agent.adjust_trust(m.id, self.society.gang_loyalty)
        # Claim the nearest facility as turf.
        f = self.world.nearest(agent.pos, FacilityType.PLAZA) or \
            self.world.facility_at(agent.pos)
        if f is not None and f.name not in gang.turf:
            f.add_role("gang_turf")
            f.controller = gang.id
            gang.turf.append(f.name)

    def _do_rebel(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.weapons):
            return
        if agent.weapons <= 0:
            return
        c = self.society
        if agent.last_rebelled_day is not None and \
                self.world.day - agent.last_rebelled_day < c.rebellion_cooldown_days:
            return
        if discontent(agent, oppressed=self._is_oppressed(agent)) < c.rebellion_discontent:
            return
        # Muster fellow armed malcontents.
        rebels = [a for a in self.agents if a.alive and a.weapons > 0
                  and discontent(a, oppressed=self._is_oppressed(a)) >= c.rebellion_discontent]
        agent.last_rebelled_day = self.world.day
        agent.rebellions_joined += 1
        if len(rebels) < c.rebellion_min_rebels:
            self.world.log("unrest", instigator=agent.id, rebels=len(rebels))
            return
        # An uprising: the mayor is deposed and the town hall stormed.
        self.metrics.rebellions += 1
        deposed = self.mayor.agent_id if self.mayor else None
        self.mayor = None
        if deposed and deposed in self._by_id:
            target = self._by_id[deposed]
            if target.alive:
                target.energy -= ATTACK_DAMAGE
                self._strike_fear(target, epicentre=target.pos, offender_id=agent.id)
        for r in rebels:
            r.last_rebelled_day = self.world.day
            r.rebellions_joined += 1
        self.world.log("rebellion", instigator=agent.id, rebels=len(rebels),
                       deposed=deposed)

    def _do_preach(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.religion):
            return
        c = self.society
        if agent.faith is None:
            # Found a faith if you have the standing; else you cannot preach yet.
            if agent.reputation < c.faith_min_reputation:
                return
            rid = f"r{self._next_faith_num}"
            self._next_faith_num += 1
            name = FAITH_NAMES[(self._next_faith_num - 2) % len(FAITH_NAMES)]
            religion = Religion(id=rid, name=name, prophet=agent.id,
                                founded_day=self.world.day)
            religion.members.append(agent.id)
            self.religions.append(religion)
            agent.faith = rid
            self.metrics.religions_founded += 1
            # Consecrate the nearest civic site as a temple.
            site = self.world.nearest(agent.pos, FacilityType.PLAZA) or \
                self.world.nearest(agent.pos, FacilityType.LIBRARY)
            if site is not None:
                site.add_role("temple")
            self.world.log("religion_founded", faith=name, prophet=agent.id)
            return
        # Spread the word: convert the unaffiliated nearby.
        religion = self._religion_of(agent.faith)
        if religion is None:
            return
        for o in self.agents:
            if o.alive and o.faith is None and o.id != agent.id \
                    and chebyshev(agent.pos, o.pos) <= c.conversion_radius \
                    and o.trust_of(agent.id) >= 0.2:
                o.faith = religion.id
                religion.members.append(o.id)
                self.metrics.conversions += 1
                o.adjust_trust(agent.id, 0.15)
                self.world.log("conversion", faith=religion.name, convert=o.id)
                break

    def _do_worship(self, agent: Agent, action: Action) -> None:
        if not (self.society.enabled and self.society.religion) or agent.faith is None:
            return
        f = self.world.facility_at(agent.pos)
        if f is None or "temple" not in f.roles:
            return
        c = self.society
        agent.fear = max(0.0, agent.fear - c.worship_fear_relief)
        agent.esteem = max(0.0, agent.esteem - c.worship_esteem_relief)
        self._reward(agent, c.worship_pleasure)
        self.metrics.acts_of_worship += 1
        # Communion binds the faithful who pray together.
        for o in self.agents:
            if o.alive and o.faith == agent.faith and o.id != agent.id \
                    and chebyshev(agent.pos, o.pos) <= 3:
                o.adjust_trust(agent.id, 0.08)
                agent.adjust_trust(o.id, 0.08)

    def _religion_of(self, rid: str):
        for r in self.religions:
            if r.id == rid:
                return r
        return None

    def _is_oppressed(self, agent: Agent) -> bool:
        """Shut out of power: ruled by an oligarchy you're not part of."""
        if self.policy.config.form is GovernanceForm.OLIGARCHY:
            return agent.id not in self._eligible_voters()
        return False

    # -- crime ----------------------------------------------------------
    def _do_steal(self, agent: Agent, action: Action) -> None:
        if self._deterred(agent):
            return
        victim = self._adjacent_or_targeted(agent, action)
        if victim is None:
            return
        self._spend(agent, ActionType.STEAL)
        loot = victim.take("money", 5) if victim.money else 0
        food = victim.take("food", 2)
        agent.money += loot
        agent.add("food", food)
        self._register_crime(agent, "theft", victim)

    def _do_attack(self, agent: Agent, action: Action) -> None:
        if self._deterred(agent):
            return
        victim = self._adjacent_or_targeted(agent, action)
        if victim is None:
            return
        self._spend(agent, ActionType.ATTACK)
        damage = ATTACK_DAMAGE
        if self.society.enabled and self.society.weapons and agent.weapons > 0:
            damage += self.society.weapon_attack_bonus  # armed: far deadlier
        victim.energy -= damage
        agent.money += victim.take("money", 3)
        self._register_crime(agent, "violence", victim)
        if victim.energy <= 0 and victim.alive:
            victim.die(self.world.day, "killed in violence")
            self.metrics.deaths += 1
            self.world.log("death", agent=victim.id, cause="violence")

    def _do_arson(self, agent: Agent, action: Action) -> None:
        if self._deterred(agent):
            return
        name = action.params.get("facility_name")
        f = next((x for x in self.world.facilities if x.name == name), None)
        if f is None:
            f = self.world.facility_at(agent.pos)
        self._spend(agent, ActionType.ARSON)
        agent.crimes_committed += 1
        self.metrics.record_crime("arson")
        self.world.log("arson", offender=agent.id,
                       facility=f.name if f else "unknown",
                       pos=(f.pos if f else agent.pos))
        self._strike_fear(None, epicentre=(f.pos if f else agent.pos),
                          offender_id=agent.id)
        # Burning a granary spills the commons.
        if f and f.ftype == FacilityType.GRANARY:
            self.world.granary_food = max(0, self.world.granary_food - 5)

    def _do_report_crime(self, agent: Agent, action: Action) -> None:
        target = self._by_id.get(action.params.get("target"))
        if target is not None:
            self.world.log("crime_report", reporter=agent.id, accused=target.id)

    # -- crime helpers --------------------------------------------------
    def _adjacent_or_targeted(self, agent: Agent, action: Action) -> Optional[Agent]:
        target = self._by_id.get(action.params.get("target"))
        if target is None or not target.alive:
            return None
        # Must close to within striking range; if far, step toward them instead.
        if chebyshev(agent.pos, target.pos) > 1:
            agent.pos = self.world.step_towards(agent.pos, target.pos)
            if chebyshev(agent.pos, target.pos) > 1:
                return None
        return target

    def _register_crime(self, offender: Agent, kind: str, victim: Agent) -> None:
        offender.crimes_committed += 1
        victim.times_victimized += 1
        self.metrics.record_crime(kind)
        victim.adjust_trust(offender.id, -0.6)
        victim.remember(f"Day {self.world.day}: {offender.name} committed {kind} against me.")
        self.world.log(kind, offender=offender.id, victim=victim.id, pos=victim.pos)
        self._strike_fear(victim, epicentre=victim.pos, offender_id=offender.id)
        # If a punishment law is active and the victim is near a police station,
        # the offender is fined immediately.
        if self.policy.has_punishment_law():
            nearest_police = self.world.nearest(victim.pos, FacilityType.POLICE_STATION)
            if nearest_police and chebyshev(victim.pos, nearest_police.pos) <= self.policy.config.police_range:
                fine = min(offender.money, self.policy.config.fine_amount)
                offender.money -= fine
                victim.money += fine // 2
                self.world.granary_food += 1
                self.world.log("fine", offender=offender.id, amount=fine)
                self.metrics.fines_collected += 1

    def _strike_fear(self, victim: Optional[Agent], epicentre: tuple[int, int],
                     offender_id: str) -> None:
        """A crime radiates dread: the victim is shaken hard, witnesses less so."""
        if not self.psyche.enabled:
            return
        if victim is not None:
            victim.fear = min(100.0, victim.fear + self.psyche.fear_per_victimization)
        for o in self.agents:
            if not o.alive or o.id == offender_id or (victim and o.id == victim.id):
                continue
            if chebyshev(o.pos, epicentre) <= self.psyche.witness_radius:
                o.fear = min(100.0, o.fear + self.psyche.fear_per_witness)
        self.metrics.peak_fear = max(
            self.metrics.peak_fear,
            max((a.fear for a in self.agents if a.alive), default=0.0),
        )

    # ==================================================================
    # Upkeep, day boundaries, finalisation
    # ==================================================================
    def _tick_upkeep(self, agent: Agent) -> None:
        decay = ENERGY_DECAY_PER_TICK
        if self.environment is not None:
            decay *= self.environment.energy_multiplier()  # cold seasons drain more
        agent.energy -= decay
        if self.drives.enabled:
            self._drive_upkeep(agent)
        if self.status.enabled:
            # The need for recognition quietly builds, like the primal urges.
            agent.esteem = min(100.0, agent.esteem + self.status.esteem_per_tick)
        if self.psyche.enabled and agent.fear > 0:
            # Quiet time heals fear — faster in the shadow of safety.
            decay = self.psyche.fear_decay_per_tick
            if self._near_safety(agent):
                decay += self.psyche.fear_decay_safe_bonus
            agent.fear = max(0.0, agent.fear - decay)
            # Chronic terror is stress; it eats at the body.
            if agent.fear > self.psyche.fear_threshold:
                agent.energy -= self.psyche.fear_energy_penalty
        if self.society.enabled and self.society.drugs and agent.addiction > 0:
            agent.addiction = max(0.0, agent.addiction - self.society.addiction_decay_per_tick)
            # Withdrawal: the craving sickness drains the body.
            if agent.addiction > self.society.withdrawal_threshold:
                agent.energy -= self.society.withdrawal_energy_penalty
        if agent.energy <= 0 and agent.alive:
            cause = "starvation"
            if self.drives.enabled and agent.fatigue >= 100.0:
                cause = "exhaustion"
            if self.society.enabled and agent.addiction > self.society.withdrawal_threshold:
                cause = "overdose/withdrawal"
            agent.die(self.world.day, cause)
            self.metrics.deaths += 1
            self.world.log("death", agent=agent.id, cause=cause)

    def _drive_upkeep(self, agent: Agent) -> None:
        """Raise hunger and fatigue; let unmet drives erode energy; and let
        neighbours grow familiar (so pair bonds can form for reproduction)."""
        d = self.drives
        agent.hunger = min(100.0, agent.hunger + d.hunger_per_tick)
        agent.fatigue = min(100.0, agent.fatigue + d.fatigue_per_tick)
        if d.reproduction:
            agent.libido = min(100.0, agent.libido + d.libido_per_tick)
        if agent.hunger > d.hunger_threshold:
            agent.energy -= d.hunger_energy_penalty
        if agent.fatigue > d.fatigue_threshold:
            agent.energy -= d.fatigue_energy_penalty
        # Familiarity: spending time near someone slowly builds mild trust.
        # Applied per-agent each tick, so it becomes mutual over the tick.
        for o in self.agents:
            if o.id != agent.id and o.alive and chebyshev(agent.pos, o.pos) <= 2:
                # Don't whitewash real grievances: only nudge non-negative ties.
                if agent.trust_of(o.id) >= -0.05:
                    agent.adjust_trust(o.id, 0.03)

    def _spawn_child(self, parent_a: Agent, parent_b: Agent) -> None:
        d = self.drives
        num = self._next_agent_num
        self._next_agent_num += 1
        # The child inherits one parent's persona (chosen by coin flip).
        persona_key = self.rng.choice([parent_a.persona, parent_b.persona])
        child = Agent(
            id=f"a{num}",
            name=f"{parent_a.name.split('-')[0]}{num}",
            profession="child",
            persona=persona_key,
            x=parent_a.x,
            y=parent_a.y,
            energy=d.child_energy,
            money=5,
            age_days=0,
            parent_ids=(parent_a.id, parent_b.id),
        )
        child.inventory = {"food": 2, "materials": 0}
        # The newborn trusts and is trusted by its parents.
        for p in (parent_a, parent_b):
            child.adjust_trust(p.id, 0.5)
            p.adjust_trust(child.id, 0.5)
            p.energy -= d.repro_energy_cost
            p.last_reproduced_day = self.world.day
            p.children += 1

        self.agents.append(child)
        self._by_id[child.id] = child
        self.brains[child.id] = self._make_newborn_brain(child, persona_key)
        if self.memory is not None:
            self.memory.register(child)
        self.metrics.births += 1
        self.world.log("birth", child=child.id, parents=f"{parent_a.id}+{parent_b.id}",
                       persona=persona_key)

    def _make_newborn_brain(self, child: Agent, persona_key: str) -> AgentBrain:
        if self.newborn_brain_factory is not None:
            return self.newborn_brain_factory(child, persona_key, self.rng)
        # Default: a persona-tuned heuristic with its own derived RNG.
        from .brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona_key, random.Random(self.rng.randint(0, 2**31)))

    def _record_frame(self) -> None:
        """Capture a lightweight snapshot of the day for the playback view."""
        day = self.world.day
        env = self.environment.snapshot() if self.environment is not None else {}
        crimes = [list(e["pos"]) for e in self.world.events
                  if e.get("day") == day and e.get("kind") in ("theft", "violence", "arson")
                  and isinstance(e.get("pos"), (tuple, list))]
        self.frames.append({
            "day": day,
            "season": env.get("season", ""),
            "weather": env.get("weather", ""),
            "alive": self._living(),
            "agents": [{"id": a.id, "name": a.name, "x": a.x, "y": a.y,
                        "persona": a.persona, "alive": a.alive,
                        "gang": a.gang_id, "faith": a.faith}
                       for a in self.agents],
            "crimes": crimes,
        })

    def _end_of_day(self, verbose: bool) -> None:
        self._record_frame()
        self._apply_daily_policy()
        self._maybe_elect_mayor()
        if self.environment is not None:
            # New weather/season, regen resources, reprice, maybe a disaster.
            self.environment.advance_day(self)
        if self.memory is not None:
            # Advance the in-game clock one day and run each agent's forgetting pass.
            self.memory.tick()
        if self.drives.enabled:
            for a in self.agents:
                if a.alive:
                    a.age_days += 1
        if self.status.enabled:
            # Honour is not permanent: prestige fades without fresh deeds.
            for a in self.agents:
                if a.alive:
                    a.reputation = max(0.0, a.reputation - self.status.rep_decay_per_day)
        total, passed, rejected = self.legislature.counts()
        summary = {
            "day": self.world.day,
            "alive": self._living(),
            "crimes_total": self.metrics.crimes_total,
            "proposals": total,
            "passed": passed,
            "rejected": rejected,
            "frauds": self.metrics.frauds,
            "fines": self.metrics.fines_collected,
            "births": self.metrics.births,
            "granary_food": self.world.granary_food,
            "active_laws": len(self.policy.laws),
            "mayor": self.mayor.agent_id if self.mayor else None,
        }
        self.daily_log.append(summary)
        if self.on_event:
            self.on_event({"kind": "day_summary", **summary})
        if verbose:
            gov_tag = self.policy.config.form.value[:4]
            mayor_tag = f" mayor={self.mayor.agent_id}" if self.mayor else ""
            births_tag = f" births={summary['births']}" if self.drives.reproduction else ""
            print(
                f"Day {summary['day']:>2}: alive={summary['alive']:>2} "
                f"crimes={summary['crimes_total']:>3} "
                f"proposals={passed}/{total} passed "
                f"frauds={summary['frauds']} fines={summary['fines']} "
                f"laws={summary['active_laws']} gov={gov_tag}{mayor_tag}{births_tag}"
            )

    def _apply_daily_policy(self) -> None:
        """Run once per day: tax, food redistribution."""
        living = [a for a in self.agents if a.alive]
        if not living:
            return
        if self.policy.has_tax():
            # Take a fraction from the richest agents; give to the commons.
            living_sorted = sorted(living, key=lambda a: a.money, reverse=True)
            top = living_sorted[: max(1, len(living_sorted) // 3)]
            for a in top:
                tribute = int(a.money * self.policy.config.tax_rate)
                a.money -= tribute
                self.world.granary_food += tribute // 2  # converts to shared food
            self.metrics.tax_days += 1
        if self.policy.has_food_redistribution():
            # Top-up the granary with a small daily grant and notify agents.
            grant = max(2, len(living))
            self.world.granary_food += grant

    def _maybe_elect_mayor(self) -> None:
        interval = self.policy.config.election_interval
        if self.world.day % interval != 0:
            return
        if self.policy.config.form is GovernanceForm.ANARCHY:
            return
        living = [a for a in self.agents if a.alive]
        if not living:
            return
        # Power flows to the engaged — and, once honour matters, to the esteemed.
        if self.status.enabled:
            winner = max(living, key=lambda a: (a.votes_cast + a.reputation))
        else:
            winner = max(living, key=lambda a: a.votes_cast)
        self.mayor = Mayor(
            agent_id=winner.id,
            elected_day=self.world.day,
            term_ends_day=self.world.day + interval,
        )
        self.world.log("election", mayor=winner.id, day=self.world.day)
        self.metrics.elections += 1
        if self.status.enabled:
            # Taking office is the height of recognition (権力).
            winner.times_mayor += 1
            self._recognise(winner, self.status.rep_per_mayor,
                            self.status.mayor_relief, "mayor")

    def _finalize(self, last_day: int) -> None:
        self.metrics.days_run = last_day
        self.metrics.survivors = self._living()
        total, passed, rejected = self.legislature.counts()
        self.metrics.proposals_total = total
        self.metrics.proposals_passed = passed
        self.metrics.proposals_rejected = rejected
        self.metrics.laws_enacted = len(self.policy.laws)
        self.metrics.gov_form = self.policy.config.form.value
        if self.society.enabled:
            thr = self.society.withdrawal_threshold
            self.metrics.addicts = sum(1 for a in self.agents
                                       if a.alive and a.addiction > thr)
        if self.environment is not None:
            s = self.environment.summary()
            self.metrics.disasters_total = s["disasters_total"]
            self.metrics.peak_food_price = s["peak_food_price"]
            self.metrics.final_season = s["final_season"]

    def _living(self) -> int:
        return sum(1 for a in self.agents if a.alive)

    def _eligible_voters(self) -> set[str]:
        """Returns the set of agent IDs allowed to vote/propose.
        Empty set means everyone is eligible (direct/constitutional/anarchy)."""
        cfg = self.policy.config
        if cfg.form is GovernanceForm.OLIGARCHY:
            living = [a for a in self.agents if a.alive]
            top = sorted(living, key=lambda a: a.money, reverse=True)
            return {a.id for a in top[: cfg.oligarch_count]}
        return set()

    def _deterred(self, agent: Agent) -> bool:
        """Return True if the agent should abort a crime due to police presence."""
        nearest_police = self.world.nearest(agent.pos, FacilityType.POLICE_STATION)
        if nearest_police is None:
            return False
        dist = chebyshev(agent.pos, nearest_police.pos)
        cfg = self.policy.config
        if dist > cfg.police_range:
            return False
        mult = self.policy.crime_deterrence_multiplier()
        # Closer to the station → stronger deterrence.
        proximity_factor = 1.0 - (dist / (cfg.police_range + 1))
        deterrence_prob = (1.0 - mult) * proximity_factor
        return self.rng.random() < deterrence_prob

    def _near_safety(self, agent: Agent) -> bool:
        """Within comforting reach of a police station or a house."""
        for ftype in (FacilityType.POLICE_STATION, FacilityType.HOUSE):
            f = self.world.nearest(agent.pos, ftype)
            if f and chebyshev(agent.pos, f.pos) <= self.psyche.safe_radius:
                return True
        return False

    # ==================================================================
    @staticmethod
    def _event_str(e: dict) -> str:
        kind = e.get("kind", "event")
        d, t = e.get("day"), e.get("tick")
        extra = {k: v for k, v in e.items() if k not in {"kind", "day", "tick"}}
        body = ", ".join(f"{k}={v}" for k, v in extra.items())
        return f"D{d}T{t} {kind}: {body}" if body else f"D{d}T{t} {kind}"
