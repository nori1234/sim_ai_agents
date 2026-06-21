"""The engine: schedules turns, applies actions, and runs the town for N days."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .actions import Action, ActionType, Event
from .agent import Agent, MAX_ENERGY
from .brains.base import AgentBrain
from .drives import DrivesConfig, can_reproduce, is_fertile, mating_urge
from .affordances import affordances_at, gather_multiplier, gather_specialty, role_of
from .economy import Ledger, LedgerEntry, apply_transfer, is_fraudulent_solicitation
from .esteem import StatusConfig, esteem_urge
from .psyche import PsycheConfig, actualization_pull, fear_level
from .society import Gang, Religion, SocietyConfig, discontent
from .society import GANG_NAMES, FAITH_NAMES
from . import publicworks as PW
from . import development as DEV
from . import market as MK
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
# Principled fiscality (economy layer): how much money the state grants each poor
# citizen per welfare day, drawn from the treasury it actually holds.
FISCAL_WELFARE = 3
REST_ENERGY = 8.0
SHELTER_REST_ENERGY = 16.0  # at a house or hospital
ACTION_ENERGY_COST = {
    ActionType.GATHER: 3.0,
    ActionType.WORK: 3.0,
    ActionType.BUILD: 4.0,
    ActionType.ATTACK: 5.0,
    ActionType.STEAL: 3.0,
    ActionType.ARSON: 4.0,
    ActionType.ARREST: 4.0,
    # Raw primitives (used directly only by the LLM brain; the heuristic emits
    # macros that spend their own cost, so the baseline is untouched).
    ActionType.TAKE: 2.0,
    ActionType.GIVE: 1.0,
    ActionType.STRIKE: 5.0,
    ActionType.MAKE: 3.0,
    ActionType.SAY: 1.0,
    ActionType.BOND: 1.0,
}
ARREST_WINDOW_DAYS = 2     # how recently a crime must have happened to be arrestable
ARREST_ENERGY_PENALTY = 6.0  # the scuffle/detainment costs the offender energy
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
    # Optional town library (library.TownLibrary): books that outlive their
    # authors — horizontal/cultural inheritance. Opt-in; None leaves the
    # baseline byte-identical (the heuristic brain ignores the knowledge view).
    library: object = None
    # Public-works civic loop (council-funded construction); opt-in.
    public_works: bool = False
    treasury: int = 0
    # Historical development: gate construction on plausible prerequisites; opt-in.
    development: bool = False
    # Economic physics (offer/accept/craft primitives); opt-in.
    economy: bool = False
    offers: list = field(default_factory=list)        # open Offer order book
    loans: list = field(default_factory=list)         # outstanding credit (Loan)
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
        self._next_offer_id = 1
        self._next_loan_id = 1
        # Emergent prices: recent settled swap ratios per ordered (give, want) pair.
        self._trade_ratios: dict[tuple, list] = {}

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

    def step_day(self, verbose: bool = False) -> bool:
        """Advance the simulation by a single day, for incremental/streamed
        running (the API observatory). Equivalent, day-for-day, to ``run()`` --
        same tick order, same RNG usage, same finalisation -- so stepping is
        byte-identical to a full run. Returns True while the world is still
        running, False once it has finished (extinct or out of days)."""
        if getattr(self, "_finished", False):
            return False
        day = getattr(self, "_step_day", 0) + 1
        self._step_day = day
        self.world.day = day
        for tick in range(1, self.config.ticks_per_day + 1):
            self.world.tick = tick
            self._run_tick()
        self._end_of_day(verbose=verbose)
        if self._living() == 0 or day >= self.config.days:
            self._finalize(day)
            self._finished = True
        return not getattr(self, "_finished", False)

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
            if self.library is not None:
                self._library_study(agent)
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
                # A passed public-works proposal commissions construction.
                if self.public_works and p.build:
                    self._build_public_work(p.build)

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
        apos = agent.pos
        for o in self.agents:
            if o.id == agent.id or not o.alive:
                continue
            # A lean per-other view: only the fields a brain reads, with the
            # SAME rounding as snapshot() for the ones compared to thresholds
            # (hunger/fatigue/reputation) so decisions stay byte-identical.
            # (self_view below keeps the full snapshot; this is the O(N^2) path.)
            others.append({
                "id": o.id, "name": o.name, "profession": o.profession,
                "distance": chebyshev(apos, o.pos),
                "trust": round(agent.trust_of(o.id), 2),
                "money": o.money, "food": o.food(), "materials": o.materials(),
                "hunger": round(o.hunger, 1), "fatigue": round(o.fatigue, 1),
                "age_days": o.age_days, "reputation": round(o.reputation, 1),
                "last_crime_day": o.last_crime_day,
            })
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
        # By a library, surface predecessors' recorded lessons (cultural
        # inheritance). Ignored by the heuristic brain, so outcomes are unchanged.
        knowledge_view: list = []
        if self.library is not None and (
            (here is not None and here["type"] == "library")
            or any(f["type"] == "library" and f["distance"] <= 3 for f in nearby)
        ):
            knowledge_view = self.library.read(
                self._memory_query(agent, here, others), k=3)
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
            knowledge=knowledge_view,
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
            norms=({"crime": True,
                    "enforcement": round(self._enforcement_expectation(), 2)}
                   if self.policy.has_crime_norm() else {}),
            laws=self._published_laws(),
            environment=self.environment.snapshot() if self.environment is not None else {},
            role=role_of(agent.profession),
            affordances=affordances_at(here_f),
            public_works=({"enabled": True, "treasury": self.treasury,
                           "cost": PW.PUBLIC_WORKS_COST,
                           "buildable": sorted(set(PW.BUILDABLE)),
                           "suggest": (DEV.next_public_work(self)
                                       if self.development else None)}
                          if self.public_works else {}),
            open_offers=([o.as_dict() for o in self.offers[:8]]
                         if self.economy else []),
            economy=({"enabled": True, "tradable": list(MK.TRADABLE),
                      "recipes": {k: v[0] for k, v in MK.RECIPES.items()},
                      "price_food_in_money": self.emergent_price("food", "money"),
                      # A capability hint (what you produce well), not a valuation
                      # — the agent judges worth itself, weighing it against price.
                      "your_specialty": gather_specialty(agent.profession)}
                     if self.economy else {}),
            debts=([l.as_dict() for l in self.loans
                    if l.debtor == agent.id and not l.settled and not l.defaulted]
                   if self.economy else []),
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
        ActionType.ARREST,
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

    def _library_study(self, agent: Agent) -> None:
        """While standing in a library, an agent both records and *studies*:
        it writes a lesson from its experience to the town shelf (a book outlives
        its author), and internalises one predecessor's lesson it doesn't yet
        carry into its own evolving memory. Purely additive: it touches no
        agent/world state a decision reads, so on/off runs stay identical."""
        f = self.world.facility_at(agent.pos)
        if f is None or f.ftype is not FacilityType.LIBRARY:
            return
        # Publish a lesson from lived experience — not something merely re-read,
        # so the shelf keeps accumulating first-hand knowledge.
        firsthand = [m for m in agent.memory
                     if not m.startswith("I read in the library:")]
        lesson = (firsthand[-1] if firsthand
                  else f"A {agent.profession}'s craft sustains the town.")
        self.library.write(self.world.day, agent.id, agent.name, lesson)
        # Study: internalise one relevant book the agent hasn't read yet (compare
        # the stored form so a book is taken in once, not re-read every visit).
        held = set(agent.memory)
        for book in self.library.read(self._memory_query(agent, None, []), k=5):
            entry = f"I read in the library: {book}"
            if entry not in held:
                agent.remember(entry)
                break

    def _spend(self, agent: Agent, action_type: ActionType) -> None:
        agent.energy -= ACTION_ENERGY_COST.get(action_type, 0.0)

    # ==================================================================
    # Physical primitives + interpretation
    # ------------------------------------------------------------------
    # The only code that moves goods between holders. Macros (steal, transfer,
    # ...) lower to this; the LLM brain may call the raw take/give verbs. The
    # helper performs the physics and returns an Event; _interpret reads the
    # Event + context to decide what institution it is (theft, gift, ...).
    # Energy is NOT spent here — the entry handler owns that, so each verb keeps
    # its own cost and the heuristic baseline is unchanged.
    # ==================================================================
    def _move_items(self, actor: Agent, other: Optional[Agent],
                    items: dict, *, kind: str,
                    consent: Optional[bool]) -> Event:
        src, dst = (other, actor) if kind == "take" else (actor, other)
        moved: dict[str, int] = {}
        if src is not None and dst is not None:
            for resource, qty in items.items():
                got = src.take(resource, int(qty))
                dst.add(resource, got)
                if got:
                    moved[resource] = got
        ev = Event(kind=kind, actor=actor, other=other, items=moved, consent=consent)
        self._interpret(ev)
        return ev

    def _interpret(self, ev: Event) -> None:
        """Read an act + its context as an institution. New institutions are new
        branches here, not new verbs."""
        if ev.kind == "take" and ev.other is not None and ev.consent is False:
            # A non-consensual take from a person is theft.
            self._register_crime(ev.actor, "theft", ev.other)
        elif ev.kind == "give" and ev.other is not None and ev.consent is True \
                and ev.items:
            # A consensual handover to a person is a gift/transfer.
            self.metrics.transfers += 1
            for resource, qty in ev.items.items():
                self.ledger.record(LedgerEntry(
                    self.world.day, self.world.tick,
                    ev.actor.id, ev.other.id, resource, qty))
                ev.other.adjust_trust(ev.actor.id, +0.15)
                ev.other.remember(
                    f"Day {self.world.day}: {ev.actor.name} gave me {qty} {resource}.")
                self.world.log("transfer", sender=ev.actor.id, receiver=ev.other.id,
                               resource=resource, amount=qty)
        elif ev.kind == "strike" and ev.other is not None:
            # Force against a person is the crime of violence.
            self._register_crime(ev.actor, "violence", ev.other)
        elif ev.kind == "strike" and ev.other is None:
            # Force against a structure is arson (no victim, so the crime is
            # accounted inline and dread radiates from the site).
            f = ev.site
            ev.actor.crimes_committed += 1
            ev.actor.last_crime_day = self.world.day
            self.metrics.record_crime("arson")
            self.world.log("arson", offender=ev.actor.id,
                           facility=f.name if f else "unknown",
                           pos=(f.pos if f else ev.actor.pos))
            self._strike_fear(None, epicentre=(f.pos if f else ev.actor.pos),
                              offender_id=ev.actor.id)
        elif ev.kind == "say" and ev.intent == "praise" and ev.other is not None:
            # A signal of public commendation grants the recipient esteem
            # relief, honour, pleasure, and warms the bond both ways.
            a, t = ev.actor, ev.other
            a.praise_given += 1
            t.praise_received += 1
            self.metrics.total_praise += 1
            t.esteem = max(0.0, t.esteem - self.status.praise_relief)
            t.reputation += self.status.rep_per_praise
            self._reward(t, self.status.pleasure_per_praise)
            t.adjust_trust(a.id, +0.1)
            a.adjust_trust(t.id, +0.05)
            self.world.log("praise", by=a.id, of=t.id)
        elif ev.kind == "say" and ev.intent == "sermon":
            # A sermon founds or spreads a faith.
            self._preach_faith(ev.actor)
        elif ev.kind == "say" and ev.intent == "proposal":
            # A proposal put to the town: the legislature takes it up.
            self._make_proposal(ev.actor, ev.payload or {})
        elif ev.kind == "bond" and ev.intent == "worship":
            # An act of worship: relief, peace, and communion among the faithful.
            self._worship_effect(ev.actor)
        elif ev.kind == "bond" and ev.intent == "gang":
            # A bond of allegiance to a crew: join the nearest, else found one.
            self._join_or_found_gang(ev.actor)

    def _do_take(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): pull items from another agent. Consent defaults
        to False — taking without an agreement is theft and is interpreted so."""
        # Resolve "from" through the same adjacency/pursuit logic as steal.
        other = self._adjacent_or_targeted(
            agent, Action(action.type, {"target": action.params.get("from")}))
        if other is None:
            return
        items = action.params.get("items") or {}
        if not items:
            return
        self._spend(agent, ActionType.TAKE)
        self._move_items(agent, other, items, kind="take",
                         consent=bool(action.params.get("consent", False)))

    def _do_give(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): push items to another agent. Consent defaults to
        True — a voluntary handover is a gift and is interpreted so."""
        other = self._by_id.get(action.params.get("to") or action.params.get("target"))
        if other is None or not other.alive:
            return
        items = action.params.get("items") or {}
        if not items:
            return
        self._spend(agent, ActionType.GIVE)
        self._move_items(agent, other, items, kind="give",
                         consent=bool(action.params.get("consent", True)))

    def _use_item(self, agent: Agent, item: str, qty: int,
                  target: Optional[Agent] = None) -> Event:
        """Consume a held item and apply its effect to self (or a target). The
        metabolic effect (food -> energy) is item physics, not an institution,
        so it lives here; _interpret is still called for any institutional
        reading of the act."""
        user = target or agent
        used = agent.take(item, qty)
        if item == "food":
            user.energy = min(MAX_ENERGY, user.energy + used * EAT_ENERGY_PER_FOOD)
            if self.drives.enabled and used:
                # Pleasure scales with how hungry you were — relief feels good.
                relief = used * self.drives.eat_hunger_relief
                self._reward(user, self.drives.pleasure_per_eat * (user.hunger / 100.0))
                user.hunger = max(0.0, user.hunger - relief)
        elif item == "drug":
            # A narcotic is cooked on the spot (materials spent by the caller),
            # so there is nothing in inventory to consume — the dose is the act.
            self._dose(user)
        ev = Event(kind="use", actor=agent, other=target,
                   items={item: used} if used else {}, consent=None)
        self._interpret(ev)
        return ev

    def _do_use(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): apply a held item to self or a named target."""
        item = action.params.get("item")
        if not item:
            return
        qty = int(action.params.get("qty", 1))
        target = self._by_id.get(action.params.get("on"))
        self._spend(agent, ActionType.USE)
        self._use_item(agent, item, qty, target=target)

    def _strike(self, agent: Agent, *, victim: Optional[Agent] = None,
                facility=None) -> Event:
        """Apply force to a target. Damaging a person robs and harms them;
        damaging a structure may spill its contents. The interpretation layer
        reads the act as violence (vs an agent) or arson (vs a structure)."""
        if victim is not None:
            damage = ATTACK_DAMAGE
            if self.society.enabled and self.society.weapons and agent.weapons > 0:
                damage += self.society.weapon_attack_bonus  # armed: far deadlier
            victim.energy -= damage
            agent.money += victim.take("money", 3)  # violence robs coin too
        ev = Event(kind="strike", actor=agent, other=victim, site=facility)
        self._interpret(ev)
        # Burning a granary spills the commons (structural after-effect).
        if victim is None and facility is not None \
                and facility.ftype == FacilityType.GRANARY:
            self.world.granary_food = max(0, self.world.granary_food - 5)
        # Burning a library destroys the public record — but not what people have
        # already learned (that lives on in their own memory). Knowledge persists
        # only while its substrate survives.
        if victim is None and facility is not None and self.library is not None \
                and facility.ftype == FacilityType.LIBRARY and len(self.library):
            lost = self.library.burn()
            self.world.log("library_burned", offender=agent.id,
                           facility=facility.name, books_lost=lost, pos=facility.pos)
        if victim is not None and victim.energy <= 0 and victim.alive:
            victim.die(self.world.day, "killed in violence")
            self.metrics.deaths += 1
            self.world.log("death", agent=victim.id, cause="violence")
        return ev

    def _do_strike(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): strike a named agent or a named facility."""
        if action.params.get("target"):
            victim = self._adjacent_or_targeted(agent, action)
            if victim is None:
                return
            self._spend(agent, ActionType.STRIKE)
            self._strike(agent, victim=victim)
            return
        name = action.params.get("facility_name")
        f = next((x for x in self.world.facilities if x.name == name), None)
        if f is None:
            f = self.world.facility_at(agent.pos)
        self._spend(agent, ActionType.STRIKE)
        self._strike(agent, facility=f)

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
        # A macro: gathering is a take from the world — the counterparty is a
        # resource node, not a holder, so the yield is produced rather than
        # drained. The gate + spend stay here; the produce-and-take is _harvest.
        f = self.world.facility_at(agent.pos)
        if f is None or not f.can_gather():
            return
        self._harvest(agent, f)
        self._spend(agent, ActionType.GATHER)

    def _harvest(self, agent: Agent, f) -> Event:
        """Take from a world resource node: the node *produces* a yield (shaped
        by the environment), which flows into the gatherer. No counterparty, so
        nothing is interpreted as an institution."""
        resource, amount = f.gather_yield()  # type: ignore[misc]
        # Economy layer: production is specialised. A specialist gathers its good
        # well; off-specialty self-supply is inefficient (a low-yield fallback,
        # never zero) — which is what gives food/materials a real demand. Gated by
        # the economy flag, so the offline baseline is byte-identical.
        if self.economy and amount:
            amount = max(1, round(amount * gather_multiplier(agent.profession, resource)))
        if self.environment is not None:
            amount = self.environment.gather(f, resource, amount)
        agent.add(resource, amount)
        ev = Event(kind="take", actor=agent, other=None, site=f,
                   items={resource: amount} if amount else {})
        self._interpret(ev)
        return ev

    def _do_eat(self, agent: Agent, action: Action) -> None:
        # A macro: eating is using food on oneself (food -> energy + relief).
        self._use_item(agent, "food", EAT_FOOD_USED)

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
        # A macro: a transfer is a consensual give to another agent. The act
        # moves the item; the interpretation layer reads "consensual give to a
        # person" as a gift (trust, ledger, the transfers tally — see _interpret).
        target = self._by_id.get(action.params.get("target"))
        if target is None or not target.alive:
            return
        resource = action.params.get("resource", "food")
        amount = int(action.params.get("amount", 1))
        self._move_items(agent, target, {resource: amount},
                         kind="give", consent=True)

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
        # A macro: proposing is a say carrying a proposal; putting it to the
        # legislature is read off the act by _interpret. The proposal's text and
        # any inferred public-works build are bundled as the say's payload.
        text = str(action.params.get("text", "Untitled proposal")).strip()
        # A public-works proposal names a facility to build (explicit param, or
        # inferred from the text). Only meaningful when the loop is enabled.
        build = None
        if self.public_works:
            raw = action.params.get("build")
            ft = PW.parse_build(str(raw)) if raw else PW.parse_build(text)
            build = ft.value if ft is not None else None
        self._interpret(Event(kind="say", actor=agent, intent="proposal",
                              payload={"text": text, "build": build}))

    def _make_proposal(self, agent: Agent, payload: dict) -> None:
        """Put a proposal to the legislature, and have the author vote yes."""
        text = payload.get("text", "Untitled proposal")
        build = payload.get("build")
        eligible = self._eligible_voters()
        p = self.legislature.propose(agent.id, text, self.world.day,
                                     eligible_ids=eligible or None, build=build)
        if p is None:
            return
        agent.proposals_made += 1
        self.metrics.proposals_total += 1
        self.legislature.cast_vote(p.id, agent.id, True, eligible_ids=eligible or None)
        self.world.log("proposal", id=p.id, author=agent.id, text=text)

    def _do_vote(self, agent: Agent, action: Action) -> None:
        # A macro: a vote is committing assent (a bond) to a proposal.
        pid = action.params.get("proposal_id")
        if pid is None:
            return
        self._bond_to_proposal(agent, int(pid),
                               bool(action.params.get("support", True)))

    def _bond_to_proposal(self, agent: Agent, pid: int, support: bool) -> None:
        """The commitment physics behind a vote: assent (or dissent) to a
        collective decision, if the agent is eligible to bind itself to it."""
        eligible = self._eligible_voters()
        if self.legislature.cast_vote(pid, agent.id, support,
                                      eligible_ids=eligible or None):
            agent.votes_cast += 1

    def _do_bond(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): commit to an agreement. To a proposal it is a
        vote; with another agent it is a pact expressed as mutual allegiance."""
        if action.params.get("proposal_id") is not None:
            self._spend(agent, ActionType.BOND)
            self._bond_to_proposal(agent, int(action.params["proposal_id"]),
                                   bool(action.params.get("support", True)))
            return
        other = self._by_id.get(action.params.get("with")
                                or action.params.get("target"))
        if other is None or other is agent or not other.alive:
            return
        self._spend(agent, ActionType.BOND)
        # A pact: allegiance expressed as mutual trust between the two.
        agent.adjust_trust(other.id, +0.2)
        other.adjust_trust(agent.id, +0.2)
        self.world.log("bond", a=agent.id, b=other.id)

    # -- construction & collaboration -----------------------------------
    def _do_build(self, agent: Agent, action: Action) -> None:
        # A macro: building is making a structure. The materials cost + spend
        # stay here; raising/joining the facility is the _make_structure physics.
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
        self._make_structure(agent, name, ftype)

    def _make_structure(self, agent: Agent, name: str, ftype) -> Event:
        """Raise a new facility (or join an existing build). A monument is a
        conspicuous achievement that earns honour."""
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
        ev = Event(kind="make", actor=agent, site=existing, items={ftype.value: 1})
        self._interpret(ev)
        return ev

    def _build_public_work(self, facility_value: str) -> None:
        """A passed public-works proposal: the state funds it and a builder
        erects it. Deterrent facilities (police/prison) go where crime clusters."""
        try:
            ftype = FacilityType(facility_value)
        except ValueError:
            return
        # Historical gate: you can't leapfrog the developmental sequence.
        if self.development and not DEV.can_build(ftype, self.world):
            self.world.log("public_works_premature", facility=ftype.value)
            return
        # Don't fund a second of a one-per-town institution.
        if ftype in PW.UNIQUE_FACILITIES and self.world.facilities_of(ftype):
            self.world.log("public_works_redundant", facility=ftype.value)
            return
        cost = PW.PUBLIC_WORKS_COST
        if self.treasury < cost:
            self.world.log("public_works_unfunded", facility=ftype.value,
                           treasury=self.treasury)
            return
        self.treasury -= cost
        # Place deterrents at the trouble spot; everything else near the centre.
        if ftype in PW.DETERRENT_FACILITIES:
            pos = self._crime_centroid() or (self.world.width // 2, self.world.height // 2)
        else:
            pos = (self.world.width // 2, self.world.height // 2)
        x, y = self._free_cell(pos)
        # A builder does the work if the town has one alive, else any citizen.
        builders = [a for a in self.agents if a.alive and a.profession == "builder"]
        crew = (builders or [a for a in self.agents if a.alive])
        builder = crew[0] if crew else None
        n = sum(1 for f in self.world.facilities if f.ftype == ftype) + 1
        fac = self.world.add_facility(Facility(
            name=f"{ftype.value.replace('_', ' ').title()} {n}", ftype=ftype,
            x=x, y=y, built_on_day=self.world.day,
            builders=[builder.id] if builder else []))
        self.metrics.public_works_built += 1
        if builder is not None:
            builder.collaborations += 1
        self.world.log("public_works", facility=fac.name, type=ftype.value,
                       by=(builder.id if builder else None), cost=cost)

    def _crime_centroid(self):
        pts = [e["pos"] for e in self.world.events
               if e.get("kind") in ("theft", "violence", "arson")
               and isinstance(e.get("pos"), (tuple, list))]
        if not pts:
            return None
        return (round(sum(p[0] for p in pts) / len(pts)),
                round(sum(p[1] for p in pts) / len(pts)))

    def _free_cell(self, pos):
        """A grid cell at/near pos not already occupied by a facility."""
        for radius in range(0, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    c = (max(0, min(self.world.width - 1, pos[0] + dx)),
                         max(0, min(self.world.height - 1, pos[1] + dy)))
                    if self.world.facility_at(c) is None:
                        return c
        return pos

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

    def _say(self, agent: Agent, *, content: str = "",
             to: Optional[Agent] = None, kind: str = "speech") -> Event:
        """Broadcast a signal. A plain statement is logged as speech; a signal
        aimed at an offender is an accusation. Meaning beyond the log (a
        proposal, a sermon) is left to richer interpretation later."""
        if kind == "accusation":
            if to is not None:
                self.world.log("crime_report", reporter=agent.id, accused=to.id)
        elif kind in ("praise", "sermon"):
            pass  # the effects + log are handled in _interpret
        else:
            self.world.log("speech", agent=agent.id, text=content)
        ev = Event(kind="say", actor=agent, other=to, intent=kind)
        self._interpret(ev)
        return ev

    def _do_speak(self, agent: Agent, action: Action) -> None:
        # A macro: a speech is just a public say.
        self._say(agent, content=str(action.params.get("text", "")))

    def _do_say(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): broadcast a statement, optionally at a target."""
        to = self._by_id.get(action.params.get("to") or action.params.get("target"))
        self._spend(agent, ActionType.SAY)
        self._say(agent, content=str(action.params.get("text", "")), to=to)

    def _do_praise(self, agent: Agent, action: Action) -> None:
        """A macro: praise is a say aimed at a peer; the esteem effects are read
        off it by the interpretation layer (褒められて気持ちいい). Gating stays
        here — praise only means anything when the esteem layer is active."""
        if not self.status.enabled:
            return
        target = self._by_id.get(action.params.get("target"))
        if target is None or target is agent or not target.alive:
            return
        self._say(agent, to=target, kind="praise")

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
        # A macro: creating is making a work of art/scholarship at a venue.
        self._make_work(agent, str(action.params.get("title", "Untitled Work")))

    def _make_work(self, agent: Agent, title: str) -> Optional[Event]:
        """Self-actualization: produce a work. Only possible when every lower
        need is quiet — a hungry, scared or unrecognised mind cannot create."""
        if not self.psyche.enabled:
            return None
        if actualization_pull(agent, self.psyche) <= 0:
            return None
        f = self.world.facility_at(agent.pos)
        if f is None or f.ftype not in {FacilityType.LIBRARY, FacilityType.WORKSHOP,
                                        FacilityType.PLAZA}:
            return None
        agent.works_created += 1
        agent.fulfillment += self.psyche.fulfillment_per_work
        self.metrics.works_created += 1
        self.metrics.total_fulfillment += self.psyche.fulfillment_per_work
        # Creation is the deepest joy — and, if honour matters, it is admired.
        self._reward(agent, self.psyche.pleasure_per_work)
        self._recognise(agent, self.psyche.rep_per_work, 0.0, "work")
        self.world.log("work_created", by=agent.id, title=title, at=f.name)
        ev = Event(kind="make", actor=agent, site=f)
        self._interpret(ev)
        return ev

    def _do_make(self, agent: Agent, action: Action) -> None:
        """Raw primitive (LLM): transform effort/inputs into an output. A work
        of art/scholarship, or a recipe good (routes to the craft physics)."""
        output = action.params.get("output") or action.params.get("item") or "work"
        if output in MK.RECIPES:
            self._spend(agent, ActionType.MAKE)
            self._do_craft(agent, Action(ActionType.CRAFT, {"item": output}))
            return
        if output == "weapon":
            self._spend(agent, ActionType.MAKE)
            self._do_craft_weapon(agent, Action(ActionType.CRAFT_WEAPON, {}))
            return
        self._spend(agent, ActionType.MAKE)
        self._make_work(agent, str(action.params.get("title", "Untitled Work")))

    # -- society: weapons, drugs, gangs, religion -----------------------
    def _do_craft_weapon(self, agent: Agent, action: Action) -> None:
        # A macro: forging a weapon is a make at a workshop. Gating + the
        # material cost stay here; the production is the _make_weapon physics.
        if not (self.society.enabled and self.society.weapons):
            return
        f = self.world.facility_at(agent.pos)
        if f is None or not f.is_workplace():
            return
        if agent.take("materials", self.society.weapon_material_cost) \
                < self.society.weapon_material_cost:
            return
        self._make_weapon(agent, f)

    def _make_weapon(self, agent: Agent, f) -> Event:
        agent.weapons += 1
        self.metrics.weapons_crafted += 1
        f.add_role("weapons_factory")
        self.world.log("craft_weapon", by=agent.id, at=f.name)
        ev = Event(kind="make", actor=agent, site=f, items={"weapon": 1})
        self._interpret(ev)
        return ev

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
        # A macro: taking a dose is using a (self-cooked) drug. Gating + the
        # material self-supply stay here; the dose effect is in _use_item.
        if not (self.society.enabled and self.society.drugs):
            return
        # Self-supply if needed (an addict will cook their own).
        if agent.take("materials", self.society.drug_material_cost) \
                < self.society.drug_material_cost and agent.addiction < 10:
            return
        self._use_item(agent, "drug", 1)
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
        # A macro: joining a gang is a bond of allegiance to a crew. Gating
        # stays here; the join/found physics is read off the act by _interpret.
        if not (self.society.enabled and self.society.gangs) or agent.gang_id:
            return
        self._interpret(Event(kind="bond", actor=agent, intent="gang"))

    def _join_or_found_gang(self, agent: Agent) -> None:
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
        # A macro: preaching is a say carrying a sermon; founding/spreading the
        # faith is read off it by the interpretation layer. Gating stays here.
        if not (self.society.enabled and self.society.religion):
            return
        self._say(agent, kind="sermon")

    def _preach_faith(self, agent: Agent) -> None:
        """A sermon: found a faith if you have the standing, else spread yours
        to the trusting unaffiliated nearby."""
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
        # A macro: worship is a bond of allegiance to one's faith at a temple;
        # its relief/communion effects are read off it by the interpretation
        # layer. Gating (a faith, standing on a temple) stays here.
        if not (self.society.enabled and self.society.religion) or agent.faith is None:
            return
        f = self.world.facility_at(agent.pos)
        if f is None or "temple" not in f.roles:
            return
        self._interpret(Event(kind="bond", actor=agent, intent="worship"))

    def _worship_effect(self, agent: Agent) -> None:
        """Prayer eases fear and the need for esteem, brings a hit of peace, and
        binds the faithful who pray together."""
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

    # -- economic physics: offer / accept / craft ----------------------
    def _do_offer(self, agent: Agent, action: Action) -> None:
        if not self.economy:
            return
        p = action.params
        gi, wi = str(p.get("give_item", "")), str(p.get("want_item", ""))
        gq, wq = int(p.get("give_qty", 0) or 0), int(p.get("want_qty", 0) or 0)
        if gi not in MK.TRADABLE or wi not in MK.TRADABLE or gi == wi:
            return
        if gq <= 0 or wq <= 0 or MK.holdings(agent, gi) < gq:
            return  # can't offer what you don't have
        # One open offer per maker keeps the book honest and bounded.
        if any(o.maker == agent.id for o in self.offers):
            return
        if len(self.offers) >= MK.MAX_OPEN_OFFERS:
            return
        offer = MK.Offer(id=self._next_offer_id, maker=agent.id, give_item=gi,
                         give_qty=gq, want_item=wi, want_qty=wq, day=self.world.day)
        self._next_offer_id += 1
        self.offers.append(offer)
        self.world.log("offer", id=offer.id, by=agent.id,
                       give=f"{gq} {gi}", want=f"{wq} {wi}")

    def _do_accept(self, agent: Agent, action: Action) -> None:
        if not self.economy:
            return
        oid = action.params.get("offer_id")
        offer = next((o for o in self.offers if o.id == oid), None) if oid is not None \
            else None
        if offer is None or offer.maker == agent.id:
            return
        maker = self._by_id.get(offer.maker)
        if maker is None or not maker.alive:
            self.offers.remove(offer)
            return
        # Both sides must still hold the goods — conservation, no credit here.
        if MK.holdings(agent, offer.want_item) < offer.want_qty:
            return
        if MK.holdings(maker, offer.give_item) < offer.give_qty:
            self.offers.remove(offer)
            return
        apply_transfer(maker, agent, offer.give_item, offer.give_qty)
        apply_transfer(agent, maker, offer.want_item, offer.want_qty)
        self.offers.remove(offer)
        self.metrics.trades += 1
        # The settled ratio IS the price — emergent, recorded for observation.
        key = (offer.give_item, offer.want_item)
        ratio = offer.want_qty / offer.give_qty
        self._trade_ratios.setdefault(key, []).append(ratio)
        self._trade_ratios[key] = self._trade_ratios[key][-10:]
        maker.adjust_trust(agent.id, 0.05)
        agent.adjust_trust(maker.id, 0.05)
        self.world.log("trade", offer=offer.id, maker=maker.id, taker=agent.id,
                       gave=f"{offer.give_qty} {offer.give_item}",
                       got=f"{offer.want_qty} {offer.want_item}")

    def _do_craft(self, agent: Agent, action: Action) -> None:
        if not self.economy:
            return
        item = str(action.params.get("item", ""))
        recipe = MK.RECIPES.get(item)
        if recipe is None:
            return
        inputs, need_facility = recipe
        if need_facility is not None:
            f = self.world.facility_at(agent.pos)
            if f is None or f.ftype.value != need_facility:
                return
        if any(agent.inventory.get(k, 0) < q for k, q in inputs.items()):
            return
        for k, q in inputs.items():
            agent.take(k, q)
        agent.add(item, 1)
        self.metrics.crafted += 1
        self.world.log("craft", by=agent.id, item=item)

    def emergent_price(self, give_item: str, want_item: str):
        """The average recent settled ratio (price of give_item in want_item)."""
        ratios = self._trade_ratios.get((give_item, want_item))
        if not ratios:
            return None
        return round(sum(ratios) / len(ratios), 2)

    def _do_lend(self, agent: Agent, action: Action) -> None:
        if not self.economy:
            return
        p = action.params
        debtor = self._by_id.get(p.get("to"))
        item = str(p.get("item", "money"))
        principal = int(p.get("qty", 0) or 0)
        repay = int(p.get("repay", principal) or principal)
        due_in_raw = p.get("due_in_days", MK.DEFAULT_LOAN_DUE_DAYS)
        due_in = int(due_in_raw) if due_in_raw is not None else MK.DEFAULT_LOAN_DUE_DAYS
        due_in = max(0, due_in)
        if debtor is None or debtor is agent or not debtor.alive:
            return
        if item not in MK.TRADABLE or principal <= 0 or repay <= 0:
            return
        if MK.holdings(agent, item) < principal:
            return  # can't lend what you don't have
        # Hand over the principal now; record the promise to repay later.
        apply_transfer(agent, debtor, item, principal)
        loan = MK.Loan(id=self._next_loan_id, creditor=agent.id, debtor=debtor.id,
                       item=item, principal=principal, repay=repay,
                       due_day=self.world.day + due_in)
        self._next_loan_id += 1
        self.loans.append(loan)
        self.metrics.loans_made += 1
        self.world.log("loan", id=loan.id, creditor=agent.id, debtor=debtor.id,
                       principal=f"{principal} {item}", repay=f"{repay} {item}")

    def _do_repay(self, agent: Agent, action: Action) -> None:
        if not self.economy:
            return
        lid = action.params.get("loan_id")
        loan = next((l for l in self.loans
                     if l.id == lid and not l.settled and not l.defaulted), None)
        if loan is None or loan.debtor != agent.id:
            return
        creditor = self._by_id.get(loan.creditor)
        if creditor is None or not creditor.alive:
            loan.settled = True  # creditor gone; debt lapses
            return
        if MK.holdings(agent, loan.item) < loan.repay:
            return  # can't settle yet
        apply_transfer(agent, creditor, loan.item, loan.repay)
        loan.settled = True
        self.metrics.loans_repaid += 1
        # Honouring credit builds trust — the collateral of a credit economy.
        creditor.adjust_trust(agent.id, 0.2)
        agent.adjust_trust(creditor.id, 0.1)
        self.world.log("loan_repaid", id=loan.id, debtor=agent.id, creditor=creditor.id)

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
        # A macro: theft is just a non-consensual take from an agent. The act
        # moves the items; the interpretation layer reads "consent-less take
        # from a person" as the crime of theft (see _interpret). Money is an
        # inventory item like any other, so a thief takes coin and food alike.
        victim = self._adjacent_or_targeted(agent, action)
        if victim is None:
            return
        self._spend(agent, ActionType.STEAL)          # accounting stays in the macro
        self._move_items(agent, victim, {"money": 5, "food": 2},
                         kind="take", consent=False)

    def _do_attack(self, agent: Agent, action: Action) -> None:
        # A macro: attack is striking a person (interpreted as violence).
        victim = self._adjacent_or_targeted(agent, action)
        if victim is None:
            return
        self._spend(agent, ActionType.ATTACK)
        self._strike(agent, victim=victim)

    def _do_arson(self, agent: Agent, action: Action) -> None:
        # A macro: arson is striking a structure (interpreted as arson).
        name = action.params.get("facility_name")
        f = next((x for x in self.world.facilities if x.name == name), None)
        if f is None:
            f = self.world.facility_at(agent.pos)
        self._spend(agent, ActionType.ARSON)
        self._strike(agent, facility=f)

    def _do_report_crime(self, agent: Agent, action: Action) -> None:
        # A macro: reporting a crime is a say aimed at the accused.
        target = self._by_id.get(action.params.get("target"))
        self._say(agent, to=target, kind="accusation")

    def _is_wanted(self, a: Agent) -> bool:
        """A recent offender is arrestable until the window lapses."""
        return (a.last_crime_day is not None
                and self.world.day - a.last_crime_day <= ARREST_WINDOW_DAYS)

    def _do_arrest(self, agent: Agent, action: Action) -> None:
        """Enforcement as an *act*: an agent (typically a guard) collars a
        nearby recent offender. Order is no longer radiated by a building —
        someone has to choose to keep the peace and reach the offender."""
        target = self._adjacent_or_targeted(agent, action)
        if target is None or target is agent:
            return
        # No detaining the innocent: only a recent, still-wanted offender.
        if not self._is_wanted(target):
            return
        self._spend(agent, ActionType.ARREST)
        # A real, agent-driven cost for crime: detainment drains the offender,
        # a fine is levied, and (if one exists) they are hauled to a prison.
        fine = min(target.money, self.policy.config.fine_amount)
        if self.economy:
            self.treasury += target.take("money", fine)   # fine → the state (conserved)
        else:
            target.money -= fine
            self.world.granary_food += fine // 2
        target.energy -= ARREST_ENERGY_PENALTY
        target.times_arrested += 1
        target.last_crime_day = None      # justice served; no longer wanted
        prison = self.world.nearest(target.pos, FacilityType.PRISON)
        if prison is not None:
            target.pos = prison.pos
        agent.reputation += 1.0           # keeping the peace earns standing
        self.metrics.arrests += 1
        self.world.log("arrest", guard=agent.id, offender=target.id, fine=fine)

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
        offender.last_crime_day = self.world.day   # now "wanted" for a short while
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
                if self.economy:
                    paid = offender.take("money", fine)      # conserved:
                    half = paid // 2
                    victim.add("money", half)                # compensation to the victim
                    self.treasury += paid - half             # the rest to the state
                else:
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
        child.inventory.update({"food": 2, "materials": 0})  # keep money the ctor set
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
        # Vertical cultural inheritance: the elders pass on a lesson, which the
        # child will revise through its own life. Only the *core* lesson carries
        # over, so oral tradition doesn't nest ("taught me: taught me: ...").
        if self.library is not None:
            for p in (parent_a, parent_b):
                firsthand = [m for m in p.memory
                             if not m.startswith("I read in the library:")]
                if firsthand:
                    child.remember(f"My elder {p.name} taught me: "
                                   f"{self._core_lesson(firsthand[-1])}")
        self.metrics.births += 1
        self.world.log("birth", child=child.id, parents=f"{parent_a.id}+{parent_b.id}",
                       persona=persona_key)

    @staticmethod
    def _core_lesson(text: str) -> str:
        """Strip a prior "… taught me: " wrapper so transmitted lessons keep their
        substance instead of nesting over generations."""
        marker = "taught me: "
        return text.split(marker)[-1] if marker in text else text

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
        if self.economy:
            self.offers = [o for o in self.offers
                           if self.world.day - o.day < MK.OFFER_TTL_DAYS]
            # Overdue, unsettled loans default — the creditor eats the loss and
            # learns to distrust the debtor (credit dries up for defaulters).
            for loan in self.loans:
                if loan.settled or loan.defaulted or self.world.day <= loan.due_day:
                    continue
                loan.defaulted = True
                self.metrics.loan_defaults += 1
                creditor = self._by_id.get(loan.creditor)
                debtor = self._by_id.get(loan.debtor)
                if creditor is not None and debtor is not None:
                    creditor.adjust_trust(debtor.id, -0.4)
                    debtor.remember(
                        f"Day {self.world.day}: I defaulted on a loan from {debtor.name}.")
                self.world.log("loan_default", id=loan.id,
                               debtor=loan.debtor, creditor=loan.creditor)
            self.loans = [l for l in self.loans if not (l.settled or l.defaulted)]
        if self.public_works:
            # A daily civic levy fills the state treasury that funds construction.
            for a in self.agents:
                if a.alive:
                    paid = min(a.money, PW.CIVIC_LEVY_PER_AGENT)
                    a.money -= paid
                    self.treasury += paid
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
            "arrests": self.metrics.arrests,
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

    def _published_laws(self) -> list:
        """Every enacted law, published as text for the brain to act on — so an
        LLM agent can read, obey, or enforce even legislation the engine has no
        built-in mechanism for. De-duplicated by text (re-passing the same bill
        is not new news), most recent first, capped to keep the view lean. The
        heuristic brain ignores this, so offline outcomes are unchanged."""
        out: list = []
        seen: set[str] = set()
        for law in reversed(self.policy.laws):
            if law.text in seen:
                continue
            seen.add(law.text)
            out.append({"text": law.text,
                        "effects": [e.value for e in law.effects],
                        "day": law.enacted_day})
            if len(out) >= 8:
                break
        return out

    def _enforcement_expectation(self) -> float:
        """How credibly crime is punished, in 0..1 — derived from real world
        state, not a constant. Norms only restrain when someone can actually
        enforce them: living guards who can ARREST, backed by the facilities
        that host them. A published norm with no enforcers deters no one."""
        guards = sum(1 for a in self.agents
                     if a.alive and a.profession == "guard")
        stations = (len(self.world.facilities_of(FacilityType.POLICE_STATION))
                    + len(self.world.facilities_of(FacilityType.PRISON)))
        guard_term = min(1.0, guards * 0.5)       # one guard -> 0.5, two+ -> 1.0
        station_term = min(1.0, stations * 0.5)
        return min(1.0, 0.6 * guard_term + 0.4 * station_term)

    def _apply_daily_policy(self) -> None:
        """Run once per day: tax, food redistribution. With ``--economy`` these
        obey conservation — money moves to/from the treasury rather than vanishing
        or being conjured (the principled form: tax = a coerced take into the
        state, welfare = a grant of money the state actually holds). Offline keeps
        the legacy behaviour, so the four-society baseline is byte-identical."""
        living = [a for a in self.agents if a.alive]
        if not living:
            return
        if self.policy.has_tax():
            top = sorted(living, key=lambda a: a.money, reverse=True)[: max(1, len(living) // 3)]
            for a in top:
                tribute = int(a.money * self.policy.config.tax_rate)
                if self.economy:
                    paid = a.take("money", tribute)          # coerced take → state
                    self.treasury += paid                    # conserved
                    if paid:
                        self.world.log("tax", payer=a.id, amount=paid)
                else:
                    a.money -= tribute
                    self.world.granary_food += tribute // 2  # legacy: money→food magic
            self.metrics.tax_days += 1
        if self.policy.has_food_redistribution():
            if self.economy:
                # Welfare: grant money the state actually holds to the poorest
                # (who then buy food at the market) — conserved, not conjured.
                poor = sorted(living, key=lambda a: a.money)[: max(1, len(living) // 3)]
                for a in poor:
                    pay = min(self.treasury, FISCAL_WELFARE)
                    if pay <= 0:
                        break
                    self.treasury -= pay
                    a.add("money", pay)
                    self.world.log("welfare", payee=a.id, amount=pay)
            else:
                grant = max(2, len(living))
                self.world.granary_food += grant             # legacy: food from nothing

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
        self.metrics.treasury_final = self.treasury
        if self.development:
            self.metrics.prosperity = DEV.prosperity(self)

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

    def _near_safety(self, agent: Agent) -> bool:
        """Within comforting reach of a police station, prison, or a house."""
        for ftype in (FacilityType.POLICE_STATION, FacilityType.PRISON,
                      FacilityType.HOUSE):
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
