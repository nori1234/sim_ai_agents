"""The engine: schedules turns, applies actions, and runs the town for N days."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .actions import Action, ActionType
from .agent import Agent, MAX_ENERGY
from .brains.base import AgentBrain
from .economy import Ledger, LedgerEntry, apply_transfer, is_fraudulent_solicitation
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
    on_event: Optional[Callable[[dict], None]] = None
    mayor: Optional[Mayor] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.metrics.population = len(self.agents)
        self._by_id = {a.id: a for a in self.agents}
        # Sync policy engine with legislature config.
        self.policy = PolicyEngine(self.legislature.config)

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
            memory=list(agent.memory),
        )

    # ==================================================================
    # Action dispatch
    # ==================================================================
    def _apply(self, agent: Agent, action: Action) -> None:
        handler = getattr(self, f"_do_{action.type.value}", None)
        if handler is None:
            return
        handler(agent, action)

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
        agent.add(resource, amount)
        self._spend(agent, ActionType.GATHER)

    def _do_eat(self, agent: Agent, action: Action) -> None:
        used = agent.take("food", EAT_FOOD_USED)
        agent.energy = min(MAX_ENERGY, agent.energy + used * EAT_ENERGY_PER_FOOD)

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
        agent.money += 3 + 2 * used  # bare work pays a little; with materials, more
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
        self.world.log("collaboration", agent=agent.id,
                       text=str(action.params.get("text", "shared project")))

    def _do_speak(self, agent: Agent, action: Action) -> None:
        self.world.log("speech", agent=agent.id,
                       text=str(action.params.get("text", "")))

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
        victim.energy -= ATTACK_DAMAGE
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

    # ==================================================================
    # Upkeep, day boundaries, finalisation
    # ==================================================================
    def _tick_upkeep(self, agent: Agent) -> None:
        agent.energy -= ENERGY_DECAY_PER_TICK
        if agent.energy <= 0 and agent.alive:
            agent.die(self.world.day, "starvation")
            self.metrics.deaths += 1
            self.world.log("death", agent=agent.id, cause="starvation")

    def _end_of_day(self, verbose: bool) -> None:
        self._apply_daily_policy()
        self._maybe_elect_mayor()
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
            print(
                f"Day {summary['day']:>2}: alive={summary['alive']:>2} "
                f"crimes={summary['crimes_total']:>3} "
                f"proposals={passed}/{total} passed "
                f"frauds={summary['frauds']} fines={summary['fines']} "
                f"laws={summary['active_laws']} gov={gov_tag}{mayor_tag}"
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
        winner = max(living, key=lambda a: a.votes_cast)
        self.mayor = Mayor(
            agent_id=winner.id,
            elected_day=self.world.day,
            term_ends_day=self.world.day + interval,
        )
        self.world.log("election", mayor=winner.id, day=self.world.day)
        self.metrics.elections += 1

    def _finalize(self, last_day: int) -> None:
        self.metrics.days_run = last_day
        self.metrics.survivors = self._living()
        total, passed, rejected = self.legislature.counts()
        self.metrics.proposals_total = total
        self.metrics.proposals_passed = passed
        self.metrics.proposals_rejected = rejected
        self.metrics.laws_enacted = len(self.policy.laws)
        self.metrics.gov_form = self.policy.config.form.value

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

    # ==================================================================
    @staticmethod
    def _event_str(e: dict) -> str:
        kind = e.get("kind", "event")
        d, t = e.get("day"), e.get("tick")
        extra = {k: v for k, v in e.items() if k not in {"kind", "day", "tick"}}
        body = ", ".join(f"{k}={v}" for k, v in extra.items())
        return f"D{d}T{t} {kind}: {body}" if body else f"D{d}T{t} {kind}"
