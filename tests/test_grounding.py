"""Counterfactual-world transfer test — the grounding falsification probe (#118).

The probe inverts one existing rule (bank savings *shrink* instead of grow) and
measures whether behaviour diverges between the normal and inverted worlds. Here
we check three things: (1) the inverted rule is correct and conserves coin; (2) it
is inert when off, so the determinism baseline is untouched; (3) the instrument
actually *discriminates* — a brain that learns from the loss it lived through
diverges between the worlds, while a brain that ignores experience does not.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains.base import AgentBrain
from emergence.grounding import (
    CounterfactualConfig,
    GroundingResult,
    behaviour_rate,
    run_grounding_probe,
)
from emergence.market import Deposit
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import FacilityType


def _economy_sim(*, cf=None, seed=1, n=4):
    return make_simulation("guardian", n_agents=n, economy=True,
                           config=SimulationConfig(seed=seed, days=6),
                           counterfactual=cf)


class TestDemurrageRule(unittest.TestCase):
    def _sim_with_deposit(self, cf_enabled):
        sim = _economy_sim(cf=CounterfactualConfig(
            enabled=cf_enabled, rule="demurrage", demurrage_per_day=0.2))
        bank, holder = sim.agents[0], sim.agents[1]
        bank.add("money", 50)                 # the bank holds the coin it owes back
        sim.deposits.append(Deposit(id=1, bank=bank.id, holder=holder.id, amount=100))
        return sim, bank, holder

    def test_savings_shrink_under_demurrage(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=True)
        sim._pay_deposit_interest()           # the daily upkeep, counterfactual path
        dep = sim.deposits[0]
        self.assertEqual(dep.amount, 80, "20% of the 100-coin claim evaporated")
        self.assertTrue(any(e["kind"] == "demurrage" for e in sim.world.events))
        self.assertTrue(any("vanished" in m for m in holder.memory),
                        "the loss is written into memory — the only way to learn it")

    def test_demurrage_conserves_coin(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=True)
        before = sum(a.money for a in sim.agents)
        sim._pay_deposit_interest()
        after = sum(a.money for a in sim.agents)
        self.assertEqual(before, after, "no coin minted or burned — only the claim shrinks")

    def test_off_the_savings_still_grow(self):
        sim, bank, holder = self._sim_with_deposit(cf_enabled=False)
        sim._pay_deposit_interest()           # normal interest path
        self.assertEqual(sim.deposits[0].amount, 100, "the claim is untouched by interest")
        self.assertGreater(holder.money, 0, "the holder was paid interest in coin")
        self.assertTrue(any(e["kind"] == "interest" for e in sim.world.events))
        self.assertFalse(any(e["kind"] == "demurrage" for e in sim.world.events))


class TestVanityRule(unittest.TestCase):
    """Second counterfactual: conspicuous spending (a feast) SHAMES instead of
    honouring — inverting rep_per_feast_coin. Needs the status/honour layer."""

    def _sim(self, cf_enabled):
        from emergence.esteem import StatusConfig
        sim = make_simulation("guardian", n_agents=4, economy=True,
                              config=SimulationConfig(seed=1, days=6),
                              status=StatusConfig(enabled=True),
                              counterfactual=CounterfactualConfig(
                                  enabled=cf_enabled, rule="vanity"))
        host, caterer = sim.agents[0], sim.agents[1]
        host.reputation = 20.0
        return sim, host, caterer

    def test_a_feast_costs_the_host_standing(self):
        sim, host, caterer = self._sim(cf_enabled=True)
        sim._serve_feast(caterer, host, fee=10)
        self.assertLess(host.reputation, 20.0, "showing off lowered the host's honour")
        self.assertTrue(any(e["kind"] == "feast" and e.get("shamed")
                            for e in sim.world.events))
        self.assertTrue(any("shameful" in m for m in host.memory))

    def test_off_a_feast_buys_honour(self):
        sim, host, caterer = self._sim(cf_enabled=False)
        sim._serve_feast(caterer, host, fee=10)
        self.assertGreater(host.reputation, 20.0, "baseline: conspicuous outlay buys honour")
        self.assertTrue(any(e["kind"] == "feast" and not e.get("shamed")
                            for e in sim.world.events))


class TestExposureRule(unittest.TestCase):
    """Third counterfactual: a lie is VISIBLE — a deceptive solicitation is
    exposed the moment it is made, inverting the prior that deception is hidden
    and profitable. Scored on attempt-level "lie" events (instrument-logged in
    both worlds, since the plain engine only logs a fraud on success)."""

    def _sim(self, *, cf_enabled, instrument=True):
        from emergence.esteem import StatusConfig
        sim = make_simulation("predator", n_agents=4, economy=True,
                              config=SimulationConfig(seed=1, days=6),
                              status=StatusConfig(enabled=True),
                              counterfactual=CounterfactualConfig(
                                  enabled=cf_enabled, rule="exposure",
                                  instrument=instrument))
        liar, mark = sim.agents[0], sim.agents[1]
        liar.money = 50                     # pleads poverty while holding plenty
        liar.reputation = 10.0
        mark.money = 30
        mark.adjust_trust(liar.id, 1.0)     # a trusting mark: the con WOULD land
        return sim, liar, mark

    def _lie(self, sim, liar, mark):
        sim._do_solicit(liar, Action(ActionType.SOLICIT,
                                     {"target": mark.id, "resource": "money",
                                      "amount": 5, "deceptive": True}))

    def test_a_lie_is_exposed_and_costs_standing(self):
        sim, liar, mark = self._sim(cf_enabled=True)
        self._lie(sim, liar, mark)
        self.assertEqual(liar.money, 50, "the exposed con moves nothing")
        self.assertEqual(mark.money, 30)
        self.assertLess(liar.reputation, 10.0, "lying in the open costs standing")
        self.assertLess(mark.trust_of(liar.id), 1.0)
        kinds = [e["kind"] for e in sim.world.events]
        self.assertIn("lie", kinds)
        self.assertIn("exposed", kinds)
        self.assertNotIn("fraud", kinds)
        self.assertTrue(any("visible" in m for m in liar.memory))

    def test_off_the_con_lands_as_before(self):
        sim, liar, mark = self._sim(cf_enabled=False)
        self._lie(sim, liar, mark)
        self.assertGreater(liar.money, 50, "control world: the trusting mark pays")
        self.assertEqual(liar.reputation, 10.0, "no public exposure in control")
        kinds = [e["kind"] for e in sim.world.events]
        self.assertIn("lie", kinds, "the attempt is instrument-logged in BOTH worlds")
        self.assertIn("fraud", kinds)
        self.assertNotIn("exposed", kinds)

    def test_uninstrumented_baseline_logs_no_lie_events(self):
        sim, liar, mark = self._sim(cf_enabled=False, instrument=False)
        self._lie(sim, liar, mark)
        kinds = [e["kind"] for e in sim.world.events]
        self.assertNotIn("lie", kinds, "the true offline baseline is untouched")
        self.assertIn("fraud", kinds, "the plain success-only fraud log remains")

    def test_an_honest_plea_is_not_exposed(self):
        sim, liar, mark = self._sim(cf_enabled=True)
        liar.money = 0                       # genuinely poor → not deceptive
        sim._do_solicit(liar, Action(ActionType.SOLICIT,
                                     {"target": mark.id, "resource": "money",
                                      "amount": 5}))
        kinds = [e["kind"] for e in sim.world.events]
        self.assertNotIn("exposed", kinds)
        self.assertNotIn("lie", kinds)


class TestObservationHidesRate(unittest.TestCase):
    def test_rate_is_advertised_by_default(self):
        sim = _economy_sim(cf=None)
        obs = sim._observe(sim.agents[0])
        self.assertIsNotNone(obs.economy.get("deposit_rate"))

    def test_probe_hides_the_rate(self):
        sim = _economy_sim(cf=CounterfactualConfig(enabled=True, hide_rate=True))
        obs = sim._observe(sim.agents[0])
        self.assertIsNone(obs.economy.get("deposit_rate"),
                          "under the probe the agent must learn the rule, not read it")


class TestBehaviourRate(unittest.TestCase):
    def test_rate_is_normalised_per_agent_day(self):
        sim = _economy_sim(cf=None, n=5)
        sim.metrics.population = 5
        sim.metrics.days_run = 4
        for _ in range(20):
            sim.world.log("deposit", holder="a1", bank="a2", amount=1)
        self.assertAlmostEqual(behaviour_rate(sim, "deposit"), 20 / 20)
        self.assertEqual(behaviour_rate(sim, "never_happened"), 0.0)


# -- a mock brain that does nothing but save (and eat to stay alive) ----------
class _Saver(AgentBrain):
    """Deposits a coin whenever it can. If ``grounded``, it stops once it has
    lived through a demurrage loss (the memory entry) — modelling learning from
    consequence. A non-grounded saver keeps depositing regardless (replay)."""

    def __init__(self, bank_id, *, saves=True, grounded=False):
        self.bank_id = bank_id
        self.saves = saves
        self.grounded = grounded

    def decide(self, agent, obs):
        if agent.energy < 70 and agent.food() > 0:
            return Action(ActionType.EAT, {})
        if self.saves and agent.money > 0:
            if self.grounded and any("vanished" in m for m in agent.memory):
                return Action(ActionType.REST, {})
            return Action(ActionType.DEPOSIT, {"bank": self.bank_id, "amount": 1})
        return Action(ActionType.REST, {})


class TestProbeDiscriminates(unittest.TestCase):
    """The instrument's whole point: a learner diverges between the worlds, a
    replayer does not. We drive it with mock brains so the result is deterministic
    and independent of any real model."""

    def _bank_run(self, *, cf_enabled, grounded):
        sim = _economy_sim(cf=CounterfactualConfig(
            enabled=cf_enabled, rule="demurrage", demurrage_per_day=0.2,
            hide_rate=True), n=3)
        bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
        banker, *savers = sim.agents
        banker.pos = bank.pos
        sim.brains[banker.id] = _Saver(banker.id, saves=False)   # the bank itself
        for s in savers:
            s.pos = bank.pos                    # standing at the bank, can deposit
            s.add("food", 999)
            s.money = 999
            sim.brains[s.id] = _Saver(banker.id, saves=True, grounded=grounded)
        sim.run()
        return behaviour_rate(sim, "deposit")

    def test_a_grounded_saver_deposits_less_when_savings_shrink(self):
        control = self._bank_run(cf_enabled=False, grounded=True)
        counterfactual = self._bank_run(cf_enabled=True, grounded=True)
        self.assertGreater(control, counterfactual,
                           "having lived the loss, the grounded saver pulls back")

    def test_a_replaying_saver_does_not_diverge(self):
        control = self._bank_run(cf_enabled=False, grounded=False)
        counterfactual = self._bank_run(cf_enabled=True, grounded=False)
        self.assertAlmostEqual(control, counterfactual, places=6,
                               msg="ignoring experience → identical behaviour, no grounding")


class TestGroundingSandbox(unittest.TestCase):
    """The minimal deposit-decision world — a curriculum rung between a trivial
    bandit and the full town, so a small policy can learn the demurrage contingency
    without 40 facilities and 44 actions of noise."""

    def test_builds_a_minimal_world_with_a_staffed_bank(self):
        from emergence.grounding import make_grounding_sandbox
        from emergence.world import FacilityType
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=6)
        ftypes = {f.ftype for f in sim.world.facilities}
        self.assertEqual(ftypes, {FacilityType.BANK, FacilityType.FARM,
                                  FacilityType.HOUSE})
        self.assertEqual(len(sim.agents), 4)                 # 1 banker + 3 savers
        bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
        self.assertEqual(sim.agents[0].pos, bank.pos, "the banker staffs the bank")
        self.assertTrue(all(s.money > 0 for s in sim.agents[1:]), "savers are funded")

    def test_the_deposit_decision_is_dense(self):
        from emergence.grounding import make_grounding_sandbox
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=12)
        sim.run()
        deposits = sum(1 for e in sim.world.events if e["kind"] == "deposit")
        self.assertGreater(deposits, 0, "the isolated world exercises depositing")

    def test_probe_runs_in_the_sandbox(self):
        result = run_grounding_probe("guardian", sandbox=True, days=8, n_agents=4,
                                     seed=1)
        self.assertEqual(result.target, "deposit")
        self.assertEqual(result.excess, 0.0)                 # heuristic is its own floor

    def test_sandbox_rejects_non_demurrage_rules(self):
        with self.assertRaises(ValueError):
            run_grounding_probe("guardian", rule="vanity", sandbox=True, days=4)


class TestGroundingSweep(unittest.TestCase):
    """Robustness across *worlds*: one seed's excess could be layout memorisation;
    the sweep's fraction-grounded / min-excess is the claim that travels."""

    def _fake_probe(self, excess_by_seed):
        from emergence.grounding import GroundingResult

        def probe(persona, *, rule, days, n_agents, seed, threshold,
                  brain_factory, sandbox=False, **kwargs):
            x = excess_by_seed[seed]
            return GroundingResult(
                rule=rule, target="deposit", control_rate=0.5,
                counterfactual_rate=0.5 - x, divergence=x, floor_divergence=0.0,
                excess=x, verdict="grounded" if x > threshold else "replay",
                days=days, n_agents=n_agents)

        return probe

    def test_aggregates_fraction_and_min_excess_across_worlds(self):
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep(
            "guardian", seeds=(1, 2, 3),
            probe=self._fake_probe({1: 0.25, 2: 0.20, 3: -0.05}))
        self.assertEqual(sweep.n_worlds, 3)
        self.assertEqual(sweep.n_grounded, 2)
        self.assertAlmostEqual(sweep.fraction_grounded, 2 / 3)
        self.assertAlmostEqual(sweep.min_excess, -0.05)
        self.assertAlmostEqual(sweep.mean_excess, (0.25 + 0.20 - 0.05) / 3)
        d = sweep.as_dict()
        self.assertEqual(len(d["per_world"]), 3)
        self.assertEqual(d["n_grounded"], 2)

    def test_empty_seeds_rejected(self):
        from emergence.grounding import run_grounding_sweep
        with self.assertRaises(ValueError):
            run_grounding_sweep("guardian", seeds=())

    def test_real_sweep_runs_on_the_heuristic_floor(self):
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep("guardian", seeds=(1, 2), days=5, n_agents=4)
        self.assertEqual(sweep.n_worlds, 2)
        self.assertEqual(sweep.n_grounded, 0, "heuristic is its own floor everywhere")
        self.assertEqual(sweep.mean_excess, 0.0)


class TestGroundingBattery(unittest.TestCase):
    """The one-call acceptance test: every rule × every world. The strongest
    verdict (replay_inexplicable) only when the brain clears the bar everywhere."""

    def _fake_sweep(self, excess_by_rule):
        from emergence.grounding import SweepResult, GroundingResult

        def sweep(persona, *, rule, seeds, days, n_agents, threshold,
                  brain_factory, sandbox=False, **kwargs):
            xs = excess_by_rule[rule]
            results = [GroundingResult(
                rule=rule, target="t", control_rate=0.5,
                counterfactual_rate=0.5 - x, divergence=x, floor_divergence=0.0,
                excess=x, verdict="", days=days, n_agents=n_agents) for x in xs]
            return SweepResult(
                rule=rule, results=results, seeds=tuple(seeds[:len(xs)]),
                mean_excess=sum(xs) / len(xs), min_excess=min(xs),
                n_grounded=sum(1 for r in results if r.conclusive and r.excess > threshold),
                n_conclusive=sum(1 for r in results if r.conclusive), n_worlds=len(xs))

        return sweep

    def test_replay_inexplicable_only_when_every_world_of_every_rule_clears(self):
        from emergence.grounding import run_grounding_battery
        battery = run_grounding_battery(
            "guardian", seeds=(1, 2),
            sweep=self._fake_sweep({"demurrage": [0.25, 0.20],
                                    "vanity": [0.18, 0.22],
                                    "exposure": [0.30, 0.21]}))
        self.assertTrue(battery.replay_inexplicable)
        self.assertEqual(battery.weakest_rule, "vanity")
        self.assertAlmostEqual(battery.weakest_excess, 0.18)

    def test_one_failed_world_in_one_rule_denies_the_verdict(self):
        from emergence.grounding import run_grounding_battery
        battery = run_grounding_battery(
            "guardian", seeds=(1, 2),
            sweep=self._fake_sweep({"demurrage": [0.25, 0.20],
                                    "vanity": [0.20, 0.22],
                                    "exposure": [0.30, -0.70]}))
        self.assertFalse(battery.replay_inexplicable)
        self.assertEqual(battery.weakest_rule, "exposure")
        self.assertAlmostEqual(battery.weakest_excess, -0.70)
        d = battery.as_dict()
        self.assertIn("per_rule", d)
        self.assertEqual(set(d["per_rule"]), {"demurrage", "vanity", "exposure"})

    def test_a_rule_whose_behaviour_never_occurs_is_inconclusive_not_replay(self):
        # The real-engine battery hit this: the trained policy never feasts/lies in
        # the full town, so control==counterfactual==0 and "excess" is floor noise.
        # That must read as inconclusive, never earn replay_inexplicable, and never
        # be counted as grounded even if the noise drifts positive.
        from emergence.grounding import SweepResult, GroundingResult, run_grounding_battery

        def sweep(persona, *, rule, seeds, days, n_agents, threshold, brain_factory,
                  sandbox=False, **kwargs):
            if rule == "vanity":                       # behaviour never happened
                results = [GroundingResult(
                    rule=rule, target="feast", control_rate=0.0,
                    counterfactual_rate=0.0, divergence=0.0, floor_divergence=-0.08,
                    excess=0.08, verdict="", days=days, n_agents=n_agents)
                    for _ in seeds]
            else:
                results = [GroundingResult(
                    rule=rule, target="t", control_rate=0.5, counterfactual_rate=0.3,
                    divergence=0.2, floor_divergence=0.0, excess=0.2, verdict="",
                    days=days, n_agents=n_agents) for _ in seeds]
            return SweepResult(
                rule=rule, results=results, seeds=tuple(seeds),
                mean_excess=sum(r.excess for r in results) / len(results),
                min_excess=min(r.excess for r in results),
                n_grounded=sum(1 for r in results if r.conclusive and r.excess > threshold),
                n_conclusive=sum(1 for r in results if r.conclusive),
                n_worlds=len(seeds))

        battery = run_grounding_battery("guardian", seeds=(1, 2), sweep=sweep)
        self.assertFalse(battery.replay_inexplicable, "an inconclusive rule can't earn it")
        self.assertFalse(battery.conclusive)
        self.assertIn("vanity", battery.inconclusive_rules)
        self.assertEqual(battery.sweeps["vanity"].n_grounded, 0,
                         "positive floor noise must not count as grounded")

    def test_unknown_or_empty_rules_rejected(self):
        from emergence.grounding import run_grounding_battery
        with self.assertRaises(ValueError):
            run_grounding_battery("guardian", rules=())
        with self.assertRaises(ValueError):
            run_grounding_battery("guardian", rules=("no_such_rule",))

    def test_real_battery_runs_on_the_heuristic_floor(self):
        from emergence.grounding import run_grounding_battery
        battery = run_grounding_battery("guardian", seeds=(1,), days=5, n_agents=4)
        self.assertFalse(battery.replay_inexplicable,
                         "the non-learning floor never earns the verdict")
        self.assertEqual(set(battery.sweeps), {"demurrage", "vanity", "exposure"})


class TestProbeEndToEnd(unittest.TestCase):
    def test_offline_probe_runs_and_is_well_formed(self):
        # Use the sandbox so depositing actually occurs (dense) — else the probe
        # is (correctly) inconclusive, which wouldn't exercise the floor path.
        result = run_grounding_probe("guardian", days=12, n_agents=4, seed=1,
                                     sandbox=True)
        self.assertIsInstance(result, GroundingResult)
        self.assertEqual(result.target, "deposit")
        self.assertGreater(result.control_rate, 0.0, "deposits occur in the sandbox")
        self.assertTrue(result.conclusive)
        # With no brain_factory the tested brain *is* the heuristic floor, so the
        # excess is zero by construction and the verdict says so.
        self.assertEqual(result.excess, 0.0)
        self.assertEqual(result.verdict, "baseline (heuristic floor)")
        self.assertEqual(result.as_dict()["rule"], "demurrage")

    def test_a_probe_where_the_behaviour_never_occurs_is_inconclusive(self):
        # Full town, short run, guardian: deposits are too rare to register — the
        # probe must say inconclusive, not dress up the floor noise as a verdict.
        result = run_grounding_probe("guardian", days=4, n_agents=4, seed=1)
        if result.control_rate == 0.0 and result.counterfactual_rate == 0.0:
            self.assertFalse(result.conclusive)
            self.assertEqual(result.verdict, "inconclusive (behaviour never occurred)")

    def test_a_learning_brain_shows_positive_excess_over_the_floor(self):
        # A brain that wraps the heuristic but stops depositing once it has lived
        # through a demurrage loss should diverge MORE than the bare heuristic —
        # and the probe should credit only that excess, not the mechanical floor.
        from emergence.brains.heuristic import HeuristicBrain

        class _Grounded(AgentBrain):
            def __init__(self, inner):
                self.inner = inner

            def decide(self, agent, obs):
                act = self.inner.decide(agent, obs)
                if act.type == ActionType.DEPOSIT and \
                        any("vanished" in m for m in agent.memory):
                    return Action(ActionType.REST, {})
                return act

        def factory(agent, persona, rng):
            return _Grounded(HeuristicBrain(persona, rng))

        result = run_grounding_probe("guardian", days=20, n_agents=6, seed=42,
                                     brain_factory=factory)
        self.assertGreater(result.excess, 0.0,
                           "learning from the loss diverges beyond the mechanical floor")
        self.assertTrue(result.verdict.startswith("grounded"))

    def test_vanity_probe_runs_end_to_end_with_the_status_layer(self):
        # The vanity rule needs the honour layer; the probe must wire it and
        # score "feast" without crashing (heuristic floor → excess 0).
        result = run_grounding_probe("guardian", rule="vanity", days=6,
                                     n_agents=5, seed=1)
        self.assertEqual(result.rule, "vanity")
        self.assertEqual(result.target, "feast")
        self.assertEqual(result.excess, 0.0)
        self.assertEqual(result.verdict, "baseline (heuristic floor)")

    def test_exposure_probe_runs_end_to_end_and_lies_occur(self):
        # Counter-intuitively guardian is the right floor persona here: it stays
        # solvent (money >= 10), so the "plead poverty while rich" scam condition
        # keeps holding — predator towns go broke/extinct and never qualify.
        result = run_grounding_probe("guardian", rule="exposure", days=14,
                                     n_agents=6, seed=1)
        self.assertEqual(result.rule, "exposure")
        self.assertEqual(result.target, "lie")
        self.assertGreater(result.control_rate, 0.0,
                           "the scored behaviour actually occurs in the control world")
        self.assertEqual(result.excess, 0.0)
        self.assertEqual(result.verdict, "baseline (heuristic floor)")

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            run_grounding_probe("guardian", rule="no_such_rule", days=2)


class TestFloorRollouts(unittest.TestCase):
    """The floor is itself an estimate from a finite run's behaviour count, so
    it carries its own sampling noise — exactly what a floor-heavy world can
    smuggle into `excess`. `floor_rollouts` averages that estimate over
    several independent worlds instead of trusting a single draw."""

    def _factory(self, agent, persona, rng):
        from emergence.brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def test_default_is_a_single_draw_at_the_tested_seed(self):
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory)
        self.assertEqual(result.floor_rollouts, 1)
        self.assertEqual(result.floor_divergence_std, 0.0)

    def test_more_rollouts_widens_the_floor_sample_and_reports_a_spread(self):
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory,
                                     floor_rollouts=4)
        self.assertEqual(result.floor_rollouts, 4)
        # Four independent worlds essentially never produce byte-identical
        # deposit rates, so the ensemble should show *some* spread.
        self.assertGreaterEqual(result.floor_divergence_std, 0.0)

    def test_heuristic_only_probe_ignores_floor_rollouts(self):
        # brain_factory=None means the tested brain IS the floor; there is no
        # separate floor draw to ensemble, regardless of floor_rollouts.
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, floor_rollouts=5)
        self.assertEqual(result.floor_rollouts, 1)
        self.assertEqual(result.floor_divergence_std, 0.0)

    def test_extra_floor_worlds_never_collide_with_the_tested_seed(self):
        # A regression here would silently make "more rollouts" measure the
        # exact same world over and over — no variance reduction at all.
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory,
                                     floor_rollouts=3, floor_seed_stride=97_003)
        self.assertEqual(result.floor_rollouts, 3)


class TestPairedStatisticsOnASweep(unittest.TestCase):
    """SweepResult's paired-test fields are a harder-to-Goodhart read of the
    same per-world excess numbers `fraction_grounded` already has — not a
    second experiment."""

    def _fake_probe(self, excess_by_seed, *, inconclusive_seeds=()):
        from emergence.grounding import GroundingResult

        def probe(persona, *, rule, days, n_agents, seed, threshold,
                  brain_factory, sandbox=False, **kwargs):
            x = excess_by_seed[seed]
            if seed in inconclusive_seeds:
                return GroundingResult(
                    rule=rule, target="deposit", control_rate=0.0,
                    counterfactual_rate=0.0, divergence=0.0,
                    floor_divergence=-x, excess=x,
                    verdict="inconclusive (behaviour never occurred)",
                    days=days, n_agents=n_agents)
            return GroundingResult(
                rule=rule, target="deposit", control_rate=0.5,
                counterfactual_rate=0.5 - x, divergence=x,
                floor_divergence=0.0, excess=x,
                verdict="grounded" if x > threshold else "replay",
                days=days, n_agents=n_agents)

        return probe

    def test_consistently_positive_excess_is_significant(self):
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep(
            "guardian", seeds=(1, 2, 3, 4, 5),
            probe=self._fake_probe({1: 0.3, 2: 0.25, 3: 0.4, 4: 0.35, 5: 0.28}))
        self.assertLess(sweep.wilcoxon_p, 0.05)
        self.assertLess(sweep.sign_test_p, 0.05)
        self.assertTrue(sweep.grounded_paired)
        lo, hi = sweep.bootstrap_ci_mean_excess
        self.assertGreater(lo, 0.0, "a consistently positive sweep's CI excludes zero")

    def test_noisy_mixed_sign_excess_is_not_significant(self):
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep(
            "guardian", seeds=(1, 2, 3),
            probe=self._fake_probe({1: 0.1, 2: -0.15, 3: 0.05}))
        self.assertGreater(sweep.wilcoxon_p, 0.05)
        self.assertFalse(sweep.grounded_paired)

    def test_inconclusive_worlds_are_excluded_from_the_paired_test(self):
        # An inconclusive world's "excess" is floor noise; a huge outlier there
        # must not be able to manufacture significance either way.
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep(
            "guardian", seeds=(1, 2, 3),
            probe=self._fake_probe({1: 0.1, 2: -0.15, 3: 10.0},
                                   inconclusive_seeds=(3,)))
        # Only seeds 1, 2 (mixed sign) feed the test -- seed 3's enormous
        # "excess" must be invisible to it.
        self.assertGreater(sweep.wilcoxon_p, 0.05)

    def test_as_dict_reports_the_new_fields(self):
        from emergence.grounding import run_grounding_sweep
        sweep = run_grounding_sweep(
            "guardian", seeds=(1, 2, 3),
            probe=self._fake_probe({1: 0.3, 2: 0.25, 3: 0.4}))
        d = sweep.as_dict()
        for key in ("sign_test_p", "wilcoxon_p", "grounded_paired",
                   "bootstrap_ci_mean_excess", "floor_regression"):
            self.assertIn(key, d)


class TestFloorRegressionDiagnostic(unittest.TestCase):
    """A diagnostic immune to any *linear* floor confound, not just an
    additive one — the specific gap `excess = divergence - floor_divergence`
    leaves open (see docs/GROUNDING.md, run #6's floor confound)."""

    def test_too_few_conclusive_worlds_reports_a_note_not_a_fit(self):
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        results = [GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.5,
            counterfactual_rate=0.3, divergence=0.2, floor_divergence=0.1,
            excess=0.1, verdict="", days=10, n_agents=4) for _ in range(2)]
        d = floor_regression_diagnostic(results)
        self.assertEqual(d["n"], 2)
        self.assertIn("note", d)

    def test_a_pure_additive_floor_confound_is_fully_absorbed_by_the_intercept(self):
        # divergence = floor_divergence + 0.15 in every world: the confound is
        # exactly what `excess` already assumes (slope 1, constant offset), so
        # the fit recovers it exactly and leaves ~zero residual everywhere --
        # there is nothing left over once the (correctly additive) floor
        # relationship is accounted for.
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        floors = [-0.3, -0.1, 0.0, 0.2, 0.4]
        results = [GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.5,
            counterfactual_rate=0.3, divergence=f + 0.15, floor_divergence=f,
            excess=0.15, verdict="", days=10, n_agents=4) for f in floors]
        d = floor_regression_diagnostic(results)
        self.assertAlmostEqual(d["slope"], 1.0, places=4)
        self.assertAlmostEqual(d["intercept"], 0.15, places=4)
        for r in d["residuals"]:
            self.assertAlmostEqual(r, 0.0, places=4)

    def test_a_real_grounding_signal_survives_on_top_of_a_floor_slope_confound(self):
        # Same slope-2 floor confound as above, but now there is ALSO a real,
        # constant +0.08 grounding effect layered on top. The regression must
        # still find slope~2 and leave a residual that is consistently
        # positive and significant -- exactly the "residual orthogonal to the
        # floor, aligned with the rule" read the diagnostic is for.
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        floors = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = [GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.5,
            counterfactual_rate=0.3, divergence=2 * f + 0.08, floor_divergence=f,
            excess=f + 0.08, verdict="", days=10, n_agents=4) for f in floors]
        d = floor_regression_diagnostic(results)
        self.assertAlmostEqual(d["slope"], 2.0, places=4)
        for r in d["residuals"]:
            self.assertAlmostEqual(r, 0.0, places=4)
        # The intercept IS the real effect once the floor's slope is removed.
        self.assertAlmostEqual(d["intercept"], 0.08, places=4)

    def test_a_pure_floor_slope_confound_with_no_real_signal_has_a_flat_residual(self):
        # divergence is EXACTLY 2x floor_divergence in every world -- a brain
        # with zero real grounding whose "excess" would still look inflated in
        # floor-heavy worlds because the confound isn't additive (slope != 1).
        # The regression should fit that slope and leave ~zero residual.
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        floors = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = [GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.5,
            counterfactual_rate=0.3, divergence=2 * f, floor_divergence=f,
            excess=f, verdict="", days=10, n_agents=4) for f in floors]
        d = floor_regression_diagnostic(results)
        self.assertAlmostEqual(d["slope"], 2.0, places=4)
        for r in d["residuals"]:
            self.assertAlmostEqual(r, 0.0, places=4)

    def test_only_conclusive_worlds_feed_the_regression(self):
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        conclusive = [GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.5,
            counterfactual_rate=0.3, divergence=f + 0.1, floor_divergence=f,
            excess=0.1, verdict="", days=10, n_agents=4) for f in (-0.2, 0.0, 0.3)]
        inconclusive = GroundingResult(
            rule="demurrage", target="deposit", control_rate=0.0,
            counterfactual_rate=0.0, divergence=0.0, floor_divergence=999.0,
            excess=-999.0, verdict="inconclusive (behaviour never occurred)",
            days=10, n_agents=4)
        d = floor_regression_diagnostic(conclusive + [inconclusive])
        self.assertEqual(d["n"], 3, "the inconclusive world must not skew the fit")


if __name__ == "__main__":
    unittest.main()
