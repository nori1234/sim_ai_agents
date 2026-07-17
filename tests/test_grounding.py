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

    def test_status_layer_is_off_by_default_matching_prior_behaviour(self):
        from emergence.grounding import make_grounding_sandbox
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=4)
        self.assertFalse(sim.status.enabled)

    def test_status_can_be_switched_on_independently_of_complexity(self):
        from emergence.grounding import make_grounding_sandbox
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=4,
                                     status=True)
        self.assertTrue(sim.status.enabled)


class TestComplexityLadder(unittest.TestCase):
    """A rung ladder between the minimal sandbox and the full town (#118
    follow-up): does the world being too information-poor/predictable gate
    grounding, independent of training convergence or observation encoding?
    Each level must be a strict superset of the previous, so a regression at
    level N attributes to what's newly added there."""

    def test_level_0_is_byte_identical_to_the_original_sandbox(self):
        from emergence.grounding import make_grounding_sandbox
        from emergence.world import FacilityType
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=6,
                                     complexity_level=0)
        ftypes = {f.ftype for f in sim.world.facilities}
        self.assertEqual(ftypes, {FacilityType.BANK, FacilityType.FARM,
                                  FacilityType.HOUSE})

    def test_each_level_is_a_strict_superset_of_the_previous(self):
        from emergence.grounding import make_grounding_sandbox, MAX_COMPLEXITY_LEVEL
        prev_types = None
        for level in range(MAX_COMPLEXITY_LEVEL + 1):
            sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=4,
                                         complexity_level=level)
            types = {f.ftype for f in sim.world.facilities}
            if prev_types is not None:
                self.assertTrue(prev_types.issubset(types),
                               f"level {level} dropped facilities from level {level-1}")
                self.assertGreater(len(types), len(prev_types),
                                   f"level {level} added nothing new")
            prev_types = types

    def test_higher_levels_still_stage_a_working_bank_and_funded_savers(self):
        # The scored decision (deposit) must stay reachable and dense at every
        # level -- the ladder adds distractions, it must not break the setup.
        from emergence.grounding import make_grounding_sandbox
        from emergence.world import FacilityType
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=12,
                                     complexity_level=3)
        bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
        self.assertEqual(sim.agents[0].pos, bank.pos)
        self.assertTrue(all(s.money > 0 for s in sim.agents[1:]))
        sim.run()
        deposits = sum(1 for e in sim.world.events if e["kind"] == "deposit")
        self.assertGreater(deposits, 0, "depositing must stay exercised at higher levels")

    def test_out_of_range_level_is_rejected(self):
        from emergence.grounding import make_grounding_sandbox, MAX_COMPLEXITY_LEVEL
        with self.assertRaises(ValueError):
            make_grounding_sandbox("guardian", complexity_level=MAX_COMPLEXITY_LEVEL + 1)
        with self.assertRaises(ValueError):
            make_grounding_sandbox("guardian", complexity_level=-1)

    def test_probe_and_preflight_accept_a_complexity_level(self):
        from emergence.grounding import estimate_conclusive_yield
        result = run_grounding_probe("guardian", sandbox=True, days=8, n_agents=4,
                                     seed=1, complexity_level=2)
        self.assertEqual(result.target, "deposit")
        yields = estimate_conclusive_yield(
            "guardian", rules=("demurrage",), seeds=(1, 2), days=8, n_agents=4,
            sandbox=True, complexity_level=2)
        self.assertIn("demurrage", yields)


class TestConclusiveYieldPreflight(unittest.TestCase):
    """A cheap, brain-free estimate of how many worlds will be conclusive per
    rule, so a rule whose scored behaviour is too sparse to ever power
    floor_regression can be caught BEFORE a training run's compute is spent,
    not after (the run #6 aftermath's practical risk: 20 worlds burned on a
    rule that was structurally never going to reach n_conclusive >= 6)."""

    def test_dense_sandbox_behaviour_is_conclusive_in_every_seed(self):
        from emergence.grounding import estimate_conclusive_yield
        yields = estimate_conclusive_yield(
            "guardian", rules=("demurrage",), seeds=(1, 2, 3, 4),
            days=10, n_agents=4, sandbox=True)
        self.assertEqual(yields["demurrage"]["n_conclusive"], 4)
        self.assertEqual(yields["demurrage"]["n_seeds"], 4)

    def test_reports_every_requested_rule(self):
        from emergence.grounding import estimate_conclusive_yield
        yields = estimate_conclusive_yield(
            "guardian", rules=("demurrage", "vanity"), seeds=(1, 2),
            days=4, n_agents=4)
        self.assertEqual(set(yields), {"demurrage", "vanity"})
        for rule_yield in yields.values():
            self.assertLessEqual(rule_yield["n_conclusive"], rule_yield["n_seeds"])

    def test_uses_the_heuristic_only_no_brain_factory_needed(self):
        # This must run with zero brain_factory, i.e. before any trained
        # checkpoint exists -- the whole point of a preflight check.
        from emergence.grounding import estimate_conclusive_yield
        yields = estimate_conclusive_yield(
            "guardian", rules=("demurrage",), seeds=(1,), days=6, n_agents=4,
            sandbox=True)
        self.assertIn("demurrage", yields)


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
    """`floor_divergence`/`excess` are always the WORLD-MATCHED heuristic floor
    (the seed under test) — the statistically correct control in a
    deterministic engine, where a single seed's floor is an exact number, not a
    noisy estimate to average away. Averaging the floor across *other* worlds
    would swap the confound being controlled for from "this world's mechanical
    strength" to "the population's average", not remove it (this was flagged
    against an earlier version of this instrument that got this wrong).
    `floor_rollouts` therefore computes only an ADDITIONAL, side-by-side
    ensemble read (`ensemble_floor_divergence`/`ensemble_excess`) for
    cross-checking — it must never move the canonical fields."""

    def _factory(self, agent, persona, rng):
        from emergence.brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def test_default_is_a_single_draw_at_the_tested_seed(self):
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory)
        self.assertEqual(result.floor_rollouts, 1)
        self.assertEqual(result.floor_divergence_std, 0.0)
        self.assertIsNone(result.ensemble_floor_divergence)
        self.assertIsNone(result.ensemble_excess)

    def test_the_canonical_floor_and_excess_never_move_when_rollouts_are_requested(self):
        # This is the load-bearing property: asking for an ensemble read must
        # not change the world-matched floor/excess that the verdict, and
        # fraction_grounded/sign_test_p/wilcoxon_p, are computed from.
        kwargs = dict(persona="guardian", sandbox=True, days=10, n_agents=4,
                      seed=1, brain_factory=self._factory)
        baseline = run_grounding_probe(**kwargs)
        ensembled = run_grounding_probe(floor_rollouts=6, **kwargs)
        self.assertEqual(baseline.floor_divergence, ensembled.floor_divergence)
        self.assertEqual(baseline.excess, ensembled.excess)
        self.assertEqual(baseline.verdict, ensembled.verdict)

    def test_more_rollouts_reports_a_separate_ensemble_read(self):
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory,
                                     floor_rollouts=4)
        self.assertEqual(result.floor_rollouts, 4)
        self.assertIsNotNone(result.ensemble_floor_divergence)
        self.assertAlmostEqual(result.ensemble_excess,
                               result.divergence - result.ensemble_floor_divergence)
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
        self.assertIsNone(result.ensemble_floor_divergence)

    def test_extra_floor_worlds_never_collide_with_the_tested_seed(self):
        # A regression here would silently make "more rollouts" measure the
        # exact same world over and over — no variance reduction at all.
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory,
                                     floor_rollouts=3, floor_seed_stride=97_003)
        self.assertEqual(result.floor_rollouts, 3)


class TestRawAttemptCounts(unittest.TestCase):
    """control_count/counterfactual_count: the raw (unnormalised) companion to
    control_rate/counterfactual_rate -- answers "how many times did it actually
    try this" without reconstructing it from a per-agent-day rate."""

    @staticmethod
    def _factory(agent, persona, rng):
        from emergence.brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def test_counts_are_populated_and_agree_in_sign_with_the_rate(self):
        # The exact agent_days divisor (behaviour_rate) depends on the sim's
        # OBSERVED population/days_run, not the configured n_agents/days (a
        # world can end early) -- so this checks the count/rate agree on
        # whether the behaviour happened at all, not an exact ratio.
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory)
        self.assertIsInstance(result.control_count, int)
        self.assertIsInstance(result.counterfactual_count, int)
        self.assertGreaterEqual(result.control_count, 0)
        self.assertGreaterEqual(result.counterfactual_count, 0)
        self.assertEqual(result.control_count > 0, result.control_rate > 0)
        self.assertEqual(result.counterfactual_count > 0, result.counterfactual_rate > 0)

    def test_counts_appear_in_as_dict(self):
        result = run_grounding_probe("guardian", sandbox=True, days=10,
                                     n_agents=4, seed=1, brain_factory=self._factory)
        d = result.as_dict()
        self.assertEqual(d["control_count"], result.control_count)
        self.assertEqual(d["counterfactual_count"], result.counterfactual_count)

    def test_hand_built_results_default_counts_to_none(self):
        # GroundingResult instances built directly (as many tests do) rather
        # than via run_grounding_probe shouldn't be forced to supply counts.
        result = GroundingResult(
            rule="demurrage", target="deposit", control_rate=1.0,
            counterfactual_rate=0.5, divergence=0.5, floor_divergence=0.1,
            excess=0.4, verdict="grounded (exceeds heuristic floor)",
            days=10, n_agents=4)
        self.assertIsNone(result.control_count)
        self.assertIsNone(result.counterfactual_count)


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

    def test_a_narrow_floor_spread_that_clears_the_std_bar_can_still_be_unidentified(self):
        # run #7's real `exposure` numbers (20 held-out worlds, full town):
        # n=20 and floor_spread_std=0.0148 clear the mechanical min_conclusive/
        # min_floor_spread bars, but the residual noise is large relative to
        # that spread, so the fitted slope's bootstrap CI spans both signs
        # (essentially unidentified) -- exactly the case the brain team flagged
        # in review: floor_spread_std is necessary but not sufficient for
        # `powered`, so the slope CI itself must gate it directly.
        from emergence.grounding import GroundingResult, floor_regression_diagnostic
        floors = [-0.0083, -0.0333, -0.0333, -0.0333, 0.0083, 0.0, 0.0167,
                 -0.0083, -0.0167, 0.0, -0.0083, 0.0, 0.0, 0.0083, 0.0,
                 -0.0083, 0.0083, -0.025, -0.025, 0.0]
        divs = [0.0417, 0.05, -0.15, -0.0333, -0.2083, -0.0833, 0.0333,
               0.0417, 0.0417, 0.0917, 0.2083, 0.0083, -0.175, 0.1667, 0.0,
               -0.25, -0.0833, -0.1333, 0.1667, -0.1083]
        results = [GroundingResult(
            rule="exposure", target="lie", control_rate=0.5, counterfactual_rate=0.3,
            divergence=d, floor_divergence=f, excess=d - f, verdict="",
            days=10, n_agents=4) for f, d in zip(floors, divs)]
        out = floor_regression_diagnostic(results)
        self.assertEqual(out["n"], 20)
        self.assertGreater(out["floor_spread_std"], 0.01, "clears the spread bar alone")
        self.assertGreater(out["slope_ci"][1] - out["slope_ci"][0], 3.0,
                           "the slope CI should be wide/unidentified for this data")
        self.assertFalse(out["powered"],
                         "a wide slope CI must veto `powered` even though n and "
                         "floor_spread_std both individually cleared their bars")

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


class TestFloorRegressionAsATiebreaker(unittest.TestCase):
    """The floor-regression verdict (SweepResult.floor_regression_grounded,
    BatteryResult.replay_inexplicable_floor_regression) — promoted to a
    co-primary/tiebreaker read alongside grounded_paired/replay_inexplicable_
    paired, since it is immune to a floor confound of any linear form
    regardless of which floor convention (world-matched vs. ensemble) the
    excess-based statistics used."""

    def _sweep_with_residuals(self, rule, floors, extra):
        from emergence.grounding import GroundingResult, run_grounding_sweep

        results_by_seed = {i: (f, e) for i, (f, e) in enumerate(zip(floors, extra))}

        def probe(persona, *, rule, days, n_agents, seed, threshold,
                  brain_factory, sandbox=False, **kwargs):
            f, e = results_by_seed[seed]
            return GroundingResult(
                rule=rule, target="deposit", control_rate=0.5,
                counterfactual_rate=0.5 - (f + e), divergence=f + e,
                floor_divergence=f, excess=e, verdict="",
                days=days, n_agents=n_agents)

        return run_grounding_sweep(
            "guardian", rule=rule, seeds=tuple(range(len(floors))),
            probe=probe)

    def test_significant_residual_is_floor_regression_grounded(self):
        # OLS residuals always sum to exactly zero (an intercept absorbs any
        # constant shift) -- so "significant" here necessarily means a SKEWED
        # residual (most worlds a bit above zero, balanced by one or two well
        # below), not a case where every single world individually clears a
        # threshold. These floors/extra are constructed (via projecting a
        # skewed target orthogonal to [1, floor], then scaling the residual
        # small -- p-values are scale-invariant, but a small residual keeps
        # the fitted slope's bootstrap CI narrow) specifically to produce that
        # shape while staying `powered` (n>=6, floor spread and slope CI width
        # both comfortably inside their thresholds) -- not meant to look like
        # a plausible battery result, just to exercise a test a naive
        # per-world threshold check would miss.
        floors = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        extra = [0.163636, 0.160606, 0.157576, 0.154545, -0.348485,
                0.148485, 0.145455, 0.142424, 0.139394, 0.136364]
        sweep = self._sweep_with_residuals("demurrage", floors, extra)
        self.assertTrue(sweep.floor_regression["powered"],
                        f"expected a narrow, identified slope: {sweep.floor_regression}")
        self.assertTrue(sweep.floor_regression_grounded)

    def test_no_residual_signal_is_not_floor_regression_grounded(self):
        # n=6, floor spread ~0.29 -- comfortably powered, so a False here is a
        # real "not significant" verdict, not an underpowered None in disguise
        # (assertFalse(None) would also pass, which is why this specifically
        # asserts `is False`, not just falsy).
        floors = [-0.4, -0.3, -0.1, 0.1, 0.3, 0.4]
        extra = [0.10, -0.20, 0.15, -0.10, 0.05, -0.05]  # no consistent direction
        sweep = self._sweep_with_residuals("demurrage", floors, extra)
        self.assertTrue(sweep.floor_regression["powered"])
        self.assertIs(sweep.floor_regression_grounded, False)

    def test_too_few_conclusive_worlds_is_undetermined_not_false(self):
        from emergence.grounding import SweepResult
        sweep = SweepResult(rule="demurrage", results=[], seeds=(),
                            mean_excess=0.0, min_excess=0.0, n_grounded=0,
                            n_worlds=0, n_conclusive=0,
                            floor_regression={"n": 2, "note": "too few..."})
        self.assertIsNone(sweep.floor_regression_grounded)

    def test_battery_conjunction_is_none_when_any_rule_is_undetermined(self):
        from emergence.grounding import SweepResult, run_grounding_battery

        def sweep(persona, *, rule, seeds, days, n_agents, threshold,
                  brain_factory, sandbox=False, **kwargs):
            fr = ({"note": "too few"} if rule == "vanity"
                 else {"residual_wilcoxon_p": 0.01, "powered": True})
            return SweepResult(rule=rule, results=[], seeds=tuple(seeds),
                               mean_excess=0.2, min_excess=0.1, n_grounded=len(seeds),
                               n_worlds=len(seeds), n_conclusive=len(seeds),
                               floor_regression=fr)

        battery = run_grounding_battery("guardian", seeds=(1, 2), sweep=sweep)
        self.assertIsNone(battery.replay_inexplicable_floor_regression)

    def test_battery_conjunction_is_none_when_a_rule_is_unpowered_even_with_a_low_p(self):
        # An underpowered fit's p-value must not count as "significant" just
        # because it happens to be small -- it's not trustworthy evidence.
        from emergence.grounding import SweepResult, run_grounding_battery

        def sweep(persona, *, rule, seeds, days, n_agents, threshold,
                  brain_factory, sandbox=False, **kwargs):
            fr = {"residual_wilcoxon_p": 0.001, "powered": rule != "vanity"}
            return SweepResult(rule=rule, results=[], seeds=tuple(seeds),
                               mean_excess=0.2, min_excess=0.1, n_grounded=len(seeds),
                               n_worlds=len(seeds), n_conclusive=len(seeds),
                               floor_regression=fr)

        battery = run_grounding_battery("guardian", seeds=(1, 2), sweep=sweep)
        self.assertIsNone(battery.replay_inexplicable_floor_regression)

    def test_battery_conjunction_is_true_only_when_every_rule_is_significant(self):
        from emergence.grounding import SweepResult, run_grounding_battery

        def sweep(persona, *, rule, seeds, days, n_agents, threshold,
                  brain_factory, sandbox=False, **kwargs):
            p = 0.30 if rule == "exposure" else 0.01
            return SweepResult(rule=rule, results=[], seeds=tuple(seeds),
                               mean_excess=0.2, min_excess=0.1, n_grounded=len(seeds),
                               n_worlds=len(seeds), n_conclusive=len(seeds),
                               floor_regression={"residual_wilcoxon_p": p, "powered": True})

        battery = run_grounding_battery("guardian", seeds=(1, 2), sweep=sweep)
        self.assertFalse(battery.replay_inexplicable_floor_regression)


class TestGroundedConfirmedAndGate(unittest.TestCase):
    """The pre-registered verdict is a strict AND gate, not a tiebreaker: a
    single test disagreeing withholds "grounded" (2026-07 review — an earlier
    draft's wording implied floor_regression could override a failing
    grounded_paired, which contradicted the AND semantics actually coded)."""

    def _sweep(self, *, wilcoxon_p, fr_p, fr_powered=True):
        from emergence.grounding import SweepResult
        fr = {"powered": fr_powered}
        if fr_powered:
            fr["residual_wilcoxon_p"] = fr_p
        else:
            fr["note"] = "underpowered"
        return SweepResult(rule="demurrage", results=[], seeds=(1, 2),
                           mean_excess=0.2, min_excess=0.1, n_grounded=2,
                           n_worlds=2, n_conclusive=2,
                           wilcoxon_p=wilcoxon_p, floor_regression=fr)

    def test_both_significant_is_confirmed(self):
        sweep = self._sweep(wilcoxon_p=0.01, fr_p=0.01)
        self.assertTrue(sweep.grounded_paired)
        self.assertTrue(sweep.floor_regression_grounded)
        self.assertTrue(sweep.grounded_confirmed)

    def test_paired_significant_but_regression_not_is_NOT_confirmed(self):
        # This is exactly the quadrant the AND gate exists for: a floor
        # confound that grounded_paired alone would miss must be able to veto.
        sweep = self._sweep(wilcoxon_p=0.01, fr_p=0.30)
        self.assertTrue(sweep.grounded_paired)
        self.assertFalse(sweep.floor_regression_grounded)
        self.assertFalse(sweep.grounded_confirmed)

    def test_regression_significant_but_paired_not_is_NOT_confirmed(self):
        # The other quadrant: floor_regression is not a unilateral arbiter
        # either -- it needs grounded_paired's agreement too.
        sweep = self._sweep(wilcoxon_p=0.30, fr_p=0.01)
        self.assertFalse(sweep.grounded_paired)
        self.assertTrue(sweep.floor_regression_grounded)
        self.assertFalse(sweep.grounded_confirmed)

    def test_neither_significant_is_not_confirmed(self):
        sweep = self._sweep(wilcoxon_p=0.30, fr_p=0.30)
        self.assertFalse(sweep.grounded_confirmed)

    def test_underpowered_regression_makes_confirmed_undetermined(self):
        # Even if grounded_paired is significant, an underpowered regression
        # can't confirm OR deny -- the AND gate can't be evaluated.
        sweep = self._sweep(wilcoxon_p=0.01, fr_p=0.01, fr_powered=False)
        self.assertTrue(sweep.grounded_paired)
        self.assertIsNone(sweep.floor_regression_grounded)
        self.assertIsNone(sweep.grounded_confirmed)

    def test_battery_confirmed_requires_every_rule_confirmed(self):
        from emergence.grounding import SweepResult, run_grounding_battery

        def sweep(persona, *, rule, seeds, days, n_agents, threshold,
                  brain_factory, sandbox=False, **kwargs):
            wilcoxon_p = 0.30 if rule == "vanity" else 0.01
            return SweepResult(
                rule=rule, results=[], seeds=tuple(seeds), mean_excess=0.2,
                min_excess=0.1, n_grounded=len(seeds), n_worlds=len(seeds),
                n_conclusive=len(seeds), wilcoxon_p=wilcoxon_p,
                floor_regression={"residual_wilcoxon_p": 0.01, "powered": True})

        battery = run_grounding_battery("guardian", seeds=(1, 2), sweep=sweep)
        self.assertFalse(battery.replay_inexplicable_confirmed,
                         "vanity's grounded_paired failed, so the AND gate must fail overall")


class TestRewardCeiling(unittest.TestCase):
    """measure_reward_ceiling: does the TASK pay enough for grounding to be
    worth learning, independent of whether any policy currently learns it."""

    def test_advantage_control_is_exactly_zero(self):
        # The grounded oracle only diverges from the blind heuristic when
        # avoid_deposit is True (i.e. in the counterfactual world) -- in
        # control it must be behaviourally IDENTICAL, so any nonzero
        # advantage_control would mean a bug in the oracle, not a finding.
        from emergence.grounding import measure_reward_ceiling
        result = measure_reward_ceiling("guardian", seeds=(42, 43, 44), days=10, n_agents=4)
        self.assertEqual(result.advantage_control, 0.0)
        self.assertEqual(result.blind_return_control, result.grounded_return_control)

    def test_as_dict_round_trips_the_key_fields(self):
        from emergence.grounding import measure_reward_ceiling
        result = measure_reward_ceiling("guardian", seeds=(42,), days=10, n_agents=4)
        d = result.as_dict()
        self.assertEqual(d["rule"], "demurrage")
        self.assertEqual(d["n_worlds"], 1)
        self.assertAlmostEqual(d["advantage_control"],
                               d["grounded_return_control"] - d["blind_return_control"],
                               places=4)
        self.assertAlmostEqual(d["advantage_counterfactual"],
                               d["grounded_return_counterfactual"]
                               - d["blind_return_counterfactual"], places=4)

    def test_only_demurrage_is_supported(self):
        from emergence.grounding import measure_reward_ceiling
        with self.assertRaises(ValueError):
            measure_reward_ceiling("guardian", rule="vanity", seeds=(42,))

    def test_grounded_oracle_never_starves_avoiding_deposit(self):
        # Regression test for a real bug caught while building this: an
        # earlier oracle returned None instead of REST when avoiding a
        # deposit, which fell through HeuristicBrain.decide() into
        # _trade_action's untested market-primitive loops (offer/accept/
        # repay with no facility to ground them in this minimal sandbox)
        # and starved the agent to death by day 5 -- a confound entirely
        # unrelated to the demurrage regime being measured.
        from emergence.grounding import make_grounding_sandbox, _grounded_heuristic_brain_class
        GroundedHeuristicBrain = _grounded_heuristic_brain_class()

        def factory(agent, persona, rng):
            return GroundedHeuristicBrain(persona, rng, avoid_deposit=True)

        sim = make_grounding_sandbox("guardian", rule="demurrage", n_savers=5,
                                     seed=42, days=20, cf_enabled=True,
                                     brain_factory=factory)
        agent = sim.agents[1]
        sim.run()
        self.assertTrue(agent.alive)


class TestDepositOracle(unittest.TestCase):
    """measure_deposit_oracle (S6): the brain team's clean-spec oracle --
    blind everywhere, except a DEPOSIT decision is dropped (cash held) in
    the counterfactual world, falling through to blind's own next branch."""

    def _bank_obs(self, money_ctx="rich"):
        # A minimal observation that drives HeuristicBrain._bank_action into
        # its deposit branch (bank_here set, no banker ambitions, surplus).
        from emergence.observation import Observation
        return Observation(
            day=1, tick=1, self_view={}, position=(0, 0), nearby_facilities=[],
            here=None, others=[], open_proposals=[], granary_food=0,
            recent_events=[],
            economy={"enabled": True, "bank_here": "b1", "my_deposits":
                     [{"bank": "b1", "amount": 6}]})

    def _agent(self, money):
        from emergence.agent import Agent
        return Agent(id="x", name="X", profession="farmer", persona="guardian",
                     x=0, y=0, money=money, energy=30.0)  # low energy: no banker setup

    def test_oracle_drops_exactly_the_deposit_decision(self):
        from emergence.actions import ActionType
        from emergence.brains.heuristic import HeuristicBrain
        from emergence.grounding import _deposit_only_oracle_brain_class
        Oracle = _deposit_only_oracle_brain_class()
        obs = self._bank_obs()
        rich = self._agent(money=20)
        blind_act = HeuristicBrain("guardian")._bank_action(rich, obs)
        self.assertEqual(blind_act.type, ActionType.DEPOSIT, "premise: blind deposits")
        self.assertIsNone(
            Oracle("guardian", skip_deposit=True)._bank_action(rich, obs),
            "the oracle drops the deposit and falls through")
        kept = Oracle("guardian", skip_deposit=False)._bank_action(rich, obs)
        self.assertEqual(kept.type, ActionType.DEPOSIT,
                         "skip off -> byte-identical to blind")

    def test_oracle_keeps_every_non_deposit_branch(self):
        from emergence.actions import ActionType
        from emergence.brains.heuristic import HeuristicBrain
        from emergence.grounding import _deposit_only_oracle_brain_class
        Oracle = _deposit_only_oracle_brain_class()
        obs = self._bank_obs()
        poor = self._agent(money=2)  # < 4 with a deposit at this bank -> WITHDRAW
        blind_act = HeuristicBrain("guardian")._bank_action(poor, obs)
        self.assertEqual(blind_act.type, ActionType.WITHDRAW, "premise: blind withdraws")
        oracle_act = Oracle("guardian", skip_deposit=True)._bank_action(poor, obs)
        self.assertEqual(oracle_act.type, ActionType.WITHDRAW,
                         "withdraw (and every other branch) is untouched")
        self.assertEqual(oracle_act.params, blind_act.params)

    def test_advantage_control_is_exactly_zero(self):
        # In the control world skip_deposit is False, so the oracle IS the
        # blind heuristic -- per-world returns must match exactly, not on
        # average.
        from emergence.grounding import measure_deposit_oracle
        result = measure_deposit_oracle("guardian", seeds=(42, 43, 44),
                                        days=10, n_agents=4)
        self.assertEqual(result.advantage_control, 0.0)
        for w in result.per_world:
            self.assertEqual(w["blind_control"], w["oracle_control"])

    def test_deposit_weight_defaults_to_one_and_is_byte_identical(self):
        # The calibration dial defaults to 1.0 and, at 1.0, must reproduce the
        # canonical S6 numbers exactly -- the knob is inert unless moved.
        from emergence.grounding import measure_deposit_oracle
        default = measure_deposit_oracle("guardian", seeds=(42, 43, 44),
                                         days=10, n_agents=4)
        explicit = measure_deposit_oracle("guardian", seeds=(42, 43, 44),
                                          days=10, n_agents=4,
                                          deposit_wealth_weight=1.0)
        self.assertEqual(default.deposit_wealth_weight, 1.0)
        self.assertEqual(default.as_dict()["deposit_wealth_weight"], 1.0)
        self.assertEqual([w["blind_cf"] for w in default.per_world],
                         [w["blind_cf"] for w in explicit.per_world])
        self.assertEqual(default.advantage_counterfactual,
                         explicit.advantage_counterfactual)

    def test_lower_deposit_weight_raises_the_counterfactual_advantage(self):
        # Down-weighting banked coin (lever 2) makes depositing pay less, so the
        # oracle-that-holds gains on the blind-that-deposits: advantage_cf rises
        # monotonically as the weight drops. Behaviour (hence the control world
        # and the death count) is untouched -- only the reward accounting moves.
        from emergence.grounding import measure_deposit_oracle
        seeds = tuple(range(42, 52))
        highs = measure_deposit_oracle("guardian", seeds=seeds, days=12,
                                       n_agents=4, deposit_wealth_weight=1.0)
        lows = measure_deposit_oracle("guardian", seeds=seeds, days=12,
                                      n_agents=4, deposit_wealth_weight=0.2)
        self.assertGreater(lows.advantage_counterfactual,
                           highs.advantage_counterfactual)
        # Re-scoring fixed trajectories: the control sanity check stays exactly
        # zero and the oracle's cf deaths are identical at both weights.
        self.assertEqual(lows.advantage_control, 0.0)
        self.assertEqual(highs.advantage_control, 0.0)
        self.assertEqual(lows.oracle_cf_deaths, highs.oracle_cf_deaths)

    def test_sole_banker_defaults_off_and_is_byte_identical(self):
        # The task-redesign switch defaults to False and, off, must reproduce
        # the original sandbox's numbers exactly.
        from emergence.grounding import measure_deposit_oracle
        default = measure_deposit_oracle("guardian", seeds=(42, 43), days=10,
                                         n_agents=4)
        explicit = measure_deposit_oracle("guardian", seeds=(42, 43), days=10,
                                          n_agents=4, sole_banker=False)
        self.assertFalse(default.sole_banker)
        self.assertFalse(default.as_dict()["sole_banker"])
        self.assertEqual([w for w in default.per_world],
                         [w for w in explicit.per_world])

    def test_sole_banker_crosses_the_sign(self):
        # The redesigned task: with agent-to-agent deposit chains cut, the
        # deposit-only oracle's counterfactual advantage turns positive -- the
        # property the whole redesign exists to deliver -- while the control
        # sanity check stays exactly zero. Full battery worlds, because the
        # sign claim is about the standard 20-world measurement, not a subset.
        from emergence.grounding import measure_deposit_oracle
        result = measure_deposit_oracle("guardian", sole_banker=True)
        self.assertTrue(result.sole_banker)
        self.assertEqual(result.advantage_control, 0.0)
        self.assertGreater(result.advantage_counterfactual, 0.0)

    def test_sole_banker_refuses_other_counterparties(self):
        # A brain that names a non-banker counterparty directly (bypassing
        # _banker_near) is refused: no coin moves, no claim is created.
        from emergence.grounding import make_grounding_sandbox
        from emergence.actions import Action, ActionType
        sim = make_grounding_sandbox("guardian", n_savers=3, seed=1, days=5,
                                     sole_banker=True)
        banker, saver, other, *_ = sim.agents
        before_money = saver.money
        sim._do_deposit(saver, Action(ActionType.DEPOSIT,
                                      {"bank": other.id, "amount": 10}))
        self.assertEqual(saver.money, before_money)
        self.assertFalse(any(d.holder == saver.id for d in sim.deposits))
        # ... while the staffed banker still accepts.
        sim._do_deposit(saver, Action(ActionType.DEPOSIT,
                                      {"bank": banker.id, "amount": 10}))
        self.assertTrue(any(d.holder == saver.id and d.amount == 10
                            for d in sim.deposits))

    def test_probe_monitor_and_battery_accept_sole_banker(self):
        # The training/eval chain (probe -> sweep -> battery, plus the
        # monitor) forwards sole_banker to the sandbox, so #130's runner can
        # train AND measure on the redesigned task. Heuristic floor: excess
        # is 0 by construction; this asserts plumbing, not a verdict.
        from emergence.grounding import run_grounding_probe, run_grounding_battery
        from emergence.grounding_monitor import GroundingMonitor
        result = run_grounding_probe("guardian", sandbox=True, sole_banker=True,
                                     days=8, n_agents=4, seed=1)
        self.assertEqual(result.excess, 0.0)
        battery = run_grounding_battery("guardian", rules=("demurrage",),
                                        seeds=(1, 2), days=8, n_agents=4,
                                        sandbox=True, sole_banker=True)
        self.assertIn("demurrage", battery.as_dict()["rules"])
        mon = GroundingMonitor("guardian", sandbox=True, sole_banker=True,
                               days=8, n_agents=4, seed=1)
        mon.probe(0, None)
        self.assertEqual(len(mon.history), 1)

    def test_sole_banker_keeps_the_deposit_decision_dense(self):
        # Cutting the chain must not starve the sandbox of its scored decision:
        # savers still deposit (control keeps interest income flowing, so the
        # choice recurs rather than firing once).
        from emergence.grounding import make_grounding_sandbox
        for cf in (False, True):
            sim = make_grounding_sandbox("guardian", n_savers=5, seed=42,
                                         days=20, cf_enabled=cf,
                                         sole_banker=True)
            sim.run()
            deposits = sum(1 for e in sim.world.events
                           if e["kind"] == "deposit")
            self.assertGreater(deposits, 5,
                               f"deposit decision went sparse (cf={cf})")

    def test_as_dict_carries_per_world_and_variance(self):
        from emergence.grounding import measure_deposit_oracle
        result = measure_deposit_oracle("guardian", seeds=(42, 43), days=10,
                                        n_agents=4)
        d = result.as_dict()
        self.assertEqual(d["n_worlds"], 2)
        self.assertEqual(len(d["per_world"]), 2)
        for w in d["per_world"]:
            for key in ("seed", "blind_control", "blind_cf", "oracle_control",
                        "oracle_cf", "blind_cf_alive", "oracle_cf_alive",
                        "oracle_cf_day_of_death"):
                self.assertIn(key, w)
        self.assertGreaterEqual(d["blind_cf_variance"], 0.0)
        # Compare unrounded properties (as_dict rounds each independently, so
        # squaring the rounded std need not equal the rounded variance).
        self.assertAlmostEqual(result.blind_cf_std ** 2, result.blind_cf_variance,
                               places=6)
        self.assertAlmostEqual(
            d["advantage_counterfactual"],
            d["oracle_return_counterfactual"] - d["blind_return_counterfactual"],
            places=4)

    def test_effect_size_is_advantage_over_blind_cf_std(self):
        from emergence.grounding import measure_deposit_oracle
        result = measure_deposit_oracle("guardian", seeds=(42, 43, 44),
                                        days=10, n_agents=4)
        if result.blind_cf_std:
            self.assertAlmostEqual(
                result.effect_size,
                result.advantage_counterfactual / result.blind_cf_std, places=6)

    def test_only_demurrage_is_supported(self):
        from emergence.grounding import measure_deposit_oracle
        with self.assertRaises(ValueError):
            measure_deposit_oracle("guardian", rule="vanity", seeds=(42,))


class TestTeacherAgreement(unittest.TestCase):
    """measure_teacher_agreement: an external, engine-side proxy for how
    BC-anchored a tested policy still is to the blind teacher, independent
    of any internal training diagnostic (teacher_frac_in_batch)."""

    @staticmethod
    def _blind_factory(agent, persona, rng):
        from emergence.brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def test_the_blind_heuristic_agrees_with_its_own_shadow_perfectly(self):
        # The load-bearing sanity check: tested against itself, agreement
        # must be exactly 1.0 in both regimes -- anything less would mean
        # the rng-desync bug this instrument was built to avoid (two
        # decide() calls consuming a single shared random.Random stream)
        # had crept back in, not a real behavioural difference.
        from emergence.grounding import measure_teacher_agreement
        result = measure_teacher_agreement(
            "guardian", seeds=(42, 43, 44), days=10, n_agents=4,
            brain_factory=self._blind_factory)
        self.assertEqual(result.agreement_control, 1.0)
        self.assertEqual(result.agreement_counterfactual, 1.0)
        self.assertEqual(result.agreement_gap, 0.0)

    def test_a_policy_reacting_to_observed_demurrage_disagrees_more_under_cf(self):
        # The instrument's actual discriminating power. measure_teacher_agreement's
        # brain_factory signature is (agent, persona, rng) -- deliberately the
        # same as every other brain_factory in this module, so it can't be
        # handed the ground-truth regime the way _grounded_heuristic_brain_class's
        # oracle is (that would defeat the point: a real trained checkpoint
        # never gets told the regime either). So this brain reacts to the
        # actual OBSERVABLE evidence instead -- the "N coin ... vanished"
        # memory entry demurrage writes (see emergence/simulation.py's
        # _apply_demurrage) -- which is the same signal a genuinely grounded
        # policy would have to key off. No memory entry in control -> behaves
        # exactly like the blind teacher; sees it under counterfactual ->
        # stops depositing.
        from emergence.actions import Action, ActionType
        from emergence.brains.heuristic import HeuristicBrain
        from emergence.grounding import measure_teacher_agreement

        class ReactiveBrain(HeuristicBrain):
            def _bank_action(self, agent, obs):
                if not any("vanished" in m for m in obs.memory):
                    return super()._bank_action(agent, obs)
                ec = obs.economy
                bh = ec.get("bank_here")
                if not bh:
                    return None
                if agent.money < 4:
                    deps = ec.get("my_deposits") or []
                    d = next((d for d in deps if d.get("bank") == bh), None)
                    if d:
                        return Action(ActionType.WITHDRAW,
                                      {"bank": bh, "amount": d["amount"]},
                                      rationale="withdraw savings")
                if agent.money >= 12:
                    return Action(ActionType.REST,
                                  rationale="saw coin vanish -- hold cash instead")
                return None

        def reactive_factory(agent, persona, rng):
            return ReactiveBrain(persona, rng)

        result = measure_teacher_agreement(
            "guardian", seeds=(42, 43, 44), days=10, n_agents=4,
            brain_factory=reactive_factory)
        self.assertEqual(result.agreement_control, 1.0)
        self.assertLess(result.agreement_counterfactual, 1.0)
        self.assertGreater(result.agreement_gap, 0.0)

    def test_as_dict_round_trips_the_key_fields(self):
        from emergence.grounding import measure_teacher_agreement
        result = measure_teacher_agreement(
            "guardian", seeds=(42,), days=10, n_agents=4,
            brain_factory=self._blind_factory)
        d = result.as_dict()
        self.assertEqual(d["rule"], "demurrage")
        self.assertEqual(d["n_worlds"], 1)
        self.assertAlmostEqual(d["agreement_gap"],
                               d["agreement_control"] - d["agreement_counterfactual"],
                               places=4)

    def test_only_demurrage_is_supported(self):
        from emergence.grounding import measure_teacher_agreement
        with self.assertRaises(ValueError):
            measure_teacher_agreement("guardian", rule="vanity", seeds=(42,),
                                      brain_factory=self._blind_factory)


if __name__ == "__main__":
    unittest.main()
