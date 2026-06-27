"""The developmental brain (NeuralDevelopmentalBrain) — opt-in, fallback-safe.

torch / llm_model_agi are optional extras not installed in CI, so here we verify
the contract that must hold *without* them: the brain imports and constructs, it
falls back to the heuristic for every decision, and — crucially — turning it on
leaves the determinism baseline byte-identical. The pure-Python reward function
is tested directly. The full learning path (stage transitions, rising competence)
runs only where the extras are installed and is exercised there, not here.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.brains._neural_reward import survival_reward
from emergence.brains.heuristic import HeuristicBrain
from emergence.brains.neural import NeuralDevelopmentalBrain
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


class TestSurvivalReward(unittest.TestCase):
    """The reward is a pure function of the observation delta — no engine, no torch."""

    def _obs(self, *, energy=50.0, money=20, reputation=0.0, others=None):
        return {"self_view": {"energy": energy, "money": money,
                              "reputation": reputation},
                "others": others or []}

    def test_gaining_energy_and_money_is_positive(self):
        prev = self._obs(energy=40, money=10)
        cur = self._obs(energy=60, money=15)
        self.assertGreater(survival_reward(prev, cur), 0.0)

    def test_decline_is_negative(self):
        prev = self._obs(energy=80, money=30, reputation=5)
        cur = self._obs(energy=50, money=20, reputation=2)
        self.assertLess(survival_reward(prev, cur), 0.0)

    def test_a_flat_transition_is_zero(self):
        o = self._obs(energy=50, money=20, reputation=3)
        self.assertAlmostEqual(survival_reward(o, dict(o)), 0.0)

    def test_reputation_drives_the_social_term(self):
        prev = self._obs(reputation=0)
        cur = self._obs(reputation=10)
        self.assertGreater(survival_reward(prev, cur), 0.0)

    def test_trust_mean_is_the_social_fallback_without_reputation(self):
        prev = {"self_view": {"energy": 50, "money": 20},
                "others": [{"trust": -0.5}, {"trust": -0.5}]}
        cur = {"self_view": {"energy": 50, "money": 20},
               "others": [{"trust": 0.5}, {"trust": 0.5}]}
        self.assertGreater(survival_reward(prev, cur), 0.0)

    def test_missing_fields_never_throw(self):
        self.assertEqual(survival_reward({}, {}), 0.0)
        self.assertEqual(survival_reward({"self_view": {}}, {"self_view": {}}), 0.0)

    def test_accepts_observation_objects_too(self):
        sim = make_simulation("guardian", n_agents=3,
                              config=SimulationConfig(seed=1))
        obs = sim._observe(sim.agents[0])
        # Same observation twice → no change → zero reward, and no exception.
        self.assertAlmostEqual(survival_reward(obs, obs), 0.0)


class TestFallbackWithoutDeps(unittest.TestCase):
    """torch / llm_model_agi are absent in CI: every decision must come from the
    heuristic understudy, and decisions must match a bare HeuristicBrain."""

    def test_construction_does_not_import_torch(self):
        # Must not raise even though the extras aren't installed.
        brain = NeuralDevelopmentalBrain("guardian")
        self.assertIsInstance(brain._fallback, HeuristicBrain)
        self.assertIsNone(brain._dev)

    def test_decide_falls_back_and_latches(self):
        sim = make_simulation("guardian", n_agents=3,
                              config=SimulationConfig(seed=1))
        agent = sim.agents[0]
        obs = sim._observe(agent)
        brain = NeuralDevelopmentalBrain("guardian")
        action = brain.decide(agent, obs)
        self.assertIsInstance(action, Action)
        self.assertTrue(brain._broken, "a failed build latches to skip retrying")

    def test_fallback_decisions_match_a_plain_heuristic(self):
        # The neural brain with no deps must be behaviourally identical to the
        # heuristic it wraps — same persona, same RNG seeding path.
        sim = make_simulation("predator", n_agents=4,
                              config=SimulationConfig(seed=7))
        agent = sim.agents[0]
        obs = sim._observe(agent)
        import random
        ref = HeuristicBrain("predator", random.Random(123))
        neu = NeuralDevelopmentalBrain("predator")
        neu._fallback = HeuristicBrain("predator", random.Random(123))
        self.assertEqual(str(neu.decide(agent, obs)), str(ref.decide(agent, obs)))


class TestLearningPathWiring(unittest.TestCase):
    """Exercise the ON path against a *fake* ``agent.adapters.emergence`` injected
    into sys.modules. This pins down the exact API the llm_model_agi side must
    provide (build_brain / DevelopmentalAgent.act+learn / to_engine_action) and
    proves decide() calls them in the right order with a reward on the 2nd turn."""

    def setUp(self):
        import sys
        import types

        self.calls = {"build": 0, "act": 0, "learn": []}
        calls = self.calls

        class _FakeDev:
            def act(self, obs, agent=None):       # contract decision (a): act(obs, agent)
                calls["act"] += 1
                calls["act_agent"] = agent
                return {"verb": "rest"}

            def learn(self, obs, reward):
                calls["learn"].append(reward)

        def build_brain(persona, teacher, ckpt):
            calls["build"] += 1
            return _FakeDev()

        def to_engine_action(spec, agent, obs):
            return Action(ActionType(spec["verb"]), {})

        # Build the fake package tree: agent -> agent.adapters -> .emergence
        pkg = types.ModuleType("agent")
        adapters = types.ModuleType("agent.adapters")
        emergence_mod = types.ModuleType("agent.adapters.emergence")
        emergence_mod.build_brain = build_brain
        emergence_mod.to_engine_action = to_engine_action
        pkg.adapters = adapters
        adapters.emergence = emergence_mod
        self._injected = {"agent": pkg, "agent.adapters": adapters,
                          "agent.adapters.emergence": emergence_mod}
        for name, mod in self._injected.items():
            sys.modules[name] = mod

    def tearDown(self):
        import sys
        for name in self._injected:
            sys.modules.pop(name, None)

    def test_decide_uses_the_dev_agent_and_learns_after_the_first_turn(self):
        sim = make_simulation("guardian", n_agents=3,
                              config=SimulationConfig(seed=1))
        agent = sim.agents[0]
        brain = NeuralDevelopmentalBrain("guardian")

        a1 = brain.decide(agent, sim._observe(agent))
        self.assertEqual(a1.type, ActionType.REST, "action came from the dev policy")
        self.assertEqual(self.calls["build"], 1)
        self.assertIs(self.calls["act_agent"], agent, "act(obs, agent) receives the agent")
        self.assertEqual(self.calls["learn"], [], "nothing to learn from on turn 1")

        agent.energy -= 10          # the world changed between turns
        brain.decide(agent, sim._observe(agent))
        self.assertEqual(len(self.calls["learn"]), 1, "turn 2 learns from turn 1")
        self.assertFalse(brain._broken, "the ON path must not latch to fallback")


class TestBaselineUntouched(unittest.TestCase):
    """The whole point of opt-in: the neural backend existing changes nothing for
    a default run. The dedicated contract test (test_baseline_contract.py) is the
    real guard; this is a fast local check that the import path is inert."""

    def _run(self):
        sim = make_simulation("guardian", n_agents=6,
                              config=SimulationConfig(seed=42, days=8))
        return sim.run().as_dict()

    def test_importing_neural_does_not_change_a_default_run(self):
        before = self._run()
        import emergence.brains.neural  # noqa: F401  (importing must be a no-op)
        after = self._run()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
