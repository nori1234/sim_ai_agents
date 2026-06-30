"""The engine ⇄ developmental-brain contract (see emergence/brains/neural_contract.py).

Two layers:

* **Drift guards** (always run): the contract constants must stay in lock-step
  with the engine. If someone adds an ActionType, an Observation field, or a
  snapshot key without updating the contract (and bumping its version), these
  fail here — loudly — instead of the external adapter breaking silently at
  integration time. This is the answer to risk ①.

* **Round-trip integration** (skipped unless the ``[neural]`` extra is installed,
  i.e. ``agent.adapters.emergence`` is importable): the adapter the
  ``llm_model_agi`` side provides must turn a real observation into a valid,
  applicable engine Action and accept a reward. This is the answer to risk ② —
  proving the neural path actually works rather than quietly falling back to the
  heuristic. The external team should make this pass.
"""

import importlib.util
import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.brains import neural_contract as C
from emergence.observation import Observation
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

def _has_neural() -> bool:
    # find_spec raises (not returns None) when a *parent* package is missing.
    try:
        return importlib.util.find_spec("agent.adapters.emergence") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_HAS_NEURAL = _has_neural()


class TestContractInSync(unittest.TestCase):
    def test_action_vocab_matches_the_engine_enum(self):
        self.assertEqual(set(C.ACTION_VOCAB), {a.value for a in ActionType})
        self.assertEqual(len(C.ACTION_VOCAB), len(ActionType), "no duplicates")

    def test_param_spec_covers_every_action_and_nothing_else(self):
        self.assertEqual(set(C.PARAM_SPEC), {a.value for a in ActionType},
                         "PARAM_SPEC must list exactly the engine's actions")

    def test_self_view_keys_match_a_real_snapshot(self):
        snap = Agent(id="t", name="T", profession="farmer", persona="guardian",
                     x=0, y=0).snapshot()
        self.assertEqual(C.SELF_VIEW_KEYS, frozenset(snap),
                         "self_view schema drifted — bump CONTRACT_VERSION")

    def test_observation_fields_match_the_dataclass(self):
        self.assertEqual(C.OBSERVATION_FIELDS,
                         frozenset(Observation.__dataclass_fields__),
                         "Observation schema drifted — bump CONTRACT_VERSION")

    def test_validator_agrees_with_the_vocab(self):
        self.assertTrue(C.is_valid_action_value("move"))
        self.assertFalse(C.is_valid_action_value("teleport"))

    def test_idle_is_the_canonical_noop_and_first_in_vocab(self):
        self.assertEqual(C.IDLE_ACTION, "idle")
        self.assertEqual(C.ACTION_VOCAB[0], C.IDLE_ACTION,
                         "the clamp fallback ACTION_VOCAB[0] must be idle")

    def test_every_vocab_action_is_dispatchable_without_raising(self):
        # The engine must have a handler for every verb a policy can emit, and an
        # action with empty params must degrade gracefully (never raise). This is
        # the safety net the idle/out-of-vocab clamp and any partial spec rely on.
        sim = make_simulation("guardian", n_agents=4,
                              config=SimulationConfig(seed=1), economy=True)
        agent = sim.agents[0]
        for verb in C.ACTION_VOCAB:
            try:
                sim._apply(agent, Action(ActionType(verb), {}))
            except Exception as exc:                       # pragma: no cover
                self.fail(f"action {verb!r} raised on empty params: {exc!r}")

    def test_target_key_maps_reference_only_real_actions(self):
        for verb in {**C.COUNTERPARTY_KEY, **C.FACILITY_TARGET_KEY}:
            self.assertIn(verb, C.ACTION_VOCAB, f"{verb!r} is not an engine action")
        # strike is the one agent-OR-facility verb; arson is facility-only.
        self.assertIn("strike", C.COUNTERPARTY_KEY)
        self.assertIn("strike", C.FACILITY_TARGET_KEY)
        self.assertEqual(C.STRIKE_DEFAULT_TARGET, "agent")


@unittest.skipUnless(_HAS_NEURAL,
                     "the [neural] extra (agent.adapters.emergence) is not installed")
class TestRoundTripContract(unittest.TestCase):
    """Runs only where llm_model_agi is installed. The external adapter must
    satisfy every assertion here for the neural brain to actually drive agents."""

    def _obs_and_agent(self):
        sim = make_simulation("guardian", n_agents=4,
                              config=SimulationConfig(seed=1), economy=True)
        agent = sim.agents[0]
        return sim, agent, sim._observe(agent)

    def test_build_brain_returns_an_act_learn_object(self):
        from agent.adapters.emergence import build_brain
        dev = build_brain("guardian", None, None)
        self.assertTrue(hasattr(dev, "act") and hasattr(dev, "learn"))

    def test_act_then_map_yields_a_valid_applicable_action(self):
        from agent.adapters.emergence import build_brain, to_engine_action
        sim, agent, obs = self._obs_and_agent()
        dev = build_brain("guardian", None, None)
        spec = dev.act(obs)
        action = to_engine_action(spec, agent, obs)
        self.assertIsInstance(action, Action)
        self.assertIn(action.type.value, C.ACTION_VOCAB)
        sim._apply(agent, action)        # must not raise on a real engine state

    def test_learn_accepts_a_scalar_reward(self):
        from agent.adapters.emergence import build_brain
        _, _, obs = self._obs_and_agent()
        dev = build_brain("guardian", None, None)
        dev.learn(obs, 0.0)              # must not raise


if __name__ == "__main__":
    unittest.main()
