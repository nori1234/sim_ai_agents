import json
import unittest

from emergence.actions import ActionType
from emergence.brains.llm import LLMBrain
from emergence.brains.heuristic import HeuristicBrain
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


class _Recorder:
    """A mock LLM client: records prompts, returns a scripted reply."""

    def __init__(self, reply):
        self.reply = reply
        self.system = None
        self.user = None
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.system, self.user, self.calls = system, user, self.calls + 1
        return self.reply(system, user) if callable(self.reply) else self.reply


def _brain(reply, persona="guardian"):
    return LLMBrain(persona=persona, client=_Recorder(reply))


class TestPromptGrounding(unittest.TestCase):
    def _obs_brain(self, **sim_kw):
        sim = make_simulation("gemini", config=SimulationConfig(seed=42, days=2),
                              **sim_kw)
        # run a couple of ticks so there are memories/events, then grab an obs
        sim.run()
        agent = sim.agents[0]
        return sim, agent, sim._observe(agent)

    def test_prompt_includes_memory_and_environment(self):
        sim, agent, obs = self._obs_brain(memory=True, environment=True)
        rec = _Recorder('{"action": "idle", "params": {}}')
        brain = LLMBrain(persona="gemini", client=rec)
        brain.decide(agent, obs)
        self.assertIn("your_memories", rec.user)
        self.assertIn("world", rec.user)            # environment snapshot
        # system prompt instructs the model to use memory + adapt
        self.assertIn("MEMORIES", rec.system)
        self.assertIn("ADAPT", rec.system)
        if sim.memory is not None:
            sim.memory.close()

    def test_system_prompt_carries_persona_and_menu(self):
        brain = _brain('{"action":"idle"}', persona="predator")
        self.assertIn("predator", brain.system_prompt)
        self.assertIn("move", brain.system_prompt)  # action menu present


class TestDecisionParsing(unittest.TestCase):
    def setUp(self):
        self.sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=1))
        self.agent = self.sim.agents[0]
        self.obs = self.sim._observe(self.agent)

    def test_valid_json_drives_the_action(self):
        brain = _brain('{"action": "eat", "params": {}, "rationale": "hungry"}')
        action = brain.decide(self.agent, self.obs)
        self.assertEqual(action.type, ActionType.EAT)
        self.assertEqual(action.rationale, "hungry")

    def test_action_with_params(self):
        brain = _brain('Sure! {"action": "move", "params": {"facility_type": "farm"}}')
        action = brain.decide(self.agent, self.obs)
        self.assertEqual(action.type, ActionType.MOVE)
        self.assertEqual(action.params["facility_type"], "farm")

    def test_garbage_falls_back_to_heuristic(self):
        brain = _brain("I refuse to answer in JSON.")
        action = brain.decide(self.agent, self.obs)
        # Fallback heuristic always returns *some* valid action.
        self.assertIsInstance(action.type, ActionType)

    def test_client_exception_falls_back(self):
        def boom(system, user):
            raise RuntimeError("endpoint down")
        brain = LLMBrain(persona="guardian", client=boom)
        action = brain.decide(self.agent, self.obs)
        self.assertIsInstance(action.type, ActionType)

    def test_unknown_action_name_falls_back(self):
        brain = _brain('{"action": "teleport", "params": {}}')
        action = brain.decide(self.agent, self.obs)
        self.assertIsInstance(action.type, ActionType)


class TestEndToEndWithMock(unittest.TestCase):
    def test_full_run_on_a_mock_llm(self):
        # Every agent eats when it can; a complete run should execute and stay
        # deterministic given a deterministic client.
        def factory(agent, persona, rng):
            return LLMBrain(persona=persona,
                            client=_Recorder('{"action": "gather", "params": {}}'))
        a = make_simulation("guardian", config=SimulationConfig(seed=3, days=2),
                            brain_factory=factory)
        a.run()
        b = make_simulation("guardian", config=SimulationConfig(seed=3, days=2),
                            brain_factory=factory)
        b.run()
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())
        self.assertEqual(a.metrics.days_run, 2)


if __name__ == "__main__":
    unittest.main()
