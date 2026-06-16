import unittest

from emergence.actions import ActionType
from emergence.affordances import affordances_at, role_of
from emergence.brains.llm import LLMBrain
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.world import Facility, FacilityType


class TestAffordances(unittest.TestCase):
    def test_town_hall_affords_governance(self):
        f = Facility("Hall", FacilityType.TOWN_HALL, 0, 0)
        aff = affordances_at(f)
        self.assertIn(ActionType.PROPOSE.value, aff)
        self.assertIn(ActionType.VOTE.value, aff)

    def test_hospital_affords_rest(self):
        self.assertIn(ActionType.REST.value,
                      affordances_at(Facility("H", FacilityType.HOSPITAL, 0, 0)))

    def test_empty_tile_has_no_specific_affordances(self):
        self.assertEqual(affordances_at(None), [])

    def test_role_known_and_default(self):
        self.assertIn("hospital", role_of("doctor"))
        self.assertTrue(role_of("zzz-unknown"))  # falls back to a generic line


class TestObservationCarriesRoleAndAffordances(unittest.TestCase):
    def test_observation_has_role(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=1))
        obs = sim._observe(sim.agents[0])
        self.assertTrue(obs.role)
        self.assertIsInstance(obs.affordances, list)


class TestLLMSeesRoleAndAffordances(unittest.TestCase):
    def test_prompt_includes_role_and_affordances(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1, days=1))
        agent = sim.agents[0]
        obs = sim._observe(agent)
        captured = {}

        def client(system, user):
            captured["user"] = user
            return '{"action": "idle"}'

        LLMBrain(persona="guardian", client=client).decide(agent, obs)
        self.assertIn("your_role", captured["user"])
        self.assertIn("you_can_here", captured["user"])


if __name__ == "__main__":
    unittest.main()
