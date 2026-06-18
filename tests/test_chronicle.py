"""The narrative layer: a curated town chronicle and per-citizen life stories."""

import unittest

from emergence.api import EmergenceAPI
from emergence.chronicle import chronicle, life_story, life_story_text
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def _run(persona="philosopher", seed=42, **kw):
    sim = make_simulation(persona, config=SimulationConfig(seed=seed, days=15), **kw)
    sim.run()
    return sim


class TestChronicle(unittest.TestCase):
    def test_chronicle_is_curated_and_ordered(self):
        sim = _run("philosopher")
        days = chronicle(sim)
        self.assertTrue(days)
        self.assertEqual([d["day"] for d in days], sorted(d["day"] for d in days))
        # Each listed day has at least one beat, and beats are strings.
        for d in days:
            self.assertTrue(d["beats"])
            self.assertTrue(all(isinstance(b, str) for b in d["beats"]))

    def test_proposals_are_summarised_not_listed(self):
        # The legislative spam should collapse into a single "passed N bills"
        # beat per day, not one line per proposal.
        sim = _run("philosopher")
        for d in chronicle(sim):
            council = [b for b in d["beats"] if "council passed" in b]
            self.assertLessEqual(len(council), 1)


class TestLifeStory(unittest.TestCase):
    def test_life_story_has_arc_and_fate(self):
        sim = _run("philosopher")
        # Pick whoever was busiest — they'll have a real arc.
        target = max(sim.agents, key=lambda a: a.crimes_committed + a.proposals_made)
        s = life_story(sim, target.id)
        self.assertEqual(s["name"], target.name)
        self.assertIn("fate", s)
        self.assertIsInstance(s["beats"], list)
        text = life_story_text(sim, target.id)
        self.assertIn(target.name, text)

    def test_unknown_agent_raises(self):
        sim = _run("guardian")
        with self.assertRaises(KeyError):
            life_story(sim, "ghost")


class TestNarrativeApi(unittest.TestCase):
    def test_chronicle_and_story_endpoints(self):
        api = EmergenceAPI()
        wid = api.create_world(persona="gemini", seed=42, days=15, rich=True)["world_id"]
        api.step(wid, days=15)
        chron = api.chronicle(wid)
        self.assertTrue(chron["finished"])
        self.assertTrue(chron["days"])
        self.assertIn("text", chron)
        # A possessed citizen has a life story.
        aid = api.world_state(wid)["agents"][0]["id"]
        story = api.agent_story(wid, aid)
        self.assertIn("beats", story)
        self.assertIn("fate", story)
        self.assertIn("text", story)


if __name__ == "__main__":
    unittest.main()
