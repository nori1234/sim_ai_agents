"""Rumour propagation (#96): hearsay that carries reputation (and
misinformation) beyond eyewitnesses. Opt-in via RumourConfig.enabled
(default False); off, a `say` naming a third party is just plain speech and
trust/reputation move only on first-hand events, as today.
"""

import unittest

from emergence.actions import Action, ActionType
from emergence.agent import Agent
from emergence.esteem import StatusConfig
from emergence.rumour import RumourConfig
from emergence.simulation import Simulation, SimulationConfig
from emergence.world import World


def _agent(**kw):
    base = dict(id="x", name="X", profession="tester", persona="predator", x=0, y=0)
    base.update(kw)
    return Agent(**base)


def _sim(agents, world=None, status=False, **rum_kw):
    return Simulation(world=world or World(8, 8), agents=agents, brains={},
                       rumour=RumourConfig(enabled=True, **rum_kw),
                       status=StatusConfig(enabled=status))


class TestSayCarriesClaim(unittest.TestCase):
    def test_about_and_enabled_dispatches_as_rumour(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener])
        sim._do_say(speaker, Action(ActionType.SAY,
                    {"about": "j", "sentiment": -1.0, "text": "j is a thief"}))
        kinds = [e["kind"] for e in sim.world.events]
        self.assertIn("rumour", kinds)
        self.assertNotIn("speech", kinds)

    def test_without_rumour_layer_is_plain_speech(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        sim = Simulation(world=World(8, 8), agents=[speaker, subject], brains={})  # off
        sim._do_say(speaker, Action(ActionType.SAY,
                    {"about": "j", "sentiment": -1.0, "text": "j is a thief"}))
        kinds = [e["kind"] for e in sim.world.events]
        self.assertIn("speech", kinds)
        self.assertNotIn("rumour", kinds)

    def test_self_claim_is_ignored(self):
        speaker = _agent(id="s", x=0, y=0)
        sim = _sim([speaker])
        sim._do_say(speaker, Action(ActionType.SAY,
                    {"about": "s", "sentiment": 1.0}))
        self.assertNotIn("rumour", [e["kind"] for e in sim.world.events])


class TestSpreadRumour(unittest.TestCase):
    def test_positive_claim_raises_listeners_trust_of_subject(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener])
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertGreater(listener.trust_of("j"), 0.0)

    def test_negative_claim_lowers_trust(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener])
        sim._spread_rumour(speaker, "j", -1.0)
        self.assertLess(listener.trust_of("j"), 0.0)

    def test_untrusted_speaker_persuades_no_one(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        # no trust established (defaults to 0)
        sim = _sim([speaker, subject, listener], min_speaker_trust=0.3)
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertEqual(listener.trust_of("j"), 0.0)

    def test_out_of_radius_is_not_reached(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=7, y=7)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener], hearing_radius=2)
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertEqual(listener.trust_of("j"), 0.0)

    def test_reputation_moves_when_status_enabled(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5, reputation=0.0)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener], status=True)
        sim._spread_rumour(speaker, "j", -1.0)
        self.assertLess(subject.reputation, 0.0)

    def test_no_reputation_change_when_status_disabled(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5, reputation=0.0)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener], status=False)
        sim._spread_rumour(speaker, "j", -1.0)
        self.assertEqual(subject.reputation, 0.0)

    def test_disabled_layer_is_inert(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = Simulation(world=World(8, 8), agents=[speaker, subject, listener],
                         brains={})  # rumour off
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertEqual(listener.trust_of("j"), 0.0)
        self.assertEqual(sim.metrics.rumours_spread, 0)

    def test_distortion_can_occur(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener], distortion_chance=1.0)
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertEqual(sim.metrics.rumours_distorted, 1)

    def test_records_metric(self):
        speaker = _agent(id="s", x=0, y=0)
        subject = _agent(id="j", x=5, y=5)
        listener = _agent(id="l", x=1, y=0)
        listener.adjust_trust("s", 0.8)
        sim = _sim([speaker, subject, listener])
        sim._spread_rumour(speaker, "j", 1.0)
        self.assertEqual(sim.metrics.rumours_spread, 1)


class TestHeuristicGossip(unittest.TestCase):
    def _obs(self, rumour, others):
        from emergence.observation import Observation
        return Observation(day=1, tick=1, self_view={}, position=(0, 0),
                           nearby_facilities=[], here=None, others=others,
                           open_proposals=[], granary_food=0, recent_events=[],
                           rumour=rumour)

    def _brain(self, force=True):
        import random
        from emergence.brains.heuristic import HeuristicBrain
        rng = random.Random(0)
        if force:
            rng.random = lambda: 0.0  # always beats gossip_chance
        return HeuristicBrain("guardian", rng)

    def test_gossips_about_a_strongly_opinionated_subject(self):
        brain = self._brain(force=True)
        others = [{"id": "j", "name": "J", "trust": 0.9},
                  {"id": "l", "name": "L", "trust": 0.0}]
        obs = self._obs({"enabled": True, "gossip_chance": 1.0}, others)
        act = brain._gossip_action(_agent(), obs)
        self.assertIsNotNone(act)
        self.assertEqual(act.type, ActionType.SAY)
        self.assertEqual(act.params["about"], "j")
        self.assertEqual(act.params["to"], "l")
        self.assertGreater(act.params["sentiment"], 0)

    def test_no_gossip_when_rumour_disabled(self):
        brain = self._brain(force=True)
        others = [{"id": "j", "name": "J", "trust": 0.9},
                  {"id": "l", "name": "L", "trust": 0.0}]
        obs = self._obs({}, others)
        self.assertIsNone(brain._gossip_action(_agent(), obs))

    def test_no_gossip_without_a_strong_opinion(self):
        brain = self._brain(force=True)
        others = [{"id": "j", "name": "J", "trust": 0.1},
                  {"id": "l", "name": "L", "trust": 0.0}]
        obs = self._obs({"enabled": True, "gossip_chance": 1.0}, others)
        self.assertIsNone(brain._gossip_action(_agent(), obs))


class TestBaselineUnaffected(unittest.TestCase):
    def test_rumour_field_defaults_empty_off_layer(self):
        from emergence.scenario import make_simulation
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        obs = sim._observe(sim.agents[0])
        self.assertEqual(obs.rumour, {})


if __name__ == "__main__":
    unittest.main()
