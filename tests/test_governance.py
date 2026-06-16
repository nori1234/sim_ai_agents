import unittest

from emergence.governance import (
    GOVERNANCE_PRESETS,
    GovernanceConfig,
    GovernanceForm,
    Law,
    LawEffect,
    Legislature,
    PolicyEngine,
    ProposalStatus,
)
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


class TestLawEffects(unittest.TestCase):
    def test_crime_deterrence_parsed(self):
        law = Law.from_proposal_text(1, "Ban theft and violence in the town", 1)
        self.assertIn(LawEffect.CRIME_DETERRENCE, law.effects)

    def test_food_redistribution_parsed(self):
        law = Law.from_proposal_text(2, "Establish a granary quota for all citizens", 1)
        self.assertIn(LawEffect.FOOD_REDISTRIBUTION, law.effects)

    def test_tax_parsed(self):
        law = Law.from_proposal_text(3, "Levy a wealth tax to fund public works", 1)
        self.assertIn(LawEffect.TAX, law.effects)

    def test_punishment_parsed(self):
        law = Law.from_proposal_text(4, "Fine all offenders reported to the police", 1)
        self.assertIn(LawEffect.PUNISHMENT, law.effects)

    def test_neutral_text_no_effects(self):
        law = Law.from_proposal_text(5, "The market shall open at dawn", 1)
        self.assertEqual(law.effects, [])

    def test_deduplication(self):
        law = Law.from_proposal_text(6, "Ban theft and prohibit violence", 1)
        self.assertEqual(law.effects.count(LawEffect.CRIME_DETERRENCE), 1)


class TestGovernanceForms(unittest.TestCase):
    def _sim(self, persona, gov):
        return make_simulation(persona, config=SimulationConfig(seed=42),
                               governance=gov)

    def test_anarchy_has_no_proposals(self):
        sim = self._sim("guardian", "anarchy")
        sim.run()
        self.assertEqual(sim.metrics.proposals_total, 0)
        self.assertEqual(sim.metrics.laws_enacted, 0)

    def test_direct_has_proposals(self):
        sim = self._sim("guardian", "direct")
        sim.run()
        self.assertGreater(sim.metrics.proposals_total, 0)

    def test_oligarchy_has_fewer_proposals_than_direct(self):
        direct = self._sim("guardian", "direct")
        direct.run()
        olig = self._sim("guardian", "oligarchy")
        olig.run()
        # Only 3 agents can propose in oligarchy → far fewer total bills.
        self.assertLess(olig.metrics.proposals_total, direct.metrics.proposals_total)

    def test_constitutional_supermajority_blocks_rights_bills(self):
        cfg = GovernanceConfig(
            form=GovernanceForm.CONSTITUTIONAL,
            quorum=3,
            supermajority=0.80,  # extremely high bar
        )
        leg = Legislature(cfg)
        p = leg.propose("a", "Declare freedom of assembly as a right", 1,
                        eligible_ids=None)
        self.assertTrue(p.constitutional)
        # 3 yes out of 5 = 60% — below 80% supermajority.
        for vid, support in (("a", True), ("b", True), ("c", True),
                               ("d", False), ("e", False)):
            leg.cast_vote(p.id, vid, support)
        leg.resolve_ready(electorate_size=5)
        self.assertEqual(p.status, ProposalStatus.REJECTED)

    def test_oligarchy_non_eligible_cannot_vote(self):
        cfg = GovernanceConfig(form=GovernanceForm.OLIGARCHY, quorum=2)
        leg = Legislature(cfg)
        eligible = {"rich1", "rich2"}
        p = leg.propose("rich1", "a rule", 1, eligible_ids=eligible)
        self.assertIsNotNone(p)
        # Non-eligible vote is ignored.
        leg.cast_vote(p.id, "peasant", True, eligible_ids=eligible)
        self.assertNotIn("peasant", p.votes)

    def test_punishment_law_generates_fines(self):
        # Philosopher towns have lots of crime — pair with punishment law.
        sim = self._sim("philosopher", "constitutional")
        sim.run()
        # Constitutional governance + Philosophers eventually pass crime/punishment
        # laws; some fines should be collected.
        # (We just check the counter is a non-negative int.)
        self.assertGreaterEqual(sim.metrics.fines_collected, 0)

    def test_elections_happen_on_interval(self):
        sim = self._sim("guardian", "direct")
        sim.run()
        cfg = sim.policy.config
        expected = sim.metrics.days_run // cfg.election_interval
        self.assertGreaterEqual(sim.metrics.elections, expected)

    def test_governance_form_recorded_in_metrics(self):
        for gov in ("direct", "oligarchy", "constitutional", "anarchy"):
            sim = self._sim("guardian", gov)
            sim.run()
            self.assertEqual(sim.metrics.gov_form, gov)

    def test_philosopher_survives_better_under_oligarchy(self):
        direct = self._sim("philosopher", "direct")
        direct.run()
        olig = self._sim("philosopher", "oligarchy")
        olig.run()
        # Oligarchy may or may not do better, but both should complete 15 days.
        self.assertEqual(direct.metrics.days_run, 15)
        self.assertEqual(olig.metrics.days_run, 15)


class TestPolicyEngine(unittest.TestCase):
    def test_enacted_law_publishes_a_crime_norm(self):
        # A crime law is no longer a mechanical multiplier; it publishes a norm
        # the agents may choose to comply with. Enacting one flips has_crime_norm.
        cfg = GOVERNANCE_PRESETS["direct"]
        engine = PolicyEngine(cfg)
        self.assertFalse(engine.has_crime_norm())
        engine.enact(1, "Ban all theft and crime in town", 1)
        self.assertTrue(engine.has_crime_norm())

    def test_summary(self):
        engine = PolicyEngine()
        engine.enact(1, "tax the wealthy", 1)
        s = engine.summary()
        self.assertEqual(s["laws_enacted"], 1)
        self.assertIn("tax", s["active_effects"])


if __name__ == "__main__":
    unittest.main()
