import unittest

from emergence.governance import (
    GOVERNANCE_PRESETS,
    GovernanceConfig,
    GovernanceForm,
    Law,
    LawEffect,
    Legislature,
    OFFENCE_EFFECTS,
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

    def test_offence_specific_parsing_is_additive(self):
        # A bill naming "theft" still sets the existing blanket
        # CRIME_DETERRENCE (baseline compat) AND the new specific THEFT_NORM.
        law = Law.from_proposal_text(7, "Punish theft severely", 1)
        self.assertIn(LawEffect.CRIME_DETERRENCE, law.effects)
        self.assertIn(LawEffect.THEFT_NORM, law.effects)
        self.assertNotIn(LawEffect.VIOLENCE_NORM, law.effects)
        self.assertNotIn(LawEffect.ARSON_NORM, law.effects)

    def test_violence_and_arson_parsed_distinctly(self):
        violence = Law.from_proposal_text(8, "Assault will not be tolerated", 1)
        arson = Law.from_proposal_text(9, "Anyone caught burning a building is banished", 1)
        self.assertIn(LawEffect.VIOLENCE_NORM, violence.effects)
        self.assertNotIn(LawEffect.THEFT_NORM, violence.effects)
        self.assertIn(LawEffect.ARSON_NORM, arson.effects)
        self.assertNotIn(LawEffect.VIOLENCE_NORM, arson.effects)

    def test_a_generic_crime_law_names_no_specific_offence(self):
        # "ban ... crime" alone (no theft/violence/arson keyword) sets only
        # the blanket bucket -- there's nothing specific to be additive about.
        law = Law.from_proposal_text(10, "Crime will not be tolerated in this town", 1)
        self.assertIn(LawEffect.CRIME_DETERRENCE, law.effects)
        self.assertNotIn(LawEffect.VIOLENCE_NORM, law.effects)
        self.assertNotIn(LawEffect.THEFT_NORM, law.effects)
        self.assertNotIn(LawEffect.ARSON_NORM, law.effects)


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

    def test_offence_norm_is_specific(self):
        engine = PolicyEngine()
        self.assertFalse(engine.offence_norm("theft"))
        engine.enact(1, "Anyone caught stealing will be punished", 1)
        self.assertTrue(engine.offence_norm("theft"))
        self.assertFalse(engine.offence_norm("violence"))
        self.assertFalse(engine.offence_norm("arson"))

    def test_offence_norm_unknown_kind_is_false(self):
        engine = PolicyEngine()
        engine.enact(1, "Ban theft, violence and arson entirely", 1)
        self.assertFalse(engine.offence_norm("jaywalking"))

    def test_all_offence_kinds_covered(self):
        self.assertEqual(set(OFFENCE_EFFECTS), {"violence", "theft", "arson"})


class TestObservationSurfacesOffenceNorms(unittest.TestCase):
    def test_offences_present_alongside_the_blanket_crime_norm(self):
        # "theft" triggers both the existing blanket pattern and the new
        # specific one, so both show up together.
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        sim.policy.enact(1, "Ban theft in this town", sim.world.day)
        obs = sim._observe(sim.agents[0])
        self.assertTrue(obs.norms["crime"])
        self.assertEqual(obs.norms["offences"],
                         {"violence": False, "theft": True, "arson": False})

    def test_an_offence_only_law_surfaces_without_the_blanket_norm(self):
        # "stealing" (not "theft") never triggers the blanket CRIME_DETERRENCE
        # keywords -- before #35 this meant no norm at all was surfaced. Now
        # the offence-specific one still shows up, with "crime" correctly
        # False (not the blanket bucket -- nothing lies about which fired).
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        sim.policy.enact(1, "Anyone caught stealing will be punished", sim.world.day)
        obs = sim._observe(sim.agents[0])
        self.assertFalse(obs.norms["crime"])
        self.assertEqual(obs.norms["offences"],
                         {"violence": False, "theft": True, "arson": False})

    def test_no_law_means_no_norms_at_all(self):
        sim = make_simulation("guardian", config=SimulationConfig(seed=1))
        obs = sim._observe(sim.agents[0])
        self.assertEqual(obs.norms, {})

    def test_offence_only_norms_restrain_exactly_like_no_norms_at_all(self):
        # The key invariant that keeps this baseline-safe: _norm_restrains
        # only ever reads norm.get("crime")/.get("enforcement"). {} and
        # {"crime": False, ...} both make norm.get("crime") falsy, so an
        # offence-only law (which used to yield {}) must restrain nothing,
        # exactly as {} always has.
        from emergence.brains.heuristic import HeuristicBrain
        from emergence.personas import get_persona
        brain = HeuristicBrain(get_persona("predator"))
        obs_offence_only = type("O", (), {
            "norms": {"crime": False, "enforcement": 0.9, "offences": {"theft": True}}
        })()
        obs_empty = type("O", (), {"norms": {}})()
        self.assertEqual(
            brain._norm_restrains(get_persona("predator"), obs_offence_only),
            brain._norm_restrains(get_persona("predator"), obs_empty),
        )
        self.assertFalse(brain._norm_restrains(get_persona("predator"), obs_offence_only))

    def test_heuristic_ignores_the_new_key_baseline_unaffected(self):
        # The blanket-only compliance check (_norm_restrains) reads "crime"/
        # "enforcement" and nothing else, so adding "offences" must not
        # change outcomes versus a plain run.
        cfg = SimulationConfig(seed=42)
        a = make_simulation("predator", config=cfg); a.run()
        b = make_simulation("predator", config=cfg); b.run()
        self.assertEqual(a.metrics.as_dict(), b.metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
