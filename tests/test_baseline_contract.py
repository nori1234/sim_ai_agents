"""The baseline 'contract': the four societies that must keep emerging.

This file is the safety net for the principled migration (see
docs/PRINCIPLED_MIGRATION.md). We are about to replace several hardcoded
*institutions* (money-as-a-privileged-field, the police-aura, law-keyword
magic) with *primitives* the same outcomes can EMERGE from. The whole point
of that work is that the qualitative ending of each archetype is preserved:

    guardian    -> ORDER       (Claude:  cooperative, conformist)
    philosopher -> CHAOS       (Gemini:  pervasive crime)
    idealist    -> COLLAPSE    (GPT:     idealism that fails to sustain)
    predator    -> FAILURE     (Grok:    predation that eats its own base)

The QUALITATIVE verdicts below are the durable contract. They must survive
every phase of the migration unchanged.

The NUMERIC snapshot is a softer signal: it lets us see *how far* a mechanic
change moved the world, so drift is visible in review rather than silent. It
is expected to shift as institutions become emergent; when it does, update
the snapshot in the same commit and justify the move in the message. The
qualitative verdicts are what may not regress.
"""

import unittest

from emergence.report import one_line_verdict
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

SEED = 42

# verdict prefix -> the archetype that must produce it.
CONTRACT = {
    "guardian": "ORDER",
    "philosopher": "CHAOS",
    "idealist": "COLLAPSE",
    "predator": "FAILURE",
}

# Snapshot last updated at Phase 4 (seed 42). Phases 1-3 dissolved money,
# the police aura, and law-magic into primitives; Phase 4 finishes the money
# story by letting theft/violence loot real coin (money is an inventory item
# like any other). No persona re-tuning was needed -- the arrest + norm
# primitives absorb the change, and the four endings still emerge and stay
# distinct. Tracks drift; update deliberately when a phase moves it.
SNAPSHOT = {
    "guardian": dict(survivors=10, population=10, crimes_total=0,
                     frauds=18, collaborations=3),
    "philosopher": dict(survivors=8, population=10, crimes_total=137,
                        frauds=0, collaborations=0),
    "idealist": dict(survivors=0, population=10, crimes_total=0,
                     frauds=2, collaborations=4),
    "predator": dict(survivors=1, population=10, crimes_total=74,
                     frauds=3, collaborations=0),
}


def _run(persona):
    sim = make_simulation(persona, config=SimulationConfig(seed=SEED))
    sim.run()
    return sim


class TestQualitativeContract(unittest.TestCase):
    """The four societies must keep emerging. This may not regress."""

    def test_each_archetype_produces_its_society(self):
        for persona, expected in CONTRACT.items():
            with self.subTest(persona=persona):
                verdict = one_line_verdict(_run(persona))
                self.assertTrue(
                    verdict.startswith(expected),
                    f"{persona}: expected a {expected} society, got {verdict!r}",
                )

    def test_the_four_endings_are_distinct(self):
        verdicts = {p: one_line_verdict(_run(p)).split(" ")[0] for p in CONTRACT}
        self.assertEqual(
            len(set(verdicts.values())), 4,
            f"the archetypes must diverge, not converge: {verdicts}",
        )


class TestNumericSnapshot(unittest.TestCase):
    """Drift tracker. Update deliberately, never silently."""

    def test_snapshot_matches(self):
        for persona, expected in SNAPSHOT.items():
            with self.subTest(persona=persona):
                m = _run(persona).metrics
                got = dict(survivors=m.survivors, population=m.population,
                           crimes_total=m.crimes_total, frauds=m.frauds,
                           collaborations=m.collaborations)
                self.assertEqual(
                    got, expected,
                    f"{persona} drifted from the Phase 0 snapshot; if a "
                    f"migration phase intended this, update SNAPSHOT here.",
                )


if __name__ == "__main__":
    unittest.main()
