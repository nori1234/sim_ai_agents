import unittest

from emergence.drives import DrivesConfig
from emergence.esteem import StatusConfig
from emergence.psyche import PsycheConfig
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig
from emergence.viz import render_html


class TestViz(unittest.TestCase):
    def _run(self, persona):
        sim = make_simulation(persona, config=SimulationConfig(seed=42))
        sim.run()
        return sim

    def _run_maslow(self, persona):
        sim = make_simulation(
            persona, config=SimulationConfig(seed=42),
            drives=DrivesConfig(enabled=True, reproduction=True),
            status=StatusConfig(enabled=True),
            psyche=PsycheConfig(enabled=True),
        )
        sim.run()
        return sim

    def test_html_is_well_formed(self):
        html = render_html(self._run("guardian"), "T")
        self.assertTrue(html.lstrip().startswith("<!doctype html>"))
        self.assertTrue(html.rstrip().endswith("</html>"))
        # SVG tags balanced (timeline + map + network).
        self.assertEqual(html.count("<svg"), html.count("</svg>"))
        self.assertEqual(html.count("<svg"), 3)

    def test_html_contains_sections(self):
        html = render_html(self._run("guardian"), "T")
        for needle in ("Daily timeline", "crime heatmap", "Trust network", "Citizens"):
            self.assertIn(needle, html)

    def test_heatmap_reflects_crime(self):
        peaceful = render_html(self._run("guardian"), "T")
        violent = render_html(self._run("predator"), "T")
        self.assertIn("no crime recorded", peaceful)
        self.assertIn("hottest cell", violent)

    def test_html_escapes_content(self):
        # Personas/names are escaped; ensure no raw angle-bracket injection paths.
        html = render_html(self._run("philosopher"), "<x>")
        self.assertIn("&lt;x&gt;", html)

    def test_basic_run_hides_advanced_panels(self):
        html = render_html(self._run("guardian"), "T")
        for absent in ("Needs pyramid", "reputation ranking", "Lineage"):
            self.assertNotIn(absent, html)

    def test_maslow_run_shows_all_panels(self):
        html = render_html(self._run_maslow("guardian"), "T")
        for needle in ("Needs pyramid", "Honour", "Lineage",
                       "Pleasure", "Works", "cumulative births"):
            self.assertIn(needle, html)

    def test_maslow_svg_balanced(self):
        html = render_html(self._run_maslow("guardian"), "T")
        self.assertEqual(html.count("<svg"), html.count("</svg>"))
        # timeline + pyramid + map + network + lineage = 5
        self.assertEqual(html.count("<svg"), 5)


if __name__ == "__main__":
    unittest.main()
