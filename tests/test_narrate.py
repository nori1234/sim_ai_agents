"""LLM narration of the chronicle — where 'story' meets 'reproducibility'.

The narration goes through the same recording client as the agents, so a
narrated chronicle is recorded and replays bit-exactly. Tested offline with an
injected mock client.
"""

import unittest

from emergence.api import EmergenceAPI
from emergence.chronicle import narrate


class TestNarrateFn(unittest.TestCase):
    def test_no_client_returns_none(self):
        self.assertIsNone(narrate("# Town Chronicle\n- something", None))

    def test_client_error_falls_back_to_none(self):
        def boom(system, user):
            raise RuntimeError("model down")
        self.assertIsNone(narrate("chronicle", boom))

    def test_client_prose_is_returned(self):
        out = narrate("chronicle md", lambda s, u: "  On the first day... \n")
        self.assertEqual(out, "On the first day...")


class TestNarrateApi(unittest.TestCase):
    def test_heuristic_world_has_no_narrative(self):
        api = EmergenceAPI()
        wid = api.create_world(persona="guardian", days=3)["world_id"]
        api.step(wid, days=3)
        ch = api.chronicle(wid, narrate_prose=True)
        self.assertIsNone(ch["narrative"])     # no LLM -> curated text only
        self.assertTrue(ch["text"])

    def test_llm_world_narrates_and_records(self):
        calls = []

        def mock(system, user):
            calls.append(system)
            # Agents get the action menu; the chronicler gets the narrate prompt.
            if "chronicler" in system:
                return "A turbulent few days unfolded."
            return '{"action": "speak", "params": {"text": "hi"}}'

        api = EmergenceAPI()
        wid = api.create_world(persona="philosopher", seed=42, days=3,
                               brain="local", llm_client=mock)["world_id"]
        api.step(wid, days=3)
        ch = api.chronicle(wid, narrate_prose=True)
        self.assertEqual(ch["narrative"], "A turbulent few days unfolded.")
        # The narration call was recorded into the transcript.
        self.assertGreater(api.transcript(wid)["size"], 0)

    def test_narration_replays_without_the_model(self):
        def mock(system, user):
            if "chronicler" in system:
                return "The town frayed but held."
            return '{"action": "speak", "params": {"text": "hi"}}'

        api = EmergenceAPI()
        wid = api.create_world(persona="philosopher", seed=42, days=3,
                               brain="local", llm_client=mock)["world_id"]
        api.step(wid, days=3)
        api.chronicle(wid, narrate_prose=True)          # record the narration
        transcript = api.transcript(wid)["transcript"]

        # Replay with a model that explodes if called — narration still resolves.
        def explode(system, user):
            raise AssertionError("model must not be called on replay")

        api2 = EmergenceAPI()
        wid2 = api2.create_world(persona="philosopher", seed=42, days=3,
                                 brain="local", llm_client=explode,
                                 replay=transcript)["world_id"]
        api2.step(wid2, days=3)
        ch = api2.chronicle(wid2, narrate_prose=True)
        self.assertEqual(ch["narrative"], "The town frayed but held.")


if __name__ == "__main__":
    unittest.main()
