"""Record & replay: an LLM run is recorded by default and replays bit-exactly
without the model — the reproducibility backbone for research."""

import unittest

from emergence.api import EmergenceAPI
from emergence.replay import RecordingClient, ReplayClient, key


class TestReplayClients(unittest.TestCase):
    def test_recording_then_replay_serves_without_inner(self):
        t = {}
        rec = RecordingClient(lambda s, u: f"resp:{u}", t)
        out = rec("sys", "hello")
        self.assertEqual(out, "resp:hello")
        self.assertIn(key("sys", "hello"), t)
        # Replay returns the recorded answer and never touches a model.
        rep = ReplayClient(t)
        self.assertEqual(rep("sys", "hello"), "resp:hello")
        self.assertEqual(rep.hits, 1)

    def test_replay_miss_raises_without_inner(self):
        rep = ReplayClient({})
        with self.assertRaises(KeyError):
            rep("sys", "unseen")


class TestRecordReplayRun(unittest.TestCase):
    def test_recorded_run_replays_bit_exactly(self):
        # 1) Record a local-brain run through a deterministic mock client.
        def mock(system, user):
            # A trivial-but-deterministic policy so the world actually moves.
            return '{"action": "speak", "params": {"text": "hi"}, "rationale": "x"}'

        api = EmergenceAPI()
        a = api.create_world(persona="philosopher", seed=42, days=4,
                             brain="local", llm_client=mock)
        wid = a["world_id"]
        api.step(wid, days=4)
        recorded = api.transcript(wid)
        self.assertGreater(recorded["size"], 0)
        metrics_a = api.world_state(wid)["metrics"]

        # 2) Replay it with NO model at all (a client that would explode if
        #    called). Same seed/config -> same prompts -> every key hits.
        def explode(system, user):
            raise AssertionError("the model must not be called during replay")

        api2 = EmergenceAPI()
        b = api2.create_world(persona="philosopher", seed=42, days=4,
                              brain="local", llm_client=explode,
                              replay=recorded["transcript"])
        wid2 = b["world_id"]
        api2.step(wid2, days=4)
        metrics_b = api2.world_state(wid2)["metrics"]

        # Bit-exact: the replayed run reproduces the recorded one.
        self.assertEqual(metrics_b, metrics_a)

    def test_brain_label_marks_replay(self):
        api = EmergenceAPI()
        st = api.create_world(brain="local", llm_client=lambda s, u: "{}",
                              replay={})
        self.assertIn("replay", st["brain"])


if __name__ == "__main__":
    unittest.main()
