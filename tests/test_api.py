"""The observatory service layer: EmergenceAPI + the HTTP route mapping.

Tested without sockets — the API is pure logic returning JSON-able dicts, and
the router is a plain function — so these are fast and deterministic.
"""

import unittest

from emergence.api import APIError, EmergenceAPI
from emergence.scenario import make_simulation
from emergence.server import _route
from emergence.simulation import SimulationConfig


class TestStepDayMatchesRun(unittest.TestCase):
    def test_stepping_day_by_day_equals_a_full_run(self):
        # The API advances worlds via step_day(); it must be byte-identical to
        # a full run() so the observatory stays as deterministic as the engine.
        cfg = SimulationConfig(seed=42)
        full = make_simulation("philosopher", config=cfg); full.run()
        stepped = make_simulation("philosopher", config=SimulationConfig(seed=42))
        guard = 0
        while stepped.step_day():
            guard += 1
            self.assertLess(guard, 1000)
        self.assertEqual(stepped.metrics.as_dict(), full.metrics.as_dict())


class TestWorldLifecycle(unittest.TestCase):
    def setUp(self):
        self.api = EmergenceAPI()

    def test_create_returns_id_and_state(self):
        st = self.api.create_world(persona="claude", seed=1, days=5)
        self.assertIn("world_id", st)
        self.assertEqual(st["day"], 1)
        self.assertEqual(st["population"], 10)
        self.assertEqual(st["config"], {"days": 5, "ticks": 8, "seed": 1})
        self.assertEqual(len(st["agents"]), 10)

    def test_bad_persona_rejected(self):
        with self.assertRaises(APIError) as cm:
            self.api.create_world(persona="nope")
        self.assertEqual(cm.exception.status, 400)

    def test_inputs_are_clamped(self):
        st = self.api.create_world(persona="guardian", days=9999, ticks=0)
        self.assertLessEqual(st["config"]["days"], 60)
        self.assertGreaterEqual(st["config"]["ticks"], 1)

    def test_step_advances_and_reports_new_events(self):
        st = self.api.create_world(persona="philosopher", seed=42, days=15)
        wid = st["world_id"]
        after = self.api.step(wid, days=3)
        self.assertEqual(after["day"], 3)
        self.assertIn("new_events", after)
        self.assertTrue(all(e["day"] <= 3 for e in after["new_events"]))

    def test_run_to_completion_sets_finished(self):
        st = self.api.create_world(persona="idealist", seed=42, days=15)
        wid = st["world_id"]
        out = self.api.step(wid, days=30)   # more than enough
        self.assertTrue(out["finished"])
        self.assertTrue(out["verdict"].startswith("COLLAPSE"))

    def test_delete_world(self):
        wid = self.api.create_world(persona="guardian")["world_id"]
        self.api.delete_world(wid)
        with self.assertRaises(APIError):
            self.api.world_state(wid)


class TestBrainSelector(unittest.TestCase):
    def setUp(self):
        self.api = EmergenceAPI()

    def test_default_brain_is_heuristic(self):
        st = self.api.create_world(persona="guardian", seed=1, days=3)
        self.assertEqual(st["brain"], "heuristic")

    def test_unknown_brain_rejected(self):
        with self.assertRaises(APIError):
            self.api.create_world(brain="quantum")

    def test_llm_brain_runs_via_injected_client(self):
        # A local/API world drives agents through an injected client (no
        # network here). The client always proposes a harmless action; the
        # world must run to completion and be labelled as an LLM brain.
        calls = []

        def fake_client(system, user):
            calls.append(1)
            return '{"action": "speak", "params": {"text": "hello"}, "rationale": "x"}'

        st = self.api.create_world(persona="guardian", seed=1, days=2,
                                   brain="local", llm_client=fake_client)
        self.assertTrue(st["brain"].startswith("llm:"))
        wid = st["world_id"]
        out = self.api.step(wid, days=2)
        self.assertTrue(out["finished"])
        self.assertTrue(calls, "the injected LLM client should have been called")

    def test_llm_brain_falls_back_when_client_errors(self):
        def broken(system, user):
            raise RuntimeError("model unreachable")

        st = self.api.create_world(persona="guardian", seed=1, days=2,
                                   brain="local", llm_client=broken)
        wid = st["world_id"]
        out = self.api.step(wid, days=2)   # survives on heuristic fallback
        self.assertTrue(out["finished"])


class TestStreaming(unittest.TestCase):
    def setUp(self):
        self.api = EmergenceAPI()

    def test_stream_days_yields_one_frame_per_day(self):
        st = self.api.create_world(persona="philosopher", seed=42, days=15)
        frames = list(self.api.stream_days(st["world_id"], days=4))
        self.assertEqual([f["day"] for f in frames], [1, 2, 3, 4])
        self.assertTrue(all("new_events" in f for f in frames))

    def test_stream_stops_when_world_finishes(self):
        st = self.api.create_world(persona="idealist", seed=42, days=15)
        frames = list(self.api.stream_days(st["world_id"], days=30))
        self.assertTrue(frames[-1]["finished"])
        self.assertLessEqual(len(frames), 15)

    def test_stream_matches_step(self):
        # Streaming day-by-day reaches the same state as stepping.
        a = self.api.create_world(persona="gemini", seed=42, days=6)
        list(self.api.stream_days(a["world_id"], days=6))
        b = EmergenceAPI().create_world(persona="gemini", seed=42, days=6)
        api_b = EmergenceAPI()
        wid = api_b.create_world(persona="gemini", seed=42, days=6)["world_id"]
        api_b.step(wid, days=6)
        self.assertEqual(self.api.world_state(a["world_id"])["metrics"],
                         api_b.world_state(wid)["metrics"])


class TestPossessView(unittest.TestCase):
    def setUp(self):
        self.api = EmergenceAPI()

    def test_agent_view_has_inner_life(self):
        st = self.api.create_world(persona="guardian", seed=42, days=4, rich=True)
        wid = st["world_id"]
        self.api.step(wid, days=4)
        aid = st["agents"][0]["id"]
        view = self.api.agent_view(wid, aid)
        self.assertIn("snapshot", view)
        self.assertIn("role", view)
        self.assertIn("relationships", view)
        self.assertIsInstance(view["memory"], list)

    def test_unknown_agent_404(self):
        wid = self.api.create_world(persona="guardian")["world_id"]
        with self.assertRaises(APIError) as cm:
            self.api.agent_view(wid, "ghost")
        self.assertEqual(cm.exception.status, 404)


class TestRouting(unittest.TestCase):
    def test_routes_map_to_api(self):
        # create -> state -> step -> events -> agent, end to end through _route.
        status, created = _route("POST", "/api/worlds", {},
                                  {"persona": "philosopher", "seed": 42, "days": 6})
        self.assertEqual(status, 201)
        wid = created["world_id"]
        status, _ = _route("GET", f"/api/worlds/{wid}", {}, {})
        self.assertEqual(status, 200)
        status, stepped = _route("POST", f"/api/worlds/{wid}/step",
                                 {"days": ["2"]}, {})
        self.assertEqual(stepped["day"], 2)
        status, evs = _route("GET", f"/api/worlds/{wid}/events", {"since": ["0"]}, {})
        self.assertIn("events", evs)
        aid = created["agents"][0]["id"]
        status, view = _route("GET", f"/api/worlds/{wid}/agents/{aid}", {}, {})
        self.assertIn("snapshot", view)

    def test_unknown_route_raises(self):
        with self.assertRaises(APIError) as cm:
            _route("GET", "/api/nope", {}, {})
        self.assertEqual(cm.exception.status, 404)


class TestHttpServer(unittest.TestCase):
    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from emergence.server import Handler
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()

    def _get(self, path):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()

    def test_root_serves_the_ui(self):
        status, ctype, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"Emergence", body)

    def test_health_is_json(self):
        status, ctype, body = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)


if __name__ == "__main__":
    unittest.main()
