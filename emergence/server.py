"""A dependency-free HTTP adapter over :class:`EmergenceAPI`.

Uses only the standard library (``http.server``), so the observatory runs
locally with nothing to install. The transport is deliberately thin: every
route maps to one ``EmergenceAPI`` method, so swapping in FastAPI/ASGI later (to
host it) is a transport change, not a rewrite.

Run it:  ``python -m emergence.server``  (then GET http://127.0.0.1:8800/).

Routes (all JSON):
  GET    /api/health
  GET    /api/worlds                      list worlds
  POST   /api/worlds                      create a world (JSON body)
  GET    /api/worlds/{id}                 world state
  DELETE /api/worlds/{id}                 delete a world
  POST   /api/worlds/{id}/step?days=N     advance N days
  GET    /api/worlds/{id}/events?since=N  event feed
  GET    /api/worlds/{id}/agents/{aid}    a citizen's "possess" view
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .api import APIError, EmergenceAPI

_API = EmergenceAPI()
_UI_PATH = os.path.join(os.path.dirname(__file__), "web", "observatory.html")


def _load_ui() -> bytes:
    try:
        with open(_UI_PATH, "rb") as fh:
            return fh.read()
    except OSError:
        return b"<!doctype html><meta charset=utf-8><p>UI file missing.</p>"


def _route(method: str, path: str, query: dict, body: dict) -> tuple[int, dict]:
    """Map a request to an EmergenceAPI call. Returns (status, payload)."""
    parts = [p for p in path.strip("/").split("/") if p]
    # /api/health
    if parts == ["api", "health"]:
        return 200, {"ok": True}
    # /api/worlds ...
    if parts[:2] == ["api", "worlds"]:
        rest = parts[2:]
        if not rest:
            if method == "GET":
                return 200, _API.list_worlds()
            if method == "POST":
                return 201, _API.create_world(**(body or {}))
        elif len(rest) == 1:
            wid = rest[0]
            if method == "GET":
                return 200, _API.world_state(wid)
            if method == "DELETE":
                return 200, _API.delete_world(wid)
        elif len(rest) == 2 and rest[1] == "step" and method == "POST":
            days = (query.get("days", [None])[0]
                    or (body or {}).get("days", 1))
            return 200, _API.step(rest[0], days=days)
        elif len(rest) == 2 and rest[1] == "events" and method == "GET":
            return 200, _API.events(rest[0],
                                    since=query.get("since", [0])[0],
                                    limit=query.get("limit", [200])[0])
        elif len(rest) == 2 and rest[1] == "chronicle" and method == "GET":
            return 200, _API.chronicle(rest[0])
        elif len(rest) == 3 and rest[1] == "agents" and method == "GET":
            return 200, _API.agent_view(rest[0], rest[2])
        elif len(rest) == 4 and rest[1] == "agents" and rest[3] == "story" \
                and method == "GET":
            return 200, _API.agent_story(rest[0], rest[2])
    raise APIError(f"no route for {method} {path}", 404)


class Handler(BaseHTTPRequestHandler):
    server_version = "EmergenceObservatory/0.1"

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        # Local-first: allow a separate dev UI origin to talk to us.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.body_written = True
        self.wfile.write(data)

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        body = {}
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8")) or {}
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "invalid JSON body"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "body must be a JSON object"})
        try:
            status, payload = _route(method, parsed.path, query, body)
            self._send(status, payload)
        except APIError as e:
            self._send(e.status, {"error": e.message})
        except TypeError as e:  # bad/unknown params to a create call, etc.
            self._send(400, {"error": str(e)})

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/app"):
            return self._send_html(_load_ui())
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._send(204, {})

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def run_server(host: str = "127.0.0.1", port: int = 8800) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Emergence observatory API on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
