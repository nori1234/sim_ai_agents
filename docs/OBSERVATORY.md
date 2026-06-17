# The Observatory — product direction & API

The commercial direction for Emergence World: an **observatory** where you watch
an AI society emerge from above and can **possess any citizen** to live their
story from the inside. Watching (the god view) and immersion (a single life) in
one product — leaning on the engine's unique asset: *different AI personalities
produce visibly different societies, reproducibly.*

## Decisions so far

- **Concept:** observatory + possess (a fused overview/first-person experience),
  with the engine's metrics available underneath as the "why".
- **Distribution:** **local-first, hostable later** ("B → A"). Ship a
  local/desktop experience first (cheap, low-risk, the user brings their own LLM
  key or uses the free offline brain), architected so the *same* code becomes a
  hosted SaaS once validated. The dev-only "sell the API/platform" path (C) is
  out — it contradicts an experience-first product.
- **Why local-first:** the strategic risk is **LLM cost**. An entertainment sim
  that runs forever would wreck SaaS unit economics if we paid for inference, so
  the offline `HeuristicBrain` (free, deterministic) and bring-your-own-key are
  the default; hosted inference is an opt-in we add later behind quotas.

## Architecture

```
  engine (emergence/*, stdlib-only, deterministic)
        ▲
  EmergenceAPI (emergence/api.py)   ← transport-agnostic; returns JSON-able dicts
        ▲
  HTTP adapter (emergence/server.py) ← stdlib http.server; thin, swappable for ASGI
        ▲
  Web UI (next)                      ← live town + story feed + possess view
```

The split is deliberate: `EmergenceAPI` is the product logic (unit-tested
without sockets); `server.py` is a thin transport. Hosting later (auth,
multi-tenant, billing, ASGI/FastAPI) is a transport/ops change, not a rewrite.

Determinism is preserved end to end: a world is reproducible from its `seed`, and
the API advances it via `Simulation.step_day()`, which is byte-identical to a
full `run()` (guarded by a test).

## API (PoC, all JSON)

Run locally: `python -m emergence.server` → `http://127.0.0.1:8800`.

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/health` | liveness |
| POST | `/api/worlds` | create a world (body: persona, seed, days, ticks, agents, rich, economy, environment, public_works) |
| GET | `/api/worlds` | list worlds |
| GET | `/api/worlds/{id}` | full world state (agents, facilities, metrics, verdict) |
| DELETE | `/api/worlds/{id}` | delete a world |
| POST | `/api/worlds/{id}/step?days=N` | advance N days, returns state + `new_events` |
| GET | `/api/worlds/{id}/events?since=N` | the story feed |
| GET | `/api/worlds/{id}/agents/{aid}` | a citizen's **possess** view (needs, memory, relationships) |

`rich=true` turns on the human-feel layers (drives, esteem, psyche, society) so a
possessed citizen has an inner life.

## Security posture (light now, scales with hosting)

Built in from the start, sized to a local single-user app:
- **Input validation + caps** — persona from a fixed set (no free text yet),
  numeric ranges clamped, world count and per-request work capped (a sim is
  CPU-bound).
- **No reflected user text** — agent names/personas are engine-generated, so the
  injection surface (prompt injection, XSS in a future UI) is currently nil. The
  day we accept free-text personas, they must be sanitised before reaching
  prompts or HTML.

To add when hosting (A): server-side LLM keys + cost quotas/rate limits, auth +
authz, per-tenant isolation, and runaway-sim limits.

## Roadmap

1. **API core** — create/inspect/step/possess + event feed. ✅ (this slice)
2. **Web UI** — a live town canvas, an event "story feed", and a possess panel
   (a citizen's needs, memory, relationships, life timeline).
3. **Streaming** — push ticks via SSE/WebSocket instead of polling `step`.
4. **Hosting (A)** — auth, quotas, multi-tenant, optional hosted inference;
   swap the stdlib transport for ASGI.
