# The Observatory — product direction & API

The commercial direction for Emergence World: an **observatory** where you watch
an AI society emerge from above and can **possess any citizen** to live their
story from the inside. Watching (the god view) and immersion (a single life) in
one product — leaning on the engine's unique asset: *different AI personalities
produce visibly different societies, reproducibly.*

## Decisions so far

- **Concept:** observatory + possess (a fused overview/first-person experience),
  with the engine's metrics available underneath as the "why".
- **Immersion lead: the story, not the map.** An abstract dots-on-a-grid view
  conveys no society; the engine's real magic is the *emergent narrative*. So the
  experience leads with a curated **Chronicle** of the town and a **life story**
  for any possessed citizen (`emergence/chronicle.py`); the spatial map is a
  light supporting view. (A later layer can hand these grounded beats to an LLM
  for flowing prose.)
- **Distribution:** **local-first, hostable later** ("B → A"). Ship a
  local/desktop experience first (cheap, low-risk, the user brings their own LLM
  key or uses the free offline brain), architected so the *same* code becomes a
  hosted SaaS once validated. The dev-only "sell the API/platform" path (C) is
  out — it contradicts an experience-first product.
- **Why local-first:** the strategic risk is **LLM cost**. An entertainment sim
  that runs forever would wreck SaaS unit economics if we paid for inference, so
  the offline `HeuristicBrain` (free, deterministic) and bring-your-own-key are
  the default; hosted inference is an opt-in we add later behind quotas.
- **The brain, and what the product *is*.** The soul is **LLM-driven
  emergence** — that is what "AI societies that differ by model" requires, and
  what makes the story real. So the product is **LLM-forward**, with three brain
  modes selectable per world:
  - `heuristic` — free, instant, deterministic; the **test + demo/preview tier**
    (caricatured personas, *not* real AI — never sold as the AI).
  - `local` — a local LLM (Llama via Ollama); the **main** mode: private,
    ~free, real reasoning.
  - `api` — an ad-hoc hosted model (OpenAI-compatible or Anthropic).
  LLM brains fall back to the heuristic per-agent if the model is unreachable.
- **Reproducibility = engine seed + record/replay (not temperature=0).** The
  engine is already deterministic from its `seed`; the only non-determinism is
  the LLM call. We do **not** chase determinism with `temperature=0` — that
  isn't guaranteed (hardware/version drift) and it flattens the emergent
  richness that *is* the soul. Instead, **record every LLM run by default**
  (cache `prompt → response`); a recording replays bit-exactly, free, offline,
  hardware- and model-drift-independent — the gold standard for research/audit,
  and shippable alongside a paper. Temperature stays a plain knob (sensible
  default for richness; set 0 only to *study* modal behaviour), not a "mode".
  Sweep seeds for the *space* of societies; each seed stays reproducible.
  *(record/replay client: planned next.)*

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
| POST | `/api/worlds` | create a world (body: persona, seed, days, ticks, agents, rich, economy, environment, public_works, brain[`heuristic`\|`local`\|`api`], provider, model, base_url, temperature) |
| GET | `/api/worlds` | list worlds |
| GET | `/api/worlds/{id}` | full world state (agents, facilities, metrics, verdict) |
| DELETE | `/api/worlds/{id}` | delete a world |
| POST | `/api/worlds/{id}/step?days=N` | advance N days, returns state + `new_events` |
| GET | `/api/worlds/{id}/stream?days=N` | **SSE**: one `data:` frame per simulated day (live playback) |
| GET | `/api/worlds/{id}/events?since=N` | the story feed |
| GET | `/api/worlds/{id}/agents/{aid}` | a citizen's **possess** view (needs, memory, relationships) |
| GET | `/api/worlds/{id}/chronicle` | the curated, day-by-day **story** of the town |
| GET | `/api/worlds/{id}/agents/{aid}/story` | a citizen's **life story** (arc, ties, beliefs, fate) |
| GET | `/api/worlds/{id}/transcript` | export the recorded LLM transcript (feed back as `replay`) |

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

1. **API core** — create/inspect/step/possess + event feed. ✅
2. **Web UI** — a live town canvas (agents coloured by persona, click to
   possess), an event **story feed**, and a **possess panel** (needs bars,
   wealth, relationships, memories). Single self-contained HTML served at `/`;
   no build step, no dependencies; polls `step`. ✅
3. **Narrative engine** — a curated chronicle + per-citizen life stories
   (`emergence/chronicle.py`), the immersion lead. ✅
4. **Brain selector** — `heuristic` / `local` / `api` per world, LLM brains
   wired through `LLMBrain` with per-agent heuristic fallback. ✅
5. **Record/replay** — every LLM run records by default (`prompt → response`);
   `GET /worlds/{id}/transcript` exports it, and `create_world(replay=…)` re-runs
   it bit-exactly with no model call. The reproducibility backbone. ✅
5b. **LLM-narrated chronicle** — `GET /worlds/{id}/chronicle?narrate=1` turns the
   curated beats into flowing prose via the world's LLM, through the *same*
   recording client — so the narration is recorded and replays bit-exactly too
   (story × reproducibility). Heuristic worlds fall back to the curated text. ✅
6. **Story-led UI** — the web UI leads with the **Chronicle** and a possessed
   citizen's **life story** as the main reading panes; the town map is a
   supporting strip (context + click-to-possess) with a citizen roster. ✅

7. **Streaming** — `GET /worlds/{id}/stream` pushes one Server-Sent-Events
   frame per simulated day; the UI's Play uses `EventSource` for live playback
   (matters most for LLM worlds, where a day takes seconds). ✅
8. **Hosting (A)** — auth, quotas, multi-tenant, optional hosted inference;
   swap the stdlib transport for ASGI.
