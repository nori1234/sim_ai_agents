# The Observatory — the window onto the world (UI & API)

The **observatory** is how humans look at an Emergence World: watch an AI
society emerge from above, and **possess any citizen** to live their story from
the inside. Watching (the god view) and immersion (a single life) in one
interface. The project's primary direction is the research substrate
(grounding/agency — see `GROUNDING.md`); the observatory is its observation and
experience layer — how you *see* what the instruments measure — and doubles as
the inspection tool for research runs.

**What you are looking at.** A **single LLM**, individuated into distinct
citizens by **external scaffolding** — a per-agent character (its prompt/persona)
and an external, per-agent **memory** (its lived history) — let loose in the
engine's world (its physics, affordances and motives) so that a **society
emerges** from their interaction, reproducibly. The appeal is not "different
models"; it is *one* model becoming *many* individuals through what we bolt onto
it, and the society that grows out of them.

> **Note on "the four societies".** The four personas (Guardian/Philosopher/
> Idealist/Predator) are a **legacy of the original experiment**, which compared
> AI *models*. They survive only as (a) the **free/heuristic** caricatures and
> (b) a **reproducible reference benchmark** (`test_baseline_contract.py`). Real
> diversity in an LLM world should come from each agent's own character + memory,
> not from being one of four archetypes. *State today:* per-agent **memory** is
> already external and individual; per-agent **character** is still drawn from
> the four presets (`llm.py: _build_system_prompt`) — generating a distinct
> character per citizen is the natural next step for this layer.

## Design decisions (still in force)

- **Concept:** observatory + possess (a fused overview/first-person experience),
  with the engine's metrics available underneath as the "why".
- **Immersion lead: the story, not the map.** An abstract dots-on-a-grid view
  conveys no society; the engine's real magic is the *emergent narrative*. So the
  experience leads with a curated **Chronicle** of the town and a **life story**
  for any possessed citizen (`emergence/chronicle.py`); the spatial map is a
  light supporting view. (A later layer can hand these grounded beats to an LLM
  for flowing prose.)
- **Local-first, hostable later.** Everything runs on one machine with zero
  dependencies (the user brings their own LLM key or uses the free offline
  brain), architected so the *same* code can be hosted later if ever needed —
  hosting is a transport/ops change, not a rewrite. The driver is **LLM cost**:
  a sim that runs for weeks must not depend on paid inference, so the offline
  `HeuristicBrain` (free, deterministic) and bring-your-own-key are the default.
- **Three brain modes, selectable per world.** The soul is **LLM-driven
  emergence** — one model individuated by per-agent character + memory into many
  citizens, whose interaction grows a society:
  - `heuristic` — free, instant, deterministic; the demo/preview and floor/bench
    layer (caricatured personas, *not* real AI).
  - `local` — a local LLM (Llama via Ollama): private, ~free, real reasoning.
  - `api` — an ad-hoc hosted model (OpenAI-compatible or Anthropic).
  LLM brains fall back to the heuristic per-agent if the model is unreachable.
  (The developmental brain, `--neural`, is a CLI/engine feature; wiring it into
  the observatory's brain selector is future work.)
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
  (Shipped: every LLM run records by default and replays bit-exactly — roadmap
  step 5 below.)

## Architecture

```
  engine (emergence/*, stdlib-only, deterministic)
        ▲
  EmergenceAPI (emergence/api.py)   ← transport-agnostic; returns JSON-able dicts
        ▲
  HTTP adapter (emergence/server.py) ← stdlib http.server; thin, swappable for ASGI
        ▲
  Web UI (emergence/web/observatory.html) ← rich-2D town + chronicle + possess view
```

Run the observatory: `python -m emergence.server` → open `http://127.0.0.1:8800`.
The UI is a single self-contained HTML (no build, no dependencies) served at `/`.

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

To add if hosting ever happens: server-side LLM keys + cost quotas/rate limits, auth +
authz, per-tenant isolation, and runaway-sim limits.

## Roadmap

1. **API core** — create/inspect/step/possess + event feed. ✅
2. **Web UI** — a live town canvas (agents coloured by persona, click to
   possess), an event **story feed**, and a **possess panel** (needs bars,
   wealth, relationships, memories). Single self-contained HTML served at `/`;
   no build step, no dependencies. ✅
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
8. **Rich-2D town** — the town becomes the hero pane: "peg" citizens (round
   head + persona-coloured trapezoid body) walking on grounded facility
   landmarks, movement tweened between days, and crime/death events flashing at
   their spot. The land is **procedurally dressed** (all in-code, no image
   assets, so the file stays dependency-free): a seeded terrain of
   grass/forest/water/sand/rock + worn paths, a **seasonal palette**, a slow
   **day/night** sky, and **weather** particles (rain/snow) driven by the
   `--environment` layer (surfaced via the state's `environment` snapshot). Plain
   canvas 2D (no WebGL/deps) — enough for ≤40 figures. The chronicle and
   possessed life read on the right. ✅ (Toward a richer, "Steam-grade" look —
   next: better sprites, juice/particles, camera, audio; see #42.)
9. **Hosting (optional, future)** — auth, quotas, multi-tenant, optional hosted
   inference; swap the stdlib transport for ASGI. Not the current direction —
   the research substrate comes first.
