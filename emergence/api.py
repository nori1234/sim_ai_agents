"""A thin, transport-agnostic service over the simulation engine.

This is the core for the *observatory* product: create a world, watch it
unfold, and "possess" any citizen to see their life. It is pure Python and
returns only JSON-able dicts; the HTTP layer (``server.py``) is a thin adapter
over these methods, so the same logic backs a local app today and a hosted
service later.

Design notes:
* No new dependencies — stdlib only, in keeping with the engine.
* Determinism is preserved: a world is reproducible from its ``seed``, and the
  API steps it day-by-day via :meth:`Simulation.step_day`, which matches a full
  ``run()`` exactly.
* Light safety from the start: inputs are validated and clamped, the number of
  worlds and the work per request are capped. (Free-text persona/agent input —
  which would need sanitising before reaching prompts or HTML — is not accepted
  yet; personas come from a fixed set.)
"""

from __future__ import annotations

import uuid

from .affordances import role_of
from .chronicle import (chronicle, chronicle_text, life_story, life_story_text,
                        narrate)
from .drives import DrivesConfig
from .esteem import StatusConfig
from .personas import ALIASES, PERSONAS
from .psyche import PsycheConfig
from .report import one_line_verdict
from .scenario import make_simulation
from .simulation import SimulationConfig
from .society import SocietyConfig

# Caps — cheap guards against runaway resource use (a sim is CPU-bound).
MAX_WORLDS = 64
MAX_DAYS = 60
MAX_TICKS = 24
MAX_STEP_DAYS = 30
MAX_AGENTS = 40


class APIError(Exception):
    """A client-facing error with an HTTP-ish status code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _clamp(value, lo, hi, default):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


class EmergenceAPI:
    """In-memory registry of running worlds plus the verbs to drive them."""

    def __init__(self) -> None:
        self._worlds: dict[str, object] = {}

    # -- helpers --------------------------------------------------------
    def _valid_persona(self, persona) -> str:
        key = str(persona or "").lower()
        key = ALIASES.get(key, key)
        if key not in PERSONAS:
            raise APIError(
                f"unknown persona {persona!r}; choose from "
                f"{sorted(PERSONAS)} or aliases {sorted(ALIASES)}"
            )
        return key

    def _get(self, world_id: str):
        sim = self._worlds.get(world_id)
        if sim is None:
            raise APIError(f"no such world {world_id!r}", 404)
        return sim

    # -- world lifecycle ------------------------------------------------
    def create_world(self, *, persona="guardian", seed=42, days=15, ticks=8,
                     agents=10, rich=False, economy=False, environment=False,
                     public_works=False, brain="heuristic", provider="openai",
                     model="llama3.1", base_url=None, api_key=None,
                     temperature=0.8, llm_client=None, replay=None) -> dict:
        """Create a world.

        ``rich`` turns on the human-feel layers (drives, esteem, psyche,
        society) so possessed citizens have inner lives.

        ``brain`` chooses who does the thinking:
          * ``"heuristic"`` (default) — free, instant, deterministic; the test
            and demo tier (caricatured personas, not real AI).
          * ``"local"`` — a local LLM (e.g. Llama via Ollama); the main mode:
            private, ~free, real reasoning.
          * ``"api"`` — an ad-hoc hosted model (OpenAI-compatible or Anthropic).
        LLM brains fall back to the heuristic per-agent if the model is
        unreachable, so a world always runs.
        """
        persona = self._valid_persona(persona)
        seed = _clamp(seed, 0, 2**31 - 1, 42)
        days = _clamp(days, 1, MAX_DAYS, 15)
        ticks = _clamp(ticks, 1, MAX_TICKS, 8)
        agents = _clamp(agents, 1, MAX_AGENTS, 10)
        if brain not in ("heuristic", "local", "api"):
            raise APIError(f"unknown brain {brain!r}; "
                           "choose heuristic, local, or api")
        if len(self._worlds) >= MAX_WORLDS:
            raise APIError("world limit reached; delete a world first", 429)

        brain_factory, brain_label, transcript, client = self._brain_factory(
            brain, provider, model, base_url, api_key, temperature,
            llm_client, replay)

        cfg = SimulationConfig(days=days, ticks_per_day=ticks, seed=seed)
        sim = make_simulation(
            persona, n_agents=agents, config=cfg,
            drives=DrivesConfig(enabled=True) if rich else None,
            status=StatusConfig(enabled=True) if rich else None,
            psyche=PsycheConfig(enabled=True) if rich else None,
            society=SocietyConfig(enabled=True) if rich else None,
            economy=bool(economy),
            environment=bool(environment),
            public_works=bool(public_works),
            brain_factory=brain_factory,
        )
        sim._brain_label = brain_label
        sim._transcript = transcript
        sim._llm_client = client  # reused for recorded, reproducible narration
        world_id = uuid.uuid4().hex[:12]
        self._worlds[world_id] = sim
        state = self._state(sim)
        state["world_id"] = world_id
        return state

    def _brain_factory(self, brain, provider, model, base_url, api_key,
                       temperature, llm_client, replay):
        """Build the per-agent brain factory, the brain label, and the run's
        transcript. Every LLM run records by default (so it is reproducible);
        a ``replay`` transcript serves recorded responses instead of the model.
        Returns ``(factory_or_None, label, transcript)``."""
        if brain == "heuristic":
            return None, "heuristic", {}, None
        from .brains.llm import LLMBrain, make_http_client
        from .replay import RecordingClient, ReplayClient
        prov = "anthropic" if (brain == "api" and provider == "anthropic") else "openai"
        if brain == "local":
            prov = "openai"
            base_url = base_url or "http://localhost:11434/v1"
        else:
            base_url = base_url or ("https://api.anthropic.com/v1"
                                    if prov == "anthropic"
                                    else "http://localhost:11434/v1")
        model = str(model)[:64]
        temperature = max(0.0, min(2.0, float(temperature)))

        # One completion path, shared by every agent, so the whole run records
        # into a single transcript. Replay serves from a recording (model-free).
        inner = llm_client or make_http_client(
            prov, model, base_url, api_key, temperature)
        if replay is not None:
            client = ReplayClient(dict(replay))
            transcript = client.transcript
        else:
            transcript = {}
            client = RecordingClient(inner, transcript)

        def factory(agent, persona, rng):
            return LLMBrain(provider=prov, model=model, base_url=base_url,
                            api_key=api_key, persona=persona,
                            temperature=temperature, client=client)

        mode = "replay" if replay is not None else "live"
        return factory, f"llm:{prov}:{model} ({mode})", transcript, client

    def list_worlds(self) -> dict:
        return {"worlds": [
            {"world_id": wid, "day": s.world.day,
             "finished": getattr(s, "_finished", False),
             "living": s._living(), "population": len(s.agents)}
            for wid, s in self._worlds.items()
        ]}

    def delete_world(self, world_id: str) -> dict:
        self._get(world_id)
        del self._worlds[world_id]
        return {"deleted": world_id}

    def transcript(self, world_id: str) -> dict:
        """Export the recorded LLM transcript — the reproducibility artifact.
        Feed it back as ``replay`` on a new world to re-run it bit-exactly."""
        sim = self._get(world_id)
        t = getattr(sim, "_transcript", {})
        return {"size": len(t), "transcript": t}

    # -- running --------------------------------------------------------
    def step(self, world_id: str, days=1) -> dict:
        sim = self._get(world_id)
        days = _clamp(days, 1, MAX_STEP_DAYS, 1)
        before = len(sim.world.events)
        for _ in range(days):
            if not sim.step_day():
                break
        state = self._state(sim)
        state["new_events"] = sim.world.events[before:]
        return state

    # -- reading --------------------------------------------------------
    def world_state(self, world_id: str) -> dict:
        return self._state(self._get(world_id))

    def events(self, world_id: str, since=0, limit=200) -> dict:
        sim = self._get(world_id)
        evs = sim.world.events
        since = _clamp(since, 0, len(evs), 0)
        limit = _clamp(limit, 1, 1000, 200)
        return {"since": since, "total": len(evs), "events": evs[since:since + limit]}

    def agent_view(self, world_id: str, agent_id: str) -> dict:
        """The "possess" view: one citizen's inner life — needs, memory, and
        who they trust or distrust."""
        sim = self._get(world_id)
        a = sim._by_id.get(agent_id)
        if a is None:
            raise APIError(f"no such citizen {agent_id!r}", 404)
        rel = [
            {"id": oid,
             "name": sim._by_id[oid].name if oid in sim._by_id else oid,
             "trust": round(t, 2)}
            for oid, t in sorted(a.trust.items(), key=lambda kv: -kv[1])
        ]
        return {
            "snapshot": a.snapshot(),
            "role": role_of(a.profession),
            "alive": a.alive,
            "cause_of_death": a.cause_of_death,
            "day_of_death": a.day_of_death,
            "memory": list(a.memory)[-20:],
            "relationships": rel,
        }

    # -- narrative (the story-led experience) ---------------------------
    def chronicle(self, world_id: str, narrate_prose=False) -> dict:
        """The curated, day-by-day story of the town. With ``narrate_prose`` and
        an LLM brain, also return flowing prose — recorded into the world's
        transcript, so the narration is as reproducible as the run."""
        sim = self._get(world_id)
        text = chronicle_text(sim)
        out = {
            "finished": getattr(sim, "_finished", False),
            "verdict": one_line_verdict(sim) if getattr(sim, "_finished", False) else None,
            "days": chronicle(sim),
            "text": text,
            "narrative": None,
        }
        if narrate_prose:
            out["narrative"] = narrate(text, getattr(sim, "_llm_client", None))
        return out

    def agent_story(self, world_id: str, agent_id: str) -> dict:
        """One citizen's life as a readable story."""
        sim = self._get(world_id)
        if sim._by_id.get(agent_id) is None:
            raise APIError(f"no such citizen {agent_id!r}", 404)
        story = life_story(sim, agent_id)
        story["text"] = life_story_text(sim, agent_id)
        return story

    # -- serialization --------------------------------------------------
    def _state(self, sim) -> dict:
        return {
            "day": sim.world.day,
            "tick": sim.world.tick,
            "width": sim.world.width,
            "height": sim.world.height,
            "finished": getattr(sim, "_finished", False),
            "config": {"days": sim.config.days,
                       "ticks": sim.config.ticks_per_day,
                       "seed": sim.config.seed},
            "brain": getattr(sim, "_brain_label", "heuristic"),
            "transcript_size": len(getattr(sim, "_transcript", {})),
            "population": len(sim.agents),
            "living": sim._living(),
            "agents": [self._agent_summary(a) for a in sim.agents],
            "facilities": [
                {"name": f.name, "type": f.ftype.value, "x": f.x, "y": f.y,
                 "roles": sorted(f.roles)}
                for f in sim.world.facilities
            ],
            "granary_food": sim.world.granary_food,
            "metrics": sim.metrics.as_dict(),
            "verdict": one_line_verdict(sim),
            "event_count": len(sim.world.events),
        }

    @staticmethod
    def _agent_summary(a) -> dict:
        return {
            "id": a.id, "name": a.name, "profession": a.profession,
            "persona": a.persona, "x": a.x, "y": a.y, "alive": a.alive,
            "energy": round(a.energy, 1), "money": a.money,
        }
