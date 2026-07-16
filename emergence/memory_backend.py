"""Adapter that gives town agents real long-term memory via the `memory-agent`
library (https://github.com/nori1234/memory-agent).

The library is an optional, dependency-free add-on. emergence stays zero-dependency
by default: this module only imports it lazily, and the whole feature is opt-in
(``--memory`` / ``make_simulation(memory=True)``). If the library is not installed,
constructing :class:`TownMemory` raises a clear, actionable error; nothing else in
the simulation is affected.

What it buys us (the long-horizon problem from the Emergence World write-up):
  * each agent gets an **isolated, persistent** memory (SQLite, namespaced by id)
  * events are stored **salience-gated** (important facts kept, chatter dropped)
  * recall returns **a few** memories reranked by relevance x recency x importance
  * memories **decay and are evicted** on the in-game clock (real forgetting)
  * contradictory facts **supersede** old ones (kept as history)
  * memory **survives across runs** — a second run recalls the first run's events

So instead of feeding an agent's entire growing history into every decision
(the quadratic-cost / context-rot failure mode), the brain is handed only the
handful of memories relevant to the moment.
"""

from __future__ import annotations

from typing import Optional

try:  # the library is optional — keep emergence importable without it
    from memory_agent import GameWorld, Persona
    MEMORY_AGENT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when the lib is absent
    GameWorld = None  # type: ignore
    Persona = None  # type: ignore
    MEMORY_AGENT_AVAILABLE = False


_INSTALL_HINT = (
    "The 'memory' feature needs the optional `memory-agent` library.\n"
    "Install it (zero runtime deps):\n"
    "    git clone https://github.com/nori1234/memory-agent\n"
    "    pip install -e ./memory-agent\n"
)


class TownMemory:
    """Per-agent long-term memory for a town, backed by a single ``GameWorld``.

    One memory namespace per agent id; an in-game clock drives forgetting. Build
    it from the town's agents, then have the simulation feed it events
    (``perceive`` / ``broadcast``), query it when an agent decides (``recall``),
    and advance + decay it once per day (``tick``).
    """

    def __init__(self, agents, *, path: str = ":memory:", semantic: bool = True,
                 use_llm: bool = False, day_seconds: float = 86400.0,
                 half_life_event_days: float = 7.0,
                 half_life_fact_days: float = 30.0):
        if not MEMORY_AGENT_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)
        self.world = GameWorld(
            path, semantic=semantic, use_llm=use_llm, day_seconds=day_seconds,
            half_life_event_days=half_life_event_days,
            half_life_fact_days=half_life_fact_days,
        )
        self._known: set[str] = {a["agent_id"] for a in self.world.list_agents()}
        for agent in agents:
            self.register(agent)

    def register(self, agent) -> None:
        """Add an agent's memory namespace (e.g. a newborn mid-run)."""
        if agent.id in self._known:
            return  # already persisted from a previous run — keep its memories
        persona = Persona(
            name=agent.name,
            role=agent.profession,
            traits=[agent.persona],
            backstory=f"A {agent.persona} living in the town.",
        )
        self.world.create_agent(persona, agent_id=agent.id)
        self._known.add(agent.id)

    # -- writing --------------------------------------------------------
    def perceive(self, agent_id: str, event_text: str) -> None:
        """One agent witnesses/experiences an event (stored in game-time)."""
        if agent_id in self._known:
            self.world.perceive(agent_id, event_text)

    def broadcast(self, event_text: str, witnesses: Optional[list[str]] = None) -> None:
        """A town-wide event everyone (or the given witnesses) perceives."""
        ids = [w for w in witnesses if w in self._known] if witnesses is not None else None
        if ids is None or ids:
            self.world.broadcast(event_text, witnesses=ids)

    # -- reading --------------------------------------------------------
    def recall(self, agent_id: str, query: str, k: int = 5) -> list[str]:
        """The few memories most relevant to ``query``, as plain strings."""
        if agent_id not in self._known:
            return []
        agent = self.world.agent(agent_id)
        items = agent.recall(query, now=self.world._now())[:k]
        return [m.content for m in items]

    # -- time -----------------------------------------------------------
    def tick(self) -> None:
        """Advance the in-game clock one day and run the forgetting pass."""
        self.world.tick()

    def active_counts(self) -> dict[str, int]:
        return {a["agent_id"]: a["active_memories"] for a in self.world.list_agents()}

    def total_active(self) -> int:
        return sum(self.active_counts().values())

    def close(self) -> None:
        self.world.close()


class MemoryBackedLibrary:
    """The town library (see :mod:`emergence.library`), backed by memory-agent's
    ``GameWorld`` instead of the zero-dep in-process list.

    Implements the same port as :class:`library.TownLibrary`
    (``write``/``read``/``burn``/``__len__``), so it's a drop-in swap; nothing
    in :mod:`emergence.simulation` needs to know which one it's holding. It is
    its own ``GameWorld`` (own SQLite file, independent of :class:`TownMemory`'s
    per-agent one) with a single pseudo-agent namespace standing in for the
    shelf, so every book is a "memory" that town's librarian remembers.

    What this actually buys over the stdlib store (see issue #23's own
    assessment -- the semantic-recall gain is modest, a hashed bag-of-words
    embedder, not real embeddings):
      * **cross-run persistence** -- point two runs at the same ``path`` and
        the second recalls books the first one wrote (proven by
        :class:`TownMemory`'s own ``test_memory_survives_across_sessions``).
      * **supersession** -- memory-agent corrects a contradicted fact rather
        than piling up stale duplicates, kept as history rather than lost.

    There is no documented delete/purge call on ``GameWorld`` to reach for,
    so unlike :class:`library.TownLibrary`, ``burn()`` here cannot actually
    empty the persisted store -- it reports the count as if it had (the
    town-facing effect: an arsonist makes the town believe the record is
    gone), but the underlying SQLite file keeps the old entries as inert
    history rather than being wiped. Disclosed here rather than faked.
    """

    _AGENT_ID = "__town_library__"

    def __init__(self, path: str = ":memory:", *, use_llm: bool = False):
        if not MEMORY_AGENT_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)
        self.world = GameWorld(path, semantic=True, use_llm=use_llm)
        known = {a["agent_id"] for a in self.world.list_agents()}
        if self._AGENT_ID not in known:
            persona = Persona(
                name="The Town Library", role="archive",
                traits=["cumulative", "public"],
                backstory="The town's shared, persistent record.",
            )
            self.world.create_agent(persona, agent_id=self._AGENT_ID)
        # Exact-text dedup within this run only (like TownLibrary's dedup,
        # which guards the common case: an agent's _library_study writing
        # essentially the same firsthand lesson every visit). Not reloaded
        # from a prior run -- the persisted store itself is what carries
        # cross-run state, via GameWorld's own recall/active-memory count.
        self._written_this_run: set[str] = set()

    def write(self, day: int, author_id: str, author: str, text: str) -> dict | None:
        text = (text or "").strip()
        if not text or text in self._written_this_run:
            return None
        self._written_this_run.add(text)
        # Persisted for cross-run recall; memory-agent handles contradictory
        # facts (supersession) internally as new perceptions come in.
        self.world.perceive(self._AGENT_ID, f"{author} (day {day}): {text}")
        return {"day": day, "author_id": author_id, "author": author, "text": text}

    def read(self, query: str, k: int = 3) -> list[str]:
        agent = self.world.agent(self._AGENT_ID)
        items = agent.recall(query, now=self.world._now())[:k]
        return [m.content for m in items]

    def burn(self) -> int:
        return len(self)

    def __len__(self) -> int:
        for a in self.world.list_agents():
            if a["agent_id"] == self._AGENT_ID:
                return a["active_memories"]
        return 0

    def close(self) -> None:
        self.world.close()
