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
