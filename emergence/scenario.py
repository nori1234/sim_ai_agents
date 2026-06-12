"""Helpers to assemble a population and wire up brains for a run."""

from __future__ import annotations

import random

from .agent import Agent
from .brains.base import AgentBrain
from .brains.heuristic import HeuristicBrain
from .personas import get_persona
from .simulation import Simulation, SimulationConfig
from .world import World, build_default_world

# A rotating set of jobs so agents read as a town, not a clone army.
PROFESSIONS = [
    "farmer", "builder", "teacher", "merchant", "doctor",
    "guard", "miner", "librarian", "smith", "council clerk",
]

FIRST_NAMES = [
    "Aria", "Bao", "Caro", "Dmitri", "Esme", "Faiz", "Gus", "Hana",
    "Ivo", "Juno", "Kit", "Lia", "Mira", "Nael", "Ono", "Priya",
]


def _name(i: int, rng: random.Random) -> str:
    base = FIRST_NAMES[i % len(FIRST_NAMES)]
    return f"{base}-{i + 1}"


def build_population(
    n: int,
    world: World,
    rng: random.Random,
) -> list[Agent]:
    """Create ``n`` agents scattered across the town."""
    agents: list[Agent] = []
    for i in range(n):
        x = rng.randint(0, world.width - 1)
        y = rng.randint(0, world.height - 1)
        agents.append(
            Agent(
                id=f"a{i + 1}",
                name=_name(i, rng),
                profession=PROFESSIONS[i % len(PROFESSIONS)],
                persona="guardian",  # overwritten by the scenario below
                x=x,
                y=y,
            )
        )
    return agents


def make_simulation(
    persona_mix: list[str] | str,
    *,
    n_agents: int = 10,
    config: SimulationConfig | None = None,
    brain_factory=None,
) -> Simulation:
    """Build a ready-to-run :class:`Simulation`.

    ``persona_mix`` is either a single persona key/alias applied to everyone
    (e.g. ``"claude"``) or a list assigned round-robin across the agents
    (e.g. ``["guardian", "predator"]``).

    ``brain_factory(agent, persona, rng) -> AgentBrain`` lets you swap in an
    :class:`LLMBrain`; by default every agent gets a persona-tuned
    :class:`HeuristicBrain`.
    """
    config = config or SimulationConfig()
    rng = random.Random(config.seed)
    world = build_default_world()
    agents = build_population(n_agents, world, rng)

    if isinstance(persona_mix, str):
        persona_keys = [persona_mix] * n_agents
    else:
        persona_keys = [persona_mix[i % len(persona_mix)] for i in range(n_agents)]

    brains: dict[str, AgentBrain] = {}
    for agent, key in zip(agents, persona_keys):
        persona = get_persona(key)
        agent.persona = persona.key
        if brain_factory is not None:
            brains[agent.id] = brain_factory(agent, persona, rng)
        else:
            # Each brain gets its own derived RNG for reproducible variety.
            brains[agent.id] = HeuristicBrain(
                persona, random.Random(rng.randint(0, 2**31))
            )

    return Simulation(world=world, agents=agents, brains=brains, config=config)
