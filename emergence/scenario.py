"""Helpers to assemble a population and wire up brains for a run."""

from __future__ import annotations

import random

from .agent import Agent
from .brains.base import AgentBrain
from .brains.heuristic import HeuristicBrain
from .drives import DrivesConfig
from .ecology import EcologyConfig
from .esteem import StatusConfig
from .grounding import CounterfactualConfig
from .health import HealthConfig
from .illness import IllnessConfig
from .innovation import InnovationConfig
from .psyche import PsycheConfig
from .rumour import RumourConfig
from .society import SocietyConfig
from .governance import GOVERNANCE_PRESETS, GovernanceConfig, Legislature, PolicyEngine
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
    governance: str | GovernanceConfig = "direct",
    drives: DrivesConfig | None = None,
    status: StatusConfig | None = None,
    psyche: PsycheConfig | None = None,
    society: SocietyConfig | None = None,
    illness: IllnessConfig | None = None,
    rumour: RumourConfig | None = None,
    innovation: InnovationConfig | None = None,
    health: "HealthConfig | None" = None,
    ecology: "EcologyConfig | None" = None,
    counterfactual: "CounterfactualConfig | None" = None,
    environment: "EnvironmentConfig | bool | None" = None,
    public_works: bool = False,
    founding: bool = False,
    economy: bool = False,
    memory: bool = False,
    memory_path: str = ":memory:",
    library: bool = False,
    library_path: str = ":memory:",
    individuals: bool = False,
    world: "World | None" = None,
    brain_factory=None,
) -> Simulation:
    """Build a ready-to-run :class:`Simulation`.

    ``persona_mix`` is either a single persona key/alias applied to everyone
    (e.g. ``"claude"``) or a list assigned round-robin across the agents
    (e.g. ``["guardian", "predator"]``).

    ``brain_factory(agent, persona, rng) -> AgentBrain`` lets you swap in an
    :class:`LLMBrain`; by default every agent gets a persona-tuned
    :class:`HeuristicBrain`.

    ``individuals`` (opt-in) turns each preset *culture* into a town of distinct
    citizens: every agent's trait vector is sampled around the culture's centre,
    and newborns inherit a blend of both parents + a small mutation (vertical /
    genetic inheritance — see :mod:`emergence.personality` and issue #24). It
    only applies to the default heuristic path (no ``brain_factory``); off, every
    agent is the exact preset Persona and the baseline contract is unchanged.
    Also assigns heritable *physical* traits (sex, body build, gait — #76),
    inert engine-side, surfaced on ``Agent`` (``.sex``/``.build``/``.gait``)
    for the observatory to render distinct silhouettes and walking tempo.

    ``library`` turns on the town's shared knowledge store. With ``memory``
    also on, the shelf is backed by memory-agent's ``GameWorld`` instead of
    the zero-dep default (:class:`memory_backend.MemoryBackedLibrary`,
    issue #23) — cross-run persistence and supersession at ``library_path``,
    same optional dependency as ``memory``/``memory_path`` already need.
    """
    config = config or SimulationConfig()
    if isinstance(governance, str):
        gov_cfg = GOVERNANCE_PRESETS.get(governance, GOVERNANCE_PRESETS["direct"])
    else:
        gov_cfg = governance
    rng = random.Random(config.seed)
    # A caller-supplied world (e.g. the grounding sandbox) wins over the presets.
    if world is not None:
        pass
    # A founding town starts sparse and develops itself through public works.
    elif founding:
        from .development import founding_world
        world = founding_world()
        public_works = True
    else:
        world = build_default_world()
    agents = build_population(n_agents, world, rng)

    if isinstance(persona_mix, str):
        persona_keys = [persona_mix] * n_agents
    else:
        persona_keys = [persona_mix[i % len(persona_mix)] for i in range(n_agents)]

    # Vertical/genetic inheritance (opt-in): a town of individuals around each
    # culture, with newborns blending both parents. Lives outside the engine; we
    # hand the engine only a newborn-brain factory that closes over the pool.
    pool = None
    physical_pool = None
    newborn_factory = None
    if individuals and brain_factory is None:
        from .personality import (TraitPool, PhysicalTraitPool, DevelopingBrain,
                                  DEFAULT_WINDOW_DAYS)
        # Dedicated, seed-derived RNGs so trait draws are reproducible and don't
        # disturb the brain-RNG stream used by the default path. Physical traits
        # get their own stream (distinct salt) so they vary independently of
        # the behavioural ones, not in lockstep.
        trait_rng = random.Random((config.seed or 0) ^ 0x70017)
        physical_rng = random.Random((config.seed or 0) ^ 0x7A1CE)
        pool = TraitPool(trait_rng)
        physical_pool = PhysicalTraitPool(physical_rng)
        window = DEFAULT_WINDOW_DAYS

        def newborn_factory(child, persona_key, sim_rng):
            vec = pool.inherit(child.id, child.parent_ids, get_persona(persona_key))
            phys = physical_pool.inherit(child.id, child.parent_ids)
            child.sex, child.build, child.gait = phys.sex, phys.build, phys.gait
            return DevelopingBrain(
                vec, random.Random(sim_rng.randint(0, 2**31)), window)

    brains: dict[str, AgentBrain] = {}
    for agent, key in zip(agents, persona_keys):
        persona = get_persona(key)
        agent.persona = persona.key
        if brain_factory is not None:
            brains[agent.id] = brain_factory(agent, persona, rng)
        elif pool is not None:
            # An individual sampled around this agent's culture; matures over a
            # developmental window (founders start mature, so this is a no-op
            # for them).
            vec = pool.found(agent.id, persona)
            phys = physical_pool.found(agent.id)
            agent.sex, agent.build, agent.gait = phys.sex, phys.build, phys.gait
            brains[agent.id] = DevelopingBrain(
                vec, random.Random(rng.randint(0, 2**31)), DEFAULT_WINDOW_DAYS)
        else:
            # Each brain gets its own derived RNG for reproducible variety.
            brains[agent.id] = HeuristicBrain(
                persona, random.Random(rng.randint(0, 2**31))
            )

    env = None
    if environment:
        from .environment import Environment, EnvironmentConfig
        env_cfg = environment if isinstance(environment, EnvironmentConfig) \
            else EnvironmentConfig(enabled=True)
        if env_cfg.enabled:
            env = Environment(env_cfg, world, random.Random(config.seed ^ 0x5EED))

    eco_cfg = ecology or EcologyConfig()
    if eco_cfg.enabled and eco_cfg.start_herd:
        for agent in agents:                       # each founder starts with a herd
            agent.add("livestock", eco_cfg.start_herd)

    town_memory = None
    if memory:
        from .memory_backend import TownMemory
        town_memory = TownMemory(agents, path=memory_path)

    town_library = None
    if library:
        if memory:
            # A memory-agent-backed shelf: cross-run persistence + contradictory-
            # fact supersession (#23), swapped in for the zero-dep default only
            # when --memory is also on (same optional dependency either way).
            from .memory_backend import MemoryBackedLibrary
            town_library = MemoryBackedLibrary(path=library_path)
        else:
            from .library import TownLibrary
            town_library = TownLibrary()

    legislature = Legislature(gov_cfg)
    policy = PolicyEngine(gov_cfg)
    return Simulation(
        world=world, agents=agents, brains=brains, config=config,
        legislature=legislature, policy=policy,
        drives=drives or DrivesConfig(),
        status=status or StatusConfig(),
        psyche=psyche or PsycheConfig(),
        society=society or SocietyConfig(),
        illness=illness or IllnessConfig(),
        rumour=rumour or RumourConfig(),
        innovation=innovation or InnovationConfig(),
        health=health or HealthConfig(),
        ecology=eco_cfg,
        counterfactual=counterfactual or CounterfactualConfig(),
        environment=env,
        public_works=bool(public_works),
        development=bool(founding),
        economy=bool(economy),
        memory=town_memory,
        library=town_library,
        newborn_brain_factory=newborn_factory,
    )
