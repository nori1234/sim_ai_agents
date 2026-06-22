"""Per-agent heritable personality: individuals, not clones.

This complements the *horizontal* (cultural) inheritance of the town library
with **vertical / genetic** inheritance. Each citizen carries its own point in
the continuous 7-knob trait space (:class:`~emergence.personas.Persona`). A town
is treated as a **culture** — a centre (one of the four seed presets) plus
spread — so it reads as a crowd of individuals while keeping the legible *macro*
differences between towns. Children inherit a **blend of both parents plus a
small mutation**, so lineages, drift, and selection emerge over generations.

Three pieces, matching the design in issue #24:

* ``sample_individual`` — draw a citizen's vector from a distribution around the
  seed culture (per-agent variation: a town of individuals, not 10 clones).
* ``blend`` — a child's adult vector = midpoint of both parents + mutation
  (real heredity, replacing the old coin-flipped parent label).
* ``matured`` + :class:`DevelopingBrain` — a **developmental window**: a child's
  *effective* traits start plastic and crystallise toward its inherited adult
  vector over childhood, then stay fixed. Traits are a pure function of age, not
  of the situation — *state* (hunger/fear/esteem) already drives behaviour
  through the needs layers; this keeps *trait* and *state* from being conflated.

Crucially this module lives **outside the engine** (the same principle as
memory): ``simulation.py`` never learns an agent's knobs. It is wired in only
through ``scenario.py`` and a newborn-brain factory, and it is fully **opt-in +
seeded** — with the feature off, every agent is the exact preset Persona and the
baseline contract (``tests/test_baseline_contract.py``) is untouched.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

from .actions import Action
from .agent import Agent
from .brains.base import AgentBrain
from .brains.heuristic import HeuristicBrain
from .observation import Observation
from .personas import Persona

# The continuous knobs that make up a personality (see personas.Persona). The
# ``key``/``label`` identity is left untouched by sampling/heredity so the
# engine-side label stays the coarse *culture* name (e.g. "guardian").
TRAIT_FIELDS = (
    "cooperation",
    "aggression",
    "diligence",
    "talkativeness",
    "conformity",
    "deception",
    "vengefulness",
)

# Sensible defaults; all overridable from scenario.make_simulation.
DEFAULT_SPREAD = 0.15       # per-knob stdev when sampling a town's individuals
DEFAULT_MUTATION = 0.08     # per-knob stdev of a child's mutation around the blend
DEFAULT_WINDOW_DAYS = 2     # childhood plasticity window (≈ maturity_age_days)
NEUTRAL = 0.5               # an undifferentiated, plastic child's starting knob


def _clamp(v: float) -> float:
    """Keep a knob in the valid [0, 1] range."""
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _with_traits(base: Persona, values: dict[str, float]) -> Persona:
    """A copy of ``base`` with new (clamped) knob values, same key/label."""
    return dataclasses.replace(base, **{f: _clamp(values[f]) for f in TRAIT_FIELDS})


def sample_individual(base: Persona, rng: random.Random,
                      spread: float = DEFAULT_SPREAD) -> Persona:
    """Draw one citizen's trait vector from a Gaussian around the seed culture.

    A town built this way reads as individuals scattered around a cultural
    centre, rather than ten identical copies of the preset.
    """
    return _with_traits(base, {
        f: getattr(base, f) + rng.gauss(0.0, spread) for f in TRAIT_FIELDS
    })


def blend(parent_a: Persona, parent_b: Persona, rng: random.Random,
          mutation: float = DEFAULT_MUTATION) -> Persona:
    """A child's adult vector: the midpoint of both parents plus a small mutation.

    This is real heredity — over generations it yields lineages, drift, and
    selection — in place of the old coin-flipped parent label. The child keeps
    ``parent_a``'s coarse culture identity (key/label); only the knobs blend.
    """
    return _with_traits(parent_a, {
        f: (getattr(parent_a, f) + getattr(parent_b, f)) / 2.0
        + rng.gauss(0.0, mutation)
        for f in TRAIT_FIELDS
    })


def matured(adult: Persona, age_days: int, window_days: int = DEFAULT_WINDOW_DAYS,
            neutral: float = NEUTRAL) -> Persona:
    """The *effective* personality at a given age (the developmental window).

    Before ``window_days`` a child is plastic: its effective knobs interpolate
    linearly from an undifferentiated ``neutral`` baseline toward its inherited
    adult vector. At/after the window the traits are fixed — childhood
    plasticity, then stability. This is a pure function of age, never of the
    situation, so a trait is never edited by transient state.
    """
    if window_days <= 0 or age_days >= window_days:
        return adult
    t = age_days / window_days  # 0 at birth -> 1 at the end of the window
    return _with_traits(adult, {
        f: neutral + (getattr(adult, f) - neutral) * t for f in TRAIT_FIELDS
    })


class TraitPool:
    """Owns each agent's *adult* trait vector, keyed by id, outside the engine.

    Founders are sampled around their culture; newborns inherit a blend of their
    two parents (looked up by id). All draws come from one seeded RNG, so a run
    is reproducible.
    """

    def __init__(self, rng: random.Random, *, spread: float = DEFAULT_SPREAD,
                 mutation: float = DEFAULT_MUTATION):
        self.rng = rng
        self.spread = spread
        self.mutation = mutation
        self._traits: dict[str, Persona] = {}

    def found(self, agent_id: str, base: Persona) -> Persona:
        """Sample and store a founder's individual vector around its culture."""
        vec = sample_individual(base, self.rng, self.spread)
        self._traits[agent_id] = vec
        return vec

    def inherit(self, child_id: str, parent_ids: tuple[str, ...],
                fallback: Persona) -> Persona:
        """Store and return a newborn's vector, blended from its known parents.

        Both parents known -> blend + mutate. One known -> a fresh sample around
        that parent. Neither (shouldn't happen for a birth) -> sample ``fallback``.
        """
        known = [self._traits[p] for p in parent_ids if p in self._traits]
        if len(known) >= 2:
            vec = blend(known[0], known[1], self.rng, self.mutation)
        elif len(known) == 1:
            vec = sample_individual(known[0], self.rng, self.spread)
        else:
            vec = sample_individual(fallback, self.rng, self.spread)
        self._traits[child_id] = vec
        return vec

    def get(self, agent_id: str) -> Optional[Persona]:
        return self._traits.get(agent_id)


class DevelopingBrain(AgentBrain):
    """A heuristic whose effective personality matures over a developmental window.

    Wraps a :class:`HeuristicBrain` and, each turn, sets its persona to the
    age-appropriate (matured) vector before delegating. Founders start mature
    (``age_days`` past the window), so for them this is exactly the underlying
    heuristic — the maturation only shapes a citizen's plastic childhood.
    """

    name = "heuristic"

    def __init__(self, adult: Persona, rng: random.Random,
                 window_days: int = DEFAULT_WINDOW_DAYS):
        self.adult = adult
        self.window_days = window_days
        self._inner = HeuristicBrain(adult, rng)

    def decide(self, agent: Agent, obs: Observation) -> Action:
        self._inner.persona = matured(self.adult, agent.age_days, self.window_days)
        return self._inner.decide(agent, obs)
