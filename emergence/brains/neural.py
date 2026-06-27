"""A brain that *develops* — it learns from experience inside the world.

Where :class:`~emergence.brains.llm.LLMBrain` is a frozen model that reads its
memory and acts (clever, but it never changes), this brain continually learns
from the consequences of its own actions and is raised by a *teacher* (an
existing ``LLMBrain``, or a human). The heavy machinery — a HierMamba encoder, a
policy/value head, a world-model for curiosity, fast test-time memory (Titans), a
replay buffer and the L0→L3 developmental stages — lives in the separate
``llm_model_agi`` package and is imported lazily.

This module is deliberately thin and defensive, mirroring ``LLMBrain``:

* **Opt-in, default off.** Nothing here runs unless an agent is explicitly given
  this brain (e.g. via ``--neural``), so the determinism baseline is untouched.
* **No new engine contract.** It implements only ``decide(agent, observation)``.
  Reward is *derived from the observation delta* (see ``_neural_reward``), not
  fed in by a changed engine loop.
* **Always falls back.** If torch / ``llm_model_agi`` are not installed, or a
  checkpoint is missing, or anything throws, it degrades silently to a
  :class:`~emergence.brains.heuristic.HeuristicBrain` — a run never crashes.

Per the engine contract the same brain instance is reused for an agent across all
ticks (``Simulation.brains`` is keyed by agent id), so learning state accumulates
over the agent's whole life — which is the entire point of a developmental brain.
"""

from __future__ import annotations

from typing import Optional

from ..actions import Action
from ..agent import Agent
from .base import AgentBrain
from .heuristic import HeuristicBrain


class NeuralDevelopmentalBrain(AgentBrain):
    """Continual-learning brain; defaults to learning, degrades to heuristic."""

    name = "neural"

    def __init__(self, persona, *, learn: bool = True,
                 teacher: Optional[AgentBrain] = None,
                 checkpoint: Optional[str] = None,
                 reward_weights: Optional[dict] = None):
        self._persona = persona
        self._fallback = HeuristicBrain(persona)
        self._teacher = teacher          # parent: an LLMBrain/Heuristic to imitate early
        self._learn = learn
        self._ckpt = checkpoint
        self._reward_weights = reward_weights
        self._dev = None                 # the DevelopmentalAgent; lazy-built
        self._broken = False             # latched once deps/build fail → straight to fallback
        self._prev_obs = None            # last observation, for the reward delta

    # -- lazy build: import torch / llm_model_agi only on first real use -----
    def _ensure(self) -> None:
        if self._dev is not None or self._broken:
            return
        # Imported here, never at module load, so the engine and the offline
        # baseline have zero dependency on torch or llm_model_agi.
        from agent.adapters.emergence import build_brain  # type: ignore
        self._dev = build_brain(self._persona, self._teacher, self._ckpt)

    # -- the only engine contract -------------------------------------------
    def decide(self, agent: Agent, observation) -> Action:
        if self._broken:
            return self._fallback.decide(agent, observation)
        try:
            self._ensure()
            # 1) Learn from what the *previous* action did to the world.
            if self._learn and self._prev_obs is not None:
                from ._neural_reward import survival_reward
                reward = survival_reward(self._prev_obs, observation,
                                         self._reward_weights)
                self._dev.learn(observation, reward)   # curiosity is added internally
            # 2) Choose an action (early stages imitate the teacher; later autonomous).
            spec = self._dev.act(observation)
            self._prev_obs = observation
            # 3) Map the policy's output spec onto a concrete engine Action.
            from agent.adapters.emergence import to_engine_action  # type: ignore
            return to_engine_action(spec, agent, observation)
        except Exception:
            # Any failure (missing deps, bad checkpoint, runtime error) → never
            # crash the run; latch so we don't keep retrying a hopeless import.
            self._broken = True
            self._dev = None
            return self._fallback.decide(agent, observation)
