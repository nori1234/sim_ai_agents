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
                 reward_weights: Optional[dict] = None,
                 hparams: Optional[dict] = None):
        # The two factory paths pass persona differently — a Persona object for
        # initial agents, a plain key string for newborns — so normalise to a key
        # string and hand build_brain a single, stable type.
        self._persona = getattr(persona, "key", persona)
        self._fallback = HeuristicBrain(persona)
        self._teacher = teacher          # parent: an LLMBrain/Heuristic to imitate early
        self._learn = learn
        self._ckpt = checkpoint
        self._reward_weights = reward_weights
        # Forwarded to build_brain's optional AgentConfig overrides (e.g.
        # batch_every, lr, lr_min, lr_decay_steps, entropy_weight,
        # self_attempt_base, bc_weight) — the brain side's late-training-
        # oscillation damper. None/{} means "use their defaults".
        self._hparams = hparams
        self._dev = None                 # the DevelopmentalAgent; lazy-built
        self._broken = False             # latched once deps/build fail → straight to fallback
        self._prev_obs = None            # last observation, for the reward delta
        # The brain side's learn() may optionally return a diagnostics dict (e.g.
        # {"grad_steps": int, "lr": float}) for hparam tuning (the lr-decay
        # schedule needs the actual per-agent step count to calibrate
        # lr_decay_steps). Purely informational — never required, never used by
        # engine logic; a trainer script may read it off the brain instance.
        self.last_learn_info: Optional[dict] = None

    # -- lazy build: import torch / llm_model_agi only on first real use -----
    def _ensure(self) -> None:
        if self._dev is not None or self._broken:
            return
        # Imported here, never at module load, so the engine and the offline
        # baseline have zero dependency on torch or llm_model_agi.
        from agent.adapters.emergence import build_brain  # type: ignore
        if self._hparams:
            self._dev = build_brain(self._persona, self._teacher, self._ckpt,
                                    hparams=self._hparams)
        else:
            self._dev = build_brain(self._persona, self._teacher, self._ckpt)

    # -- the only engine contract -------------------------------------------
    def decide(self, agent: Agent, observation) -> Action:
        if self._broken:
            return self._fallback.decide(agent, observation)
        try:
            self._ensure()
            # 1) Learn from what the *previous* action did to the world. When
            #    frozen (learn=False), still ACCUMULATE episodic memory via
            #    observe() — remembering is perception, not learning — so a
            #    memory-consuming policy can recall this world's own outcomes
            #    within its life at eval (v1). No-op when the brain has no memory.
            if self._prev_obs is not None:
                from ._neural_reward import survival_reward
                reward = survival_reward(self._prev_obs, observation,
                                         self._reward_weights)
                if self._learn:
                    info = self._dev.learn(observation, reward)  # curiosity added internally
                    if isinstance(info, dict):
                        self.last_learn_info = info
                elif hasattr(self._dev, "observe"):
                    self._dev.observe(reward)   # eval: memory only, no RL update
            # 2) Choose an action (early stages imitate the teacher; later autonomous).
            #    `agent` is passed so the brain's EngineTeacher can call
            #    teacher.decide(agent, obs) for imitation (contract decision (a)).
            spec = self._dev.act(observation, agent)
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
