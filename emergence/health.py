"""The body: injury (怪我) — wounds that persist and need tending.

An attack subtracted energy alone before this layer — the harm was
instantaneous and indistinguishable from being tired, so a meal or a nap
erased it. This adds a lingering wound, in the same shape already
established by fear (psyche) and addiction (society): struck by an event,
decaying quietly over time (faster in shelter), and relieved fastest by a
doctor's care via the existing service substrate.

Opt-in via :data:`HealthConfig.enabled` (default ``False``); ``agent.injury``
sits at 0 and nothing here fires when off, so the four-society baseline is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = False

    injury_per_strike: float = 30.0          # violence wounds, not just tires
    injury_decay_per_tick: float = 1.0       # the body heals on its own, slowly
    injury_decay_shelter_bonus: float = 2.0  # resting at a house/hospital speeds it
    injury_severe_threshold: float = 50.0    # above this, the penalties bite
    injury_energy_penalty: float = 1.0       # a bad wound is a steady drain
    injury_gather_penalty: float = 0.4       # fraction of yield lost while badly hurt


def capability_factor(agent, cfg: HealthConfig) -> float:
    """1.0 unhurt, falling toward ``1 - injury_gather_penalty`` as injury
    climbs past the severe threshold. Scales gather/harvest yield."""
    if not cfg.enabled or agent.injury <= cfg.injury_severe_threshold:
        return 1.0
    span = max(1.0, 100.0 - cfg.injury_severe_threshold)
    strength = min(1.0, (agent.injury - cfg.injury_severe_threshold) / span)
    return 1.0 - cfg.injury_gather_penalty * strength
