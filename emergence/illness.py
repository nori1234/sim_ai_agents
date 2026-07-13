"""Illness & contagion (病気) — epidemics that create demand for medicine.

The only "sickness" in the engine before this was drug withdrawal
(``addiction``, society.py). There was no transmissible *disease* — no
epidemics, no contagion, no emergent public-health demand for doctors,
hospitals, or distancing. ``publicworks.py`` already has a dangling ``sick``
condition that proposes a hospital, with nothing to feed it.

This adds a contagious illness state, in the same shape already established
by fear (psyche) and injury: struck by chance, spreading by proximity
(denser clusters → faster spread → epidemics *emerge*, not scripted), harming
while it lasts, and recovering — faster with a doctor's care via the
existing service substrate.

Opt-in via :data:`IllnessConfig.enabled` (default ``False``); ``agent.illness``
sits at 0 and nothing here fires when off, so the four-society baseline is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IllnessConfig:
    enabled: bool = False

    daily_strike_chance: float = 0.02        # a healthy agent falls ill unprompted, per day
    contagion_radius: int = 2                # how close counts as "exposed"
    contagion_chance: float = 0.02           # per exposed neighbour, per TICK (denser
                                              # clusters / longer exposure -> faster spread)
    onset_severity: float = 25.0             # how sick a fresh infection starts at
    illness_decay_per_tick: float = 1.2      # the body fights it off, slowly
    illness_decay_shelter_bonus: float = 2.0  # resting at a house/hospital speeds it
    severe_threshold: float = 50.0           # above this, the penalties bite
    energy_penalty: float = 1.2              # a bad illness is a steady drain
    gather_penalty: float = 0.35             # fraction of yield lost while badly ill


def capability_factor(agent, cfg: IllnessConfig) -> float:
    """1.0 healthy, falling toward ``1 - gather_penalty`` as illness climbs
    past the severe threshold. Scales gather/harvest yield."""
    if not cfg.enabled or agent.illness <= cfg.severe_threshold:
        return 1.0
    span = max(1.0, 100.0 - cfg.severe_threshold)
    strength = min(1.0, (agent.illness - cfg.severe_threshold) / span)
    return 1.0 - cfg.gather_penalty * strength
