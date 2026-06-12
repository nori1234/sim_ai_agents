"""The three primal drives: appetite, sleep, and reproduction.

Japanese folk psychology speaks of the *san dai yokkyu* — the three great
desires of hunger (食欲), sleep (睡眠欲) and sex/reproduction (性欲). The base
simulation only modelled the first (via energy and food). This module adds the
other two as distinct, competing needs so an agent must balance eating,
sleeping, and — optionally — pairing off to raise children.

The whole layer is opt-in: with :data:`DrivesConfig.enabled` ``False`` (the
default) hunger and fatigue never rise, so the carefully-tuned archetype
outcomes are unchanged. Turn it on to study richer population dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrivesConfig:
    # Master switch for hunger + sleep. Reproduction additionally needs
    # ``reproduction=True`` below.
    enabled: bool = False

    # -- appetite (食欲) -------------------------------------------------
    hunger_per_tick: float = 4.0       # how fast hunger climbs
    hunger_threshold: float = 65.0     # above this, hunger starts hurting
    hunger_energy_penalty: float = 3.0 # extra energy lost per tick when starving
    eat_hunger_relief: float = 30.0    # hunger removed per unit of food eaten

    # -- sleep (睡眠欲) --------------------------------------------------
    fatigue_per_tick: float = 3.0      # baseline drowsiness per tick
    fatigue_action_extra: float = 2.0  # strenuous actions tire you more
    fatigue_threshold: float = 72.0    # above this, fatigue starts hurting
    fatigue_energy_penalty: float = 2.0
    sleep_relief: float = 45.0         # fatigue removed by one SLEEP action
    sleep_energy_gain: float = 10.0    # sleeping also restores a little energy

    # -- reproduction (性欲) --------------------------------------------
    reproduction: bool = False
    maturity_age_days: int = 2         # agents younger than this cannot mate
    repro_hunger_max: float = 55.0     # must be reasonably well-fed
    repro_fatigue_max: float = 62.0    # must be reasonably rested
    repro_energy_min: float = 40.0     # and have energy to spare
    repro_trust_min: float = 0.25      # partners must trust each other
    repro_cooldown_days: int = 3       # rest between children
    repro_energy_cost: float = 20.0    # mating costs each parent energy
    child_energy: float = 70.0         # a newborn's starting energy
    max_population: int = 30           # hard cap on living agents


# A reasonable "on" preset for demos.
DRIVES_ON = DrivesConfig(enabled=True, reproduction=True)
DRIVES_NO_REPRO = DrivesConfig(enabled=True, reproduction=False)


def can_reproduce(agent, cfg: DrivesConfig, day: int) -> bool:
    """Whether ``agent`` individually meets the bar to seek a mate today."""
    if not (cfg.enabled and cfg.reproduction):
        return False
    if not agent.alive or agent.age_days < cfg.maturity_age_days:
        return False
    if agent.hunger > cfg.repro_hunger_max or agent.fatigue > cfg.repro_fatigue_max:
        return False
    if agent.energy < cfg.repro_energy_min:
        return False
    if agent.last_reproduced_day is not None and \
            day - agent.last_reproduced_day < cfg.repro_cooldown_days:
        return False
    return True
