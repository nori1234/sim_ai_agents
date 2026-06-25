"""The three primal drives: appetite, sleep, and reproduction.

Japanese folk psychology speaks of the *san dai yokkyu* — the three great
desires of hunger (食欲), sleep (睡眠欲) and sex/reproduction (性欲). The base
simulation only modelled the first (via energy and food). This module adds the
other two.

The model is deliberately **instinctual, not rational**. Reproduction in
particular is not a cost-benefit decision; it is driven by an *urge* (libido)
that simply builds up over time and by the *pleasure* of discharging it — 本能
と気持ちよさ. Each of the three urges works the same way:

    urge rises every tick  →  the matching act discharges it  →  pleasure

An agent is, at bottom, an urge-relieving, pleasure-seeking creature. Rationality
only enters as a *floor*: a body too close to death simply cannot mate, however
strong the urge.

The whole layer is opt-in: with :data:`DrivesConfig.enabled` ``False`` (the
default) the urges never rise, so the carefully-tuned archetype outcomes are
unchanged. Turn it on to study richer, instinct-driven population dynamics.
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

    # -- reproduction (性欲) — driven by instinct + pleasure, not reason --
    reproduction: bool = False
    libido_per_tick: float = 3.0       # how fast the urge to mate builds
    libido_threshold: float = 45.0     # above this the urge starts to drive behaviour
    mate_libido_relief: float = 80.0   # how much mating discharges the urge
    maturity_age_days: int = 2         # agents younger than this cannot mate
    # Rational floor only — instinct does the driving, the body just has to cope:
    repro_energy_min: float = 35.0     # too weak to mate below this
    repro_hunger_max: float = 85.0     # only near-starvation blocks fertility
    repro_fatigue_max: float = 88.0    # only near-collapse blocks fertility
    repro_trust_min: float = 0.2       # a willing partner (familiarity/affection)
    repro_cooldown_days: int = 2       # brief refractory period
    repro_energy_cost: float = 18.0    # mating costs each parent some energy
    child_energy: float = 70.0         # a newborn's starting energy
    max_population: int = 30           # hard cap on living agents

    # -- aging / senescence (老化・寿命) — only bites while this layer advances
    # age_days, so the four-society baseline (drives off) is byte-identical.
    # Seeded adults start at age_days=99 (the "mature adult" sentinel), so the
    # elder phase sits just above that — founders have a believable remaining
    # span, not instant death; newborns (age 0) live a long young life first.
    senescence_age_days: int = 130     # past this the body begins to decline
    senescence_energy_penalty: float = 0.6  # extra energy lost per tick when old
    mortality_onset_days: int = 140    # natural death becomes possible past here
    mortality_hazard_per_day: float = 0.04  # daily death chance at onset, rising with age
    mortality_frailty_energy: float = 30.0  # below this energy, the hazard is worse
    mortality_frailty_mult: float = 1.5     # how much frailty raises the hazard

    # -- pleasure (気持ちよさ) — the reward for relieving any urge --------
    pleasure_per_eat: float = 1.0      # scaled by how hungry you were
    pleasure_per_sleep: float = 1.0    # scaled by how tired you were
    pleasure_per_mate: float = 6.0     # the big one — why the species bothers


# A reasonable "on" preset for demos.
DRIVES_ON = DrivesConfig(enabled=True, reproduction=True)
DRIVES_NO_REPRO = DrivesConfig(enabled=True, reproduction=False)


def is_fertile(agent, cfg: DrivesConfig, day: int) -> bool:
    """Whether ``agent``'s body is *capable* of reproducing right now.

    This is only the rational floor — viability — not the motivation. The drive
    to actually seek a mate comes from :func:`mating_urge`.
    """
    if not (cfg.enabled and cfg.reproduction):
        return False
    if not agent.alive or agent.age_days < cfg.maturity_age_days:
        return False
    if agent.energy < cfg.repro_energy_min:
        return False
    if agent.hunger > cfg.repro_hunger_max or agent.fatigue > cfg.repro_fatigue_max:
        return False
    if agent.last_reproduced_day is not None and \
            day - agent.last_reproduced_day < cfg.repro_cooldown_days:
        return False
    return True


def mating_urge(agent, cfg: DrivesConfig) -> float:
    """Instinctual pressure to seek a mate, as a 0..1 strength.

    Zero below the libido threshold, ramping to 1.0 as the urge becomes
    overwhelming. This — not a rational checklist — is what makes an agent court.
    """
    if not (cfg.enabled and cfg.reproduction):
        return 0.0
    if agent.libido <= cfg.libido_threshold:
        return 0.0
    span = max(1.0, 100.0 - cfg.libido_threshold)
    return min(1.0, (agent.libido - cfg.libido_threshold) / span)


# Backwards-compatible name: "can this agent reproduce" == is it fertile.
def can_reproduce(agent, cfg: DrivesConfig, day: int) -> bool:
    return is_fertile(agent, cfg, day)
