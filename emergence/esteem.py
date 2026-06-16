"""Higher needs: esteem, honour, and power (承認欲求・名誉・権力).

Above the primal drives sit the social desires — the wish to be recognised,
admired and obeyed. People want to be *praised*, to be thought *impressive*
(すごいと思われたい), to hold honour and power. This module models that as one
more urge in the same shape as the primal drives:

    esteem urge rises every tick  →  recognition relieves it  →  pleasure

Recognition arrives two ways:

* **Praise** from another agent (褒められる) — a direct social stroke.
* **Achievement & office** — raising a monument, getting your own law passed,
  or being elected mayor. These also build lasting **reputation** (名誉), which
  fades slowly if you stop earning it, and which in turn draws more praise and
  helps win **power** (the mayoralty).

Opt-in via :data:`StatusConfig.enabled` (default ``False``), so runs without it
behave exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatusConfig:
    enabled: bool = False

    # -- the urge to be recognised (承認欲求) ----------------------------
    esteem_per_tick: float = 4.0       # how fast the need for recognition builds
    esteem_threshold: float = 40.0     # above this it starts driving behaviour

    # how much each form of recognition relieves the urge
    praise_relief: float = 28.0        # being praised by a peer
    achievement_relief: float = 22.0   # monument / your law passes
    mayor_relief: float = 55.0         # taking office

    # -- reputation / honour (名誉) -------------------------------------
    rep_per_praise: float = 2.0
    rep_per_monument: float = 9.0
    rep_per_law_passed: float = 3.0    # an proposal you authored is enacted
    rep_per_mayor: float = 12.0
    rep_per_collab: float = 1.0
    rep_decay_per_day: float = 1.5     # prestige fades without fresh deeds

    # -- pleasure (気持ちよさ) — recognition feels good ------------------
    pleasure_per_praise: float = 2.5   # 褒められて嬉しい
    pleasure_per_achievement: float = 2.0


def esteem_urge(agent, cfg: StatusConfig) -> float:
    """Instinctual pressure to seek recognition, as a 0..1 strength."""
    if not cfg.enabled:
        return 0.0
    if agent.esteem <= cfg.esteem_threshold:
        return 0.0
    span = max(1.0, 100.0 - cfg.esteem_threshold)
    return min(1.0, (agent.esteem - cfg.esteem_threshold) / span)
