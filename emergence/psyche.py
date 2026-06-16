"""The psyche layer: safety (fear) and self-actualization (恐怖・自己実現).

Two more storeys for the needs hierarchy, one below esteem and one above it:

**Fear (safety need)** is not an urge that builds with time — it is struck into
an agent by events. Suffering a crime, or merely witnessing one nearby, spikes
``fear``; it then decays with quiet time, faster in the shadow of a police
station or at home. While fear is high it overrides almost everything: the
agent drops its plans and runs for safety, and chronic terror (stress) erodes
its energy. A violent town does not just lose goods to crime — it loses the
whole productive day of everyone who saw it happen.

**Self-actualization** sits at the very top and works the other way around: it
is not a pressure but a *pull* that only appears when every lower need is
quiet — fed, rested, unafraid, recognised. In that state an agent may CREATE:
produce a work (a book at the library, a craft at the workshop, an artwork on
the plaza) that grants deep ``fulfillment``, intense pleasure, and — if the
esteem layer is active — lasting honour. Maslow's pyramid, playable.

Opt-in via :data:`PsycheConfig.enabled` (default ``False``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PsycheConfig:
    enabled: bool = False

    # -- fear / safety (恐怖・安全欲求) ----------------------------------
    fear_per_victimization: float = 45.0  # struck when you are the victim
    fear_per_witness: float = 18.0        # struck when crime happens nearby
    witness_radius: int = 6               # how far a crime radiates dread
    fear_decay_per_tick: float = 2.5      # quiet time heals
    fear_decay_safe_bonus: float = 4.0    # extra healing near police / home
    safe_radius: int = 4                  # what counts as "near" safety
    fear_threshold: float = 45.0          # above this, flight takes over
    fear_energy_penalty: float = 1.5      # chronic terror (stress) drains energy

    # -- self-actualization (自己実現) -----------------------------------
    # The pull only appears when every lower need is quiet:
    actualization_hunger_max: float = 40.0
    actualization_fatigue_max: float = 55.0
    actualization_fear_max: float = 20.0
    actualization_esteem_max: float = 55.0   # not aching for recognition
    actualization_energy_min: float = 50.0
    fulfillment_per_work: float = 10.0
    pleasure_per_work: float = 4.0        # creation is the deepest joy
    rep_per_work: float = 5.0             # a masterpiece is admired (needs --status)


def fear_level(agent, cfg: PsycheConfig) -> float:
    """How gripped by fear the agent is, as a 0..1 strength.

    Zero below the threshold; ramping to 1.0 as terror becomes absolute.
    """
    if not cfg.enabled or agent.fear <= cfg.fear_threshold:
        return 0.0
    span = max(1.0, 100.0 - cfg.fear_threshold)
    return min(1.0, (agent.fear - cfg.fear_threshold) / span)


def actualization_pull(agent, cfg: PsycheConfig) -> float:
    """The pull toward creation, 0..1 — present only when all lower needs rest.

    Checks the lower storeys directly off the agent's state; layers that are
    disabled leave their values at zero, which trivially satisfies the bar.
    """
    if not cfg.enabled or not agent.alive:
        return 0.0
    if agent.hunger > cfg.actualization_hunger_max:
        return 0.0
    if agent.fatigue > cfg.actualization_fatigue_max:
        return 0.0
    if agent.fear > cfg.actualization_fear_max:
        return 0.0
    if agent.esteem > cfg.actualization_esteem_max:
        return 0.0
    if agent.energy < cfg.actualization_energy_min:
        return 0.0
    # The freer the mind, the stronger the pull. Scale by spare energy.
    headroom = (agent.energy - cfg.actualization_energy_min) / max(
        1.0, 100.0 - cfg.actualization_energy_min
    )
    return min(1.0, 0.4 + 0.6 * headroom)
