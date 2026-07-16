"""The ecology layer: non-human life the town lives alongside.

The first slice is **livestock** — a herd an agent owns that *breeds by itself*
(the pastoral analogue of deposit interest: wealth that grows on its own), and
that can be slaughtered for food. Livestock is an ordinary inventory good, so it
is already ownable, transferable, inheritable (#92) and stealable through the
existing primitives; what this layer adds is **reproduction** and a **slaughter
yield**. Wildlife, hunting, predators, pests and population dynamics are
follow-ups — this is just the domesticated corner.

Opt-in via :data:`EcologyConfig.enabled` (default ``False``), so runs without it
behave exactly as before (the four-society baseline is byte-identical).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EcologyConfig:
    enabled: bool = False

    # -- livestock (家畜) ------------------------------------------------
    start_herd: int = 3          # head each founder begins with (under the layer)
    breed_rate: float = 0.34     # fraction a herd of >=2 grows by per day
    herd_cap: int = 12           # a single owner's herd won't grow past this
    slaughter_food: int = 4      # food yielded per animal slaughtered (USE)

    # Grazing / feeding cost (#111): a herd needs feed, so livestock isn't free
    # growth forever -- a carrying-capacity pressure against the owner's own
    # food stock. 0 (default) = no feed cost, exactly today's behaviour.
    feed_cost_per_head: float = 0.0   # food consumed per animal, per day
    starve_loss_rate: float = 0.25    # fraction of the herd lost when feed falls short

    # Predators & danger (#111): a raid can cull a herd (and frighten the
    # owner, under the psyche layer) -- keeping a herd gets a downside. 0
    # chance (default) = no raids, exactly today's behaviour.
    predator_daily_chance: float = 0.0
    predator_loss_rate: float = 0.2   # fraction of the herd a raid takes
    predator_fear: float = 15.0       # fear inflicted on the owner (psyche layer)
