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
