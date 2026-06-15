"""Economic *physics*, not economic *institutions*.

The aim (the user's): provide primitive, general laws — conserved goods that can
be owned, transferred, swapped by mutual agreement, and transformed by recipes —
and let institutions (prices, markets, money, firms) *emerge* on top, driven by
agents who already understand economics. Nothing here encodes "a market" or
"money": there is only the mechanics of exchange and production.

Primitives:
  * OFFER  — post "I give N of A, I want M of B" (A/B are any tradable good).
  * ACCEPT — a second agent agrees; the swap executes atomically (conserved).
  * CRAFT  — transform inputs into an output per a recipe (the physics of making).

From OFFER/ACCEPT over many agents, *prices* are just the ratios that trades
settle at; if everyone converges on pricing in one good, that good has become
*money* — emergently, not by fiat. (Money is one example, not an engine concept.)

Opt-in via ``Simulation.economy`` (default off); the baseline town is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

# Goods the primitives operate over. "money" is just one tradable quantity here,
# deliberately not privileged — agents decide whether to treat it as currency.
TRADABLE = ("food", "materials", "tools", "money")

# Production physics: output -> (inputs, facility-type value required or None).
# A value chain the economy can build on (raw materials -> tools).
RECIPES: dict[str, tuple[dict[str, int], str | None]] = {
    "tools": ({"materials": 2}, "workshop"),
}

OFFER_TTL_DAYS = 2          # an unaccepted offer expires after this long
MAX_OPEN_OFFERS = 60        # keep the order book bounded


@dataclass
class Offer:
    id: int
    maker: str            # agent id
    give_item: str
    give_qty: int
    want_item: str
    want_qty: int
    day: int

    def as_dict(self) -> dict:
        return {"id": self.id, "maker": self.maker,
                "give": f"{self.give_qty} {self.give_item}",
                "want": f"{self.want_qty} {self.want_item}"}


def holdings(agent, item: str) -> int:
    """How much of a tradable good an agent holds (money or an inventory item)."""
    if item == "money":
        return agent.money
    return agent.inventory.get(item, 0)
