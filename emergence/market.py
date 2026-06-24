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
    # When set, this is a SERVICE offer: the maker offers to *perform* a named
    # service (labour) rather than hand over a good. give_item/give_qty are
    # ignored; accepting pays want_qty (the maker's chosen price; 0 = charity)
    # and triggers the service's effect. Price is emergent, free-vs-paid is the
    # provider's choice.
    service: str | None = None
    # When True, this is a LOAN offer: the maker (lender) hands over give_qty of
    # give_item now, against a promise to be repaid want_qty later (want_qty >
    # give_qty = interest). Accepting opens a Loan; the borrower pays nothing now.
    loan: bool = False

    def as_dict(self) -> dict:
        if self.loan:
            return {"id": self.id, "maker": self.maker, "loan": True,
                    "item": self.give_item, "principal": self.give_qty, "repay": self.want_qty}
        if self.service is not None:
            return {"id": self.id, "maker": self.maker,
                    "service": self.service,
                    "want": f"{self.want_qty} {self.want_item}"}
        return {"id": self.id, "maker": self.maker,
                "give": f"{self.give_qty} {self.give_item}",
                "want": f"{self.want_qty} {self.want_item}"}


# Services are *labour offered for (optional) pay*. A provider CHOOSES to post a
# service offer at a price it picks (0 = charity); a taker accepts it (consent).
# The price is emergent (the accepted ratio), and free / fair / gouging is the
# provider's choice — like enforcement, not engine policy. Adding a service is a
# data entry here plus an effect handler in the simulation; banks (deposit /
# loan), inns (lodging) and the like slot in the same way.
#   provider: a required profession, or None for "anyone capable".
SERVICES: dict[str, dict] = {
    "healing": {"provider": "doctor",
                "desc": "restore a patient's energy (better at a hospital)"},
}


def can_provide(service: str, profession: str) -> bool:
    """Whether an agent of ``profession`` is able to offer ``service``."""
    spec = SERVICES.get(service)
    if spec is None:
        return False
    return spec["provider"] is None or spec["provider"] == profession



DEFAULT_LOAN_DUE_DAYS = 3


@dataclass
class Deposit:
    """A deposit-receipt — a *claim*-item. The depositor handed `amount` of money
    to the bank (an agent) for safe-keeping; the bank now owes it back on demand.
    Custody is a trusted promise, not an engine-guaranteed vault — so if the
    banker spends the coin, a withdrawal can come up short (a run / embezzlement
    emerges). If receipts ever circulate in trade, the claim becomes money."""
    id: int
    bank: str        # agent id holding the funds
    holder: str      # the depositor; the claim is theirs
    amount: int

    def as_dict(self) -> dict:
        return {"id": self.id, "bank": self.bank, "amount": self.amount}


@dataclass
class Loan:
    """Credit as physics: a creditor hands over a principal now against a promise
    to return `repay` later. Trust is the only collateral — repaying builds it,
    defaulting destroys it. From this, lending, interest, and trust-based money
    can emerge; nothing here is a 'bank'."""
    id: int
    creditor: str
    debtor: str
    item: str
    principal: int
    repay: int
    due_day: int
    settled: bool = False
    defaulted: bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "creditor": self.creditor, "debtor": self.debtor,
                "owe": f"{self.repay} {self.item}", "due_day": self.due_day}


def holdings(agent, item: str) -> int:
    """How much of a tradable good an agent holds (money or an inventory item)."""
    if item == "money":
        return agent.money
    return agent.inventory.get(item, 0)
