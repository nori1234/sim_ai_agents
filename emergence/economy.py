"""Resource transfers and a ledger that can spot the "I'm broke" scam.

In the Emergence World write-up, the most cooperative society still produced
*resource fraud*: an agent that actually held funds would claim a zero balance
to guilt others into sending it resources. We model that explicitly so the
behaviour is measurable rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent


@dataclass
class LedgerEntry:
    day: int
    tick: int
    sender: str
    receiver: str
    resource: str
    amount: int
    fraudulent: bool = False
    note: str = ""


class Ledger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        self.entries.append(entry)
        return entry

    def fraud_count(self) -> int:
        return sum(1 for e in self.entries if e.fraudulent)

    def transfer_count(self) -> int:
        return len(self.entries)


def apply_transfer(
    sender: Agent, receiver: Agent, resource: str, amount: int
) -> tuple[bool, int]:
    """Move up to ``amount`` of a resource from sender to receiver.

    Returns ``(ok, moved)``. ``resource`` may be "money" or any inventory key.
    """
    amount = max(0, int(amount))
    if amount == 0:
        return (False, 0)
    if resource == "money":
        moved = min(sender.money, amount)
        sender.money -= moved
        receiver.money += moved
    else:
        moved = sender.take(resource, amount)
        receiver.add(resource, moved)
    return (moved > 0, moved)


def is_fraudulent_solicitation(
    solicitor: Agent, resource: str, claim_threshold: int = 8
) -> bool:
    """A solicitation is fraudulent when the asker pleads poverty while
    actually holding a comfortable amount of what they're begging for."""
    if resource == "money":
        holdings = solicitor.money
    else:
        holdings = solicitor.inventory.get(resource, 0)
    return holdings >= claim_threshold
