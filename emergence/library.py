"""The town library: knowledge that outlives the people who wrote it.

A *book* is a world artifact (engine-owned, stdlib, zero-dependency): an agent
records a lesson from its own experience, and the book stays on the shelf after
the author dies. A later, living agent standing by the library can *read* the
few books most relevant to its situation — re-living a predecessor's experience.

This is **horizontal/cultural inheritance** (fast, cumulative — the ratchet),
distinct from the **vertical/genetic** kind. It is deliberately thin: the engine
owns the books; the *smart recall model* (semantic search) is a swappable
upgrade that the memory architecture can provide later. The built-in relevance
here is a naive keyword overlap so the feature works — and is testable — with no
external dependency and no LLM.

Opt-in: a world only has a library store when one is injected
(``make_simulation(library=True)``); otherwise nothing changes, so the offline
baseline stays byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class TownLibrary:
    """A shelf of books (recorded lessons) that persists across deaths/runs."""

    books: list[dict] = field(default_factory=list)

    def write(self, day: int, author_id: str, author: str, text: str) -> dict | None:
        """Record a lesson. Identical lessons are de-duplicated so the shelf
        stays meaningful (returns the new book, or None if nothing was added)."""
        text = (text or "").strip()
        if not text:
            return None
        if any(b["text"] == text for b in self.books):
            return None
        book = {"day": day, "author_id": author_id, "author": author, "text": text}
        self.books.append(book)
        return book

    def read(self, query: str, k: int = 3) -> list[str]:
        """The few books most relevant to ``query`` (keyword overlap, then
        recency), rendered as plain strings for the brain to act on."""
        if not self.books:
            return []
        q = _tokens(query)
        ranked = sorted(
            self.books,
            key=lambda b: (len(q & _tokens(b["text"])), b["day"]),
            reverse=True,
        )
        return [f'{b["author"]} (day {b["day"]}): {b["text"]}' for b in ranked[:k]]

    def burn(self) -> int:
        """A fire destroys the public shelf. Knowledge is not unconditionally
        permanent — it survives only while its substrate does. (What people have
        already studied lives on in their own memory; only the record is lost.)
        Returns the number of books destroyed."""
        n = len(self.books)
        self.books = []
        return n

    def __len__(self) -> int:
        return len(self.books)
