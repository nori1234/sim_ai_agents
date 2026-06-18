"""Record & replay for LLM-driven runs — the reproducibility backbone.

The engine is already deterministic from its ``seed``; the only non-determinism
is the model call. Rather than chase determinism with ``temperature=0`` (not
guaranteed, and it flattens the emergent richness), we **record every call** —
``(system, user) -> response`` — and can **replay** it bit-exactly later: free,
offline, hardware- and model-drift-independent. A recorded run is a shareable,
auditable research artifact.

Both wrappers have the LLM client shape ``client(system, user) -> str`` so they
slot in wherever a client does, in front of a live HTTP client or a mock.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

ClientFn = Callable[[str, str], str]


def key(system: str, user: str) -> str:
    """A stable content hash of one prompt — the transcript key."""
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()


class RecordingClient:
    """Wrap a live client; remember every ``prompt -> response`` in ``transcript``."""

    def __init__(self, inner: ClientFn, transcript: dict[str, str]):
        self._inner = inner
        self.transcript = transcript

    def __call__(self, system: str, user: str) -> str:
        resp = self._inner(system, user)
        self.transcript[key(system, user)] = resp
        return resp


class ReplayClient:
    """Serve responses from a recorded ``transcript``. A miss optionally falls
    through to ``inner`` (recording it); with no ``inner`` a miss raises, so the
    caller (the LLM brain) degrades to its heuristic understudy."""

    def __init__(self, transcript: dict[str, str], inner: Optional[ClientFn] = None):
        self.transcript = dict(transcript)
        self._inner = inner
        self.hits = 0
        self.misses = 0

    def __call__(self, system: str, user: str) -> str:
        k = key(system, user)
        if k in self.transcript:
            self.hits += 1
            return self.transcript[k]
        self.misses += 1
        if self._inner is not None:
            resp = self._inner(system, user)
            self.transcript[k] = resp
            return resp
        raise KeyError("no recorded response for this prompt")
