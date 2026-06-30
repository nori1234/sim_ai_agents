"""Periodically probe a *training* brain's grounding and log the trend.

The grounding probe (``emergence.grounding``) answers "is this brain grounded or
replaying?" for a snapshot of a brain. During a long training run we want that
answer *as a curve* — does grounding emerge and strengthen as the developmental
brain learns? This monitor is the thin orchestration the ``llm_model_agi``
learning loop calls every N epochs/checkpoints to get that curve.

Usage (on the training side)::

    from emergence.grounding_monitor import GroundingMonitor

    monitor = GroundingMonitor(persona="claude", every=10,
                               on_result=lambda e: trainer.log(e))
    for epoch in range(n_epochs):
        train_one_epoch(...)
        # brain_factory must yield brains reflecting the CURRENT trained weights
        monitor.maybe_probe(epoch, brain_factory=current_brain_factory)

    monitor.to_jsonl("grounding.jsonl")
    assert monitor.improving()        # grounding excess trended up over training

The probe runs its own fresh control/counterfactual simulations (it does not
touch the training run), and the headline metric logged is ``excess`` — the
brain's divergence *over the non-learning heuristic floor* — never the raw
divergence (which is non-zero from mechanical feedback alone). Stdlib only; no
torch. The probe function is injectable for testing.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from .grounding import GroundingResult, run_grounding_probe


class GroundingMonitor:
    """Runs the grounding probe on a cadence and keeps the ``excess`` time series."""

    def __init__(
        self,
        persona: str = "claude",
        *,
        rule: str = "demurrage",
        days: int = 20,
        n_agents: int = 6,
        seed: int = 42,
        every: int = 1,
        threshold: float = 0.0,
        on_result: Optional[Callable[[dict], None]] = None,
        probe: Callable[..., GroundingResult] = run_grounding_probe,
    ):
        if every < 1:
            raise ValueError("every must be >= 1")
        self.persona = persona
        self.rule = rule
        self.days = days
        self.n_agents = n_agents
        self.seed = seed
        self.every = every
        self.threshold = threshold
        self._on_result = on_result
        self._probe = probe
        self.history: list[dict] = []

    # -- cadence -------------------------------------------------------------
    def due(self, epoch: int) -> bool:
        """Whether a probe should run at this epoch (every ``every`` epochs from 0)."""
        return epoch % self.every == 0

    def maybe_probe(self, epoch: int, brain_factory) -> Optional[GroundingResult]:
        """Probe iff this epoch is due; otherwise return ``None`` and do nothing."""
        if not self.due(epoch):
            return None
        return self.probe(epoch, brain_factory)

    # -- the probe ----------------------------------------------------------
    def probe(self, epoch: int, brain_factory) -> GroundingResult:
        """Run the grounding probe for the current brain and record the result.

        ``brain_factory`` must produce brains reflecting the brain's *current*
        trained state, so the curve tracks learning."""
        result = self._probe(
            self.persona, rule=self.rule, days=self.days, n_agents=self.n_agents,
            seed=self.seed, threshold=self.threshold, brain_factory=brain_factory)
        entry = {"epoch": epoch, **result.as_dict()}
        self.history.append(entry)
        if self._on_result is not None:
            self._on_result(entry)
        return result

    # -- read-out -----------------------------------------------------------
    def latest(self) -> Optional[dict]:
        return self.history[-1] if self.history else None

    def excess_series(self) -> list[float]:
        return [e["excess"] for e in self.history]

    def improving(self) -> bool:
        """True if grounding strengthened over training: the mean ``excess`` of the
        second half of the series exceeds that of the first half. Needs >= 2 probes."""
        xs = self.excess_series()
        if len(xs) < 2:
            return False
        mid = len(xs) // 2
        first, second = xs[:mid] or xs[:1], xs[mid:]
        return (sum(second) / len(second)) > (sum(first) / len(first))

    def to_jsonl(self, path: str) -> None:
        """Write the history as one JSON object per line (a training-log artifact)."""
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.history:
                fh.write(json.dumps(entry) + "\n")
