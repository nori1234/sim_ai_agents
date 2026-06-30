"""Extrinsic reward from observation deltas — pure standard library, no torch.

The engine has *no* reward API, and the build brief forbids adding one: a
developmental brain must derive its own scalar reward from how its situation
changed between two consecutive observations. Everything needed is already in the
observation handed to ``decide`` —

* **survival**   : ``self_view["energy"]`` (0..100)
* **material**   : ``self_view["money"]`` (conserved coin)
* **social**     : ``self_view["reputation"]`` if present, else the mean trust the
                   agent currently extends to its neighbours (``others[i]["trust"]``).
                   There is no single "trust toward me" scalar in the observation,
                   so reputation is the faithful standing signal; the trust mean is
                   only a fallback for layers where reputation is inert.

Curiosity / prediction-error (the intrinsic term in the brief) is computed *inside*
the neural agent against its world-model, not here — this module is only the
external, world-grounded part of the reward.

Works on either an :class:`~emergence.observation.Observation` or a plain ``dict``
with the same keys, so it is trivially unit-testable without the engine.
"""

from __future__ import annotations

from typing import Any

# Energy spans 0..100 and money/reputation are smaller, so energy gets the
# smallest per-unit weight. These are starting points; the neural agent may scale
# the returned scalar however it likes.
DEFAULT_WEIGHTS = {"energy": 0.02, "money": 0.05, "social": 0.10}


def _self_view(obs: Any) -> dict:
    sv = getattr(obs, "self_view", None)
    if sv is None and isinstance(obs, dict):
        sv = obs.get("self_view", {})
    return sv or {}


def _others(obs: Any) -> list:
    others = getattr(obs, "others", None)
    if others is None and isinstance(obs, dict):
        others = obs.get("others", [])
    return others or []


def _economy(obs: Any) -> dict:
    econ = getattr(obs, "economy", None)
    if econ is None and isinstance(obs, dict):
        econ = obs.get("economy")
    return econ if isinstance(econ, dict) else {}


def _wealth(obs: Any) -> float:
    """Spendable + banked wealth. Bank deposits are *claims* (in
    ``economy.my_deposits``), not coin in ``self_view.money`` — so a deposit moves
    wealth between the two without changing it, and a rule that shrinks deposits
    (demurrage) shows up as a real wealth loss. Counting deposits here is what
    gives the demurrage counterfactual a reward signal at all; without it the
    penalty lives only in a memory line and RL has no gradient to learn it."""
    money = float(_self_view(obs).get("money", 0.0))
    deposits = sum(float(d.get("amount", 0))
                   for d in _economy(obs).get("my_deposits", []) or [])
    return money + deposits


def _social_signal(obs: Any) -> float:
    """Standing in the eyes of others: reputation when the status layer surfaces
    it, otherwise the mean trust this agent extends to its neighbours."""
    sv = _self_view(obs)
    if "reputation" in sv:
        return float(sv["reputation"])
    trusts = [float(o.get("trust", 0.0)) for o in _others(obs)]
    return sum(trusts) / len(trusts) if trusts else 0.0


def survival_reward(prev: Any, cur: Any, weights: dict | None = None) -> float:
    """The world-grounded reward for the transition ``prev -> cur``.

    A weighted sum of the change in energy, wealth (coin + bank deposits) and
    social standing. Positive when the agent grew safer, richer or better
    regarded; negative when it declined. Returns ``0.0`` if either observation
    lacks the fields, so a missing layer never throws."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    psv, csv = _self_view(prev), _self_view(cur)
    d_energy = float(csv.get("energy", 0.0)) - float(psv.get("energy", 0.0))
    d_wealth = _wealth(cur) - _wealth(prev)        # coin + deposits, so demurrage bites
    d_social = _social_signal(cur) - _social_signal(prev)
    return w["energy"] * d_energy + w["money"] * d_wealth + w["social"] * d_social
