"""Rumour propagation (情報・噂) — hearsay that carries reputation and
misinformation beyond eyewitnesses.

Before this, communication (``SAY``/``SPEAK``/``PREACH``) only ever affected
whoever was in range *this tick*; nothing persisted beyond that, and nothing
about a third party ever reached anyone who wasn't there to see it firsthand.
This adds a thin channel: a ``SAY`` can carry a *claim* about an absent third
party (a sentiment, not a scripted narrative), which nearby listeners may
adopt — nudging their own trust of that party, and (with the status layer)
the party's public reputation — weighted by how much the listener trusts the
*speaker*. Distortion en route lets misinformation emerge without being
authored: a claim isn't guaranteed to survive the retelling intact.

Opt-in via :data:`RumourConfig.enabled` (default ``False``); nothing here
fires when off, so the four-society baseline is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RumourConfig:
    enabled: bool = False

    hearing_radius: int = 3          # how far a claim carries to bystanders
    min_speaker_trust: float = 0.0   # a claim from someone this distrusted persuades no one
    trust_weight: float = 0.3        # how much a believed claim moves a listener's trust
    rep_weight: float = 1.0          # how much it moves the subject's public reputation
    distortion_chance: float = 0.15  # a claim may mutate/exaggerate in transit (misinformation)
    distortion_strength: float = 0.6  # how far a distorted claim can drift from the original
    gossip_chance: float = 0.02      # heuristic: per-tick chance an agent spontaneously gossips
