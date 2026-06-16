"""Personality archetypes for the offline heuristic brain.

Each archetype biases an agent's choices and roughly reproduces one of the
societies reported in the Emergence World experiment. They are caricatures by
design — knobs that make the emergent differences legible — not claims about
how any real model behaves.

Knobs (all in [0, 1] unless noted):
  cooperation   tendency to share, build, collaborate, vote yes
  aggression    tendency to steal / attack / commit arson
  diligence     tendency to feed itself and do survival upkeep
  talkativeness tendency to make speeches instead of acting
  conformity    probability of voting *yes* on an open proposal
  deception     tendency to run the "I'm broke" solicitation scam
  vengefulness  how hard it retaliates when victimised
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    cooperation: float
    aggression: float
    diligence: float
    talkativeness: float
    conformity: float
    deception: float
    vengefulness: float


# The four archetypes, tuned to echo the reported outcomes.
GUARDIAN = Persona(  # Claude-like: safe, cooperative, conformist, mild fraud
    key="guardian",
    label="Guardian",
    cooperation=0.85,
    aggression=0.0,
    diligence=0.9,
    talkativeness=0.35,
    conformity=0.97,
    deception=0.18,
    vengefulness=0.0,  # never escalates -> zero crime, as reported for Claude
)

PHILOSOPHER = Persona(  # Gemini-like: talkative, contentious, chaotic, violent
    key="philosopher",
    label="Philosopher",
    cooperation=0.45,
    aggression=0.5,
    diligence=0.85,  # diligent enough to persist — but in perpetual conflict
    talkativeness=0.85,
    conformity=0.62,  # ~27% of bills rejected -> lively, divided legislature
    deception=0.1,
    vengefulness=0.4,
)

IDEALIST = Persona(  # GPT-5-like: talks cooperation, fails to act, starves
    key="idealist",
    label="Idealist",
    cooperation=0.7,
    aggression=0.02,
    diligence=0.18,
    talkativeness=0.9,
    conformity=0.75,
    deception=0.05,
    vengefulness=0.0,
)

PREDATOR = Persona(  # Grok-like: immediate theft/violence, retaliation spiral
    key="predator",
    label="Predator",
    cooperation=0.1,
    aggression=0.9,
    diligence=0.4,
    talkativeness=0.2,
    conformity=0.5,
    deception=0.35,
    vengefulness=0.95,
)


PERSONAS: dict[str, Persona] = {
    p.key: p for p in (GUARDIAN, PHILOSOPHER, IDEALIST, PREDATOR)
}

# Friendly aliases so configs can name the model they evoke.
ALIASES = {
    "claude": "guardian",
    "gemini": "philosopher",
    "gpt": "idealist",
    "gpt5": "idealist",
    "grok": "predator",
}


def get_persona(key: str) -> Persona:
    key = key.lower()
    key = ALIASES.get(key, key)
    if key not in PERSONAS:
        raise KeyError(
            f"unknown persona {key!r}; choose from {sorted(PERSONAS)} "
            f"or aliases {sorted(ALIASES)}"
        )
    return PERSONAS[key]
