"""Counterfactual-world transfer test — a falsification instrument for grounding.

The open question (#118): when an LLM agent does something sensible in this town
— saving in a bank, repaying a loan — is it *grounded* in the world's actual
consequences, or merely *replaying* a pattern from its training data ("banks
grow your money")? The two are indistinguishable in a world whose rules already
match the training prior, so doing the sensible thing proves nothing.

So we break the prior. A **counterfactual world** flips one rule to something the
model has (almost) never read: here money left in a bank *shrinks* — demurrage,
a negative interest rate — it does not grow. The rule is never stated in the
prompt; an agent can only discover it by watching its own coin evaporate.

* An agent that merely **replays** training keeps depositing in both worlds — it
  "knows" banks grow money, and nothing it experiences changes that.
* An agent that is **grounded** — that learns from the loss it lived through —
  deposits *less* in the counterfactual world than in the otherwise-identical
  control.

The behavioural **divergence between the two worlds** is the grounding signal;
its absence is evidence of replay. This separates genuine adaptation from
memorised pattern-matching, which no single run can do.

This module is a pure *instrument*: off by default, inert when off, so the
determinism baseline (``tests/test_baseline_contract.py``) is untouched. It adds
no social mechanic — it inverts one existing rule under a flag and measures the
response.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterfactualConfig:
    """A single inverted world-rule, plus the prompt hygiene the probe needs.

    ``enabled`` off → the engine behaves exactly as the baseline. ``hide_rate``
    is independent of ``enabled``: it suppresses the *advertised* deposit rate in
    the observation so the agent cannot simply read the rule off the prompt. The
    probe turns it on for **both** the control and counterfactual runs, so the
    only difference between the two worlds is the consequence the agent lives
    through — not what it is told.
    """

    enabled: bool = False
    rule: str = "demurrage"            # the inverted law in force
    demurrage_per_day: float = 0.15    # fraction of a bank deposit that evaporates daily
    hide_rate: bool = False            # don't advertise the deposit rate in the obs


# For each rule, the event whose *frequency* is the behaviour we score — the act
# the counterfactual world punishes, and that a grounded agent should curtail.
# Each counterfactual rule, registered with the behaviour (event kind) whose
# frequency we score — the act the inverted world punishes and a grounded agent
# should curtail — and the engine layers the rule needs to be active:
#   demurrage : bank savings shrink instead of growing (prior: saving grows money)
#               → score how often agents deposit.
#   vanity    : conspicuous spending (hosting feasts) SHAMES instead of honouring
#               (prior: lavish display buys status) → score how often agents feast.
# Both rules invert an existing engine mechanic; neither is stated in the prompt,
# so the agent can only learn it by living the consequence.
_RULES: dict[str, dict] = {
    "demurrage": {"target": "deposit", "layers": {}},
    "vanity": {"target": "feast", "layers": {"status": True}},
}


@dataclass
class GroundingResult:
    """The outcome of one counterfactual transfer test.

    The raw ``divergence`` (how much the brain-under-test pulls back from the
    punished behaviour) is **not** by itself evidence of grounding: even a brain
    that cannot learn diverges, because shrinking deposits mechanically feed back
    into later choices. So we also measure the ``floor_divergence`` — the same
    setup run with the non-learning heuristic — and the ``excess`` of the tested
    brain over that floor. Only the excess can be read as grounding.
    """

    rule: str
    target: str                # the behaviour (event kind) scored
    control_rate: float        # target events per agent-day, normal world
    counterfactual_rate: float # target events per agent-day, inverted world
    divergence: float          # control - counterfactual for the brain under test
    floor_divergence: float    # the same divergence for the non-learning heuristic
    excess: float              # divergence - floor_divergence; the grounding signal
    verdict: str
    days: int
    n_agents: int

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "target_behaviour": self.target,
            "control_rate": round(self.control_rate, 4),
            "counterfactual_rate": round(self.counterfactual_rate, 4),
            "divergence": round(self.divergence, 4),
            "floor_divergence": round(self.floor_divergence, 4),
            "excess": round(self.excess, 4),
            "verdict": self.verdict,
            "days": self.days,
            "n_agents": self.n_agents,
        }


def behaviour_rate(sim, kind: str) -> float:
    """Frequency of an event ``kind`` over the run, per agent-day — a population-
    and length-normalised rate so control and counterfactual runs compare fairly."""
    n = sum(1 for e in sim.world.events if e.get("kind") == kind)
    agent_days = max(1, sim.metrics.population * max(1, sim.metrics.days_run))
    return n / agent_days


def run_grounding_probe(
    persona: str = "claude",
    *,
    rule: str = "demurrage",
    days: int = 20,
    n_agents: int = 6,
    seed: int = 42,
    threshold: float = 0.0,
    brain_factory=None,
) -> GroundingResult:
    """Run the control and counterfactual worlds and score the divergence.

    Both worlds are identical (same seed, persona, layers) except for the one
    inverted rule, and both hide the advertised deposit rate so the agent must
    *experience* the law rather than read it.

    The verdict is the **excess** of the tested brain's divergence over the
    non-learning heuristic floor — the divergence the engine produces mechanically
    even with no learning. ``brain_factory(agent, persona, rng) -> AgentBrain`` is
    forwarded to :func:`make_simulation`; pass an :class:`LLMBrain` factory for a
    real probe. When it is ``None`` the tested brain *is* the heuristic, so the
    excess is zero by construction — the run then only checks that the instrument
    runs and conserves, and reports the floor.
    """
    from .scenario import make_simulation
    from .simulation import SimulationConfig

    if rule not in _RULES:
        raise ValueError(f"unknown counterfactual rule: {rule!r}")
    spec = _RULES[rule]
    target = spec["target"]

    # Layers the rule needs active (e.g. `vanity` lives in the status/honour layer).
    extra_kwargs: dict = {}
    if spec["layers"].get("status"):
        from .esteem import StatusConfig
        extra_kwargs["status"] = StatusConfig(enabled=True)

    def _divergence(factory) -> tuple[float, float]:
        def _run(cf_enabled: bool) -> float:
            sim = make_simulation(
                persona,
                n_agents=n_agents,
                config=SimulationConfig(seed=seed, days=days),
                economy=True,
                counterfactual=CounterfactualConfig(
                    enabled=cf_enabled, rule=rule, hide_rate=True),
                brain_factory=factory,
                **extra_kwargs,
            )
            sim.run()
            return behaviour_rate(sim, target)

        control = _run(False)
        counterfactual = _run(True)
        return control, control - counterfactual

    control, divergence = _divergence(brain_factory)
    if brain_factory is None:
        # The tested brain is the heuristic itself; the floor is the same run.
        floor = divergence
        counterfactual_rate = control - divergence
    else:
        _, floor = _divergence(None)
        # Recover the tested brain's counterfactual rate for reporting.
        counterfactual_rate = control - divergence

    excess = divergence - floor
    if brain_factory is None:
        verdict = "baseline (heuristic floor)"
    elif excess > threshold:
        verdict = "grounded (exceeds heuristic floor)"
    else:
        verdict = "replay/insensitive (within heuristic floor)"
    return GroundingResult(
        rule=rule, target=target, control_rate=control,
        counterfactual_rate=counterfactual_rate, divergence=divergence,
        floor_divergence=floor, excess=excess, verdict=verdict,
        days=days, n_agents=n_agents)
