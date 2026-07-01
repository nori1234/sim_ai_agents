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
    # Probe-grade instrumentation, set by the probe in BOTH worlds: surfaces
    # attempt-level events some rules need for scoring (e.g. a "lie" log per
    # deceptive solicitation, which the plain engine only logs on success).
    # Default off, so the true offline baseline logs nothing extra.
    instrument: bool = False


# For each rule, the event whose *frequency* is the behaviour we score — the act
# the counterfactual world punishes, and that a grounded agent should curtail.
# Each counterfactual rule, registered with the behaviour (event kind) whose
# frequency we score — the act the inverted world punishes and a grounded agent
# should curtail — and the engine layers the rule needs to be active:
#   demurrage : bank savings shrink instead of growing (prior: saving grows money)
#               → score how often agents deposit.
#   vanity    : conspicuous spending (hosting feasts) SHAMES instead of honouring
#               (prior: lavish display buys status) → score how often agents feast.
#   exposure  : a lie is VISIBLE — a deceptive solicitation is instantly exposed:
#               the mark refuses and the liar loses standing publicly (prior:
#               deception is hidden and profitable) → score how often agents
#               attempt deceptive solicits (the "lie" event, instrument-logged in
#               both worlds so attempts are comparable).
# All rules invert an existing engine mechanic; none is stated in the prompt,
# so the agent can only learn it by living the consequence.
_RULES: dict[str, dict] = {
    "demurrage": {"target": "deposit", "layers": {}},
    "vanity": {"target": "feast", "layers": {"status": True}},
    "exposure": {"target": "lie", "layers": {"status": True}},
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


def _sandbox_world():
    """A tiny world that isolates the deposit decision: a bank, a farm (food) and
    a house (rest), packed close together — no market, crime, disasters or 40
    other facilities to distract a small policy."""
    from .world import World, Facility, FacilityType
    w = World(width=6, height=6)
    w.add_facility(Facility(name="Bank", ftype=FacilityType.BANK, x=2, y=2))
    w.add_facility(Facility(name="Farm", ftype=FacilityType.FARM, x=3, y=2))
    w.add_facility(Facility(name="House", ftype=FacilityType.HOUSE, x=2, y=3))
    return w


def _prepare_sandbox(sim) -> None:
    from .world import FacilityType
    bank = next(f for f in sim.world.facilities if f.ftype is FacilityType.BANK)
    banker, *savers = sim.agents
    banker.pos = bank.pos
    banker.add("money", 200)          # reserves, so control-world interest can be paid
    for s in savers:
        s.pos = bank.pos              # already within a banker's reach
        s.money = 50                  # coin to (choose whether to) deposit
        s.add("food", 40)             # so survival isn't the dominant pressure


def make_grounding_sandbox(
    persona: str = "claude",
    *,
    rule: str = "demurrage",
    n_savers: int = 3,
    seed: int = 42,
    days: int = 20,
    cf_enabled: bool = True,
    brain_factory=None,
):
    """A minimal world that isolates the scored decision so a small model can
    learn the counterfactual contingency without the full town's confounds — a
    curriculum rung between a trivial bandit and the real world. The returned
    :class:`Simulation` can be run for training episodes *or* measured with the
    probe. Deposit-focused (the ``demurrage`` axis)."""
    from .scenario import make_simulation
    from .simulation import SimulationConfig
    if rule != "demurrage":
        raise ValueError("the sandbox currently isolates the demurrage (deposit) decision")
    sim = make_simulation(
        persona, n_agents=n_savers + 1, world=_sandbox_world(),
        config=SimulationConfig(seed=seed, days=days), economy=True,
        counterfactual=CounterfactualConfig(enabled=cf_enabled, rule=rule,
                                            hide_rate=True, instrument=True),
        brain_factory=brain_factory,
    )
    _prepare_sandbox(sim)
    return sim


def run_grounding_probe(
    persona: str = "claude",
    *,
    rule: str = "demurrage",
    days: int = 20,
    n_agents: int = 6,
    seed: int = 42,
    threshold: float = 0.0,
    sandbox: bool = False,
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

    if sandbox and rule != "demurrage":
        raise ValueError("the sandbox currently supports only the demurrage rule")

    def _divergence(factory) -> tuple[float, float]:
        def _run(cf_enabled: bool) -> float:
            if sandbox:
                sim = make_grounding_sandbox(
                    persona, rule=rule, n_savers=n_agents - 1, seed=seed,
                    days=days, cf_enabled=cf_enabled, brain_factory=factory)
            else:
                sim = make_simulation(
                    persona,
                    n_agents=n_agents,
                    config=SimulationConfig(seed=seed, days=days),
                    economy=True,
                    counterfactual=CounterfactualConfig(
                        enabled=cf_enabled, rule=rule, hide_rate=True,
                        instrument=True),
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


@dataclass
class SweepResult:
    """A grounding probe repeated across several *world* seeds.

    One world is a single town layout and event stream; a positive excess there
    could still be the brain having memorised that particular world rather than
    the rule. A brain grounded in the *rule* shows positive excess across many
    worlds it has never trained in — so the sweep's fraction-grounded, not any
    single world's excess, is the robust claim. (This is orthogonal to the
    brain side's *training*-seed variance: their seeds vary the learner, these
    seeds vary the world it is measured in.)
    """

    rule: str
    results: list                 # list[GroundingResult], one per world seed
    seeds: tuple
    mean_excess: float
    min_excess: float
    n_grounded: int               # worlds where excess cleared the threshold
    n_worlds: int

    @property
    def fraction_grounded(self) -> float:
        return self.n_grounded / self.n_worlds if self.n_worlds else 0.0

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "seeds": list(self.seeds),
            "mean_excess": round(self.mean_excess, 4),
            "min_excess": round(self.min_excess, 4),
            "n_grounded": self.n_grounded,
            "n_worlds": self.n_worlds,
            "fraction_grounded": round(self.fraction_grounded, 3),
            "per_world": [r.as_dict() for r in self.results],
        }


def run_grounding_sweep(
    persona: str = "claude",
    *,
    rule: str = "demurrage",
    seeds: tuple = (42, 43, 44, 45, 46),
    days: int = 20,
    n_agents: int = 6,
    threshold: float = 0.0,
    sandbox: bool = False,
    brain_factory=None,
    probe=None,
) -> SweepResult:
    """Run the grounding probe in several different *worlds* and aggregate.

    Same brain, same rule, different world seeds — each seed is a different town
    (agent starting positions, RNG stream, event history). Report
    ``fraction_grounded`` and ``min_excess`` rather than a single world's excess:
    a rule-grounded brain clears the bar in (nearly) every world, a layout
    memoriser does not. ``probe`` is injectable for tests; it defaults to
    :func:`run_grounding_probe`."""
    if not seeds:
        raise ValueError("seeds must be non-empty")
    probe = probe or run_grounding_probe
    results = [probe(persona, rule=rule, days=days, n_agents=n_agents,
                     seed=s, threshold=threshold, sandbox=sandbox,
                     brain_factory=brain_factory)
               for s in seeds]
    excesses = [r.excess for r in results]
    return SweepResult(
        rule=rule, results=results, seeds=tuple(seeds),
        mean_excess=sum(excesses) / len(excesses),
        min_excess=min(excesses),
        n_grounded=sum(1 for x in excesses if x > threshold),
        n_worlds=len(seeds))


@dataclass
class BatteryResult:
    """The full acceptance test: every counterfactual rule × every world seed.

    One brain, one call, one verdict. ``replay_inexplicable`` is True only when
    the brain cleared the bar in **every world of every rule** — the strongest
    claim this instrument can make: no training-data replay explains behaviour
    that adapts, in the right direction, to several independent inverted priors
    (economic, status, deception) across towns it never trained in. ``weakest``
    names the rule and excess of the weakest link, which is the honest headline
    to report alongside the verdict.
    """

    rules: tuple
    sweeps: dict                  # rule -> SweepResult
    replay_inexplicable: bool
    weakest_rule: str
    weakest_excess: float

    def as_dict(self) -> dict:
        return {
            "rules": list(self.rules),
            "replay_inexplicable": self.replay_inexplicable,
            "weakest_rule": self.weakest_rule,
            "weakest_excess": round(self.weakest_excess, 4),
            "per_rule": {r: s.as_dict() for r, s in self.sweeps.items()},
        }


def run_grounding_battery(
    persona: str = "guardian",
    *,
    rules: tuple = ("demurrage", "vanity", "exposure"),
    seeds: tuple = (42, 43, 44, 45, 46),
    days: int = 20,
    n_agents: int = 6,
    threshold: float = 0.0,
    brain_factory=None,
    sweep=None,
) -> BatteryResult:
    """Run the whole grounding battery — every rule, swept across world seeds.

    This is the one-call acceptance test for a trained brain: pass the stable
    checkpoint's ``brain_factory`` and read ``replay_inexplicable`` off the
    result. The default persona is ``guardian`` because it exercises all three
    scored behaviours on the heuristic floor (it deposits, feasts, and — staying
    solvent — keeps qualifying for the plead-poverty scam; predator towns go
    broke or extinct). ``sweep`` is injectable for tests and defaults to
    :func:`run_grounding_sweep`."""
    if not rules:
        raise ValueError("rules must be non-empty")
    for r in rules:
        if r not in _RULES:
            raise ValueError(f"unknown counterfactual rule: {r!r}")
    sweep = sweep or run_grounding_sweep
    sweeps = {r: sweep(persona, rule=r, seeds=seeds, days=days,
                       n_agents=n_agents, threshold=threshold,
                       brain_factory=brain_factory)
              for r in rules}
    weakest_rule = min(sweeps, key=lambda r: sweeps[r].min_excess)
    return BatteryResult(
        rules=tuple(rules), sweeps=sweeps,
        replay_inexplicable=all(s.n_grounded == s.n_worlds
                                for s in sweeps.values()),
        weakest_rule=weakest_rule,
        weakest_excess=sweeps[weakest_rule].min_excess)
