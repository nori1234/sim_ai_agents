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

import math
from dataclasses import dataclass, field

from .grounding_stats import (
    linear_regression,
    paired_bootstrap_ci,
    sign_test_p,
    wilcoxon_signed_rank_p,
)


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
    # The floor is itself an estimate (a finite run's count of a scored
    # behaviour), so it carries its own sampling noise — the exact confound a
    # floor-heavy world can smuggle into `excess`. `floor_rollouts` > 1
    # averages the floor over that many independent world draws (see
    # run_grounding_probe) to shrink that noise; `floor_divergence_std` is the
    # spread across those draws (0.0 when floor_rollouts == 1 — nothing to
    # measure spread over).
    floor_rollouts: int = 1
    floor_divergence_std: float = 0.0

    @property
    def conclusive(self) -> bool:
        """False when the tested brain never exercised the behaviour in either
        world (control_rate == counterfactual_rate == 0). Then divergence is 0 and
        the excess is just the negated heuristic floor — pure noise, not evidence.
        Such a result is *inconclusive* (the transfer test could not discriminate),
        not a verdict of replay, and must not be counted as either. The first
        real-engine battery hit exactly this for vanity/exposure: the freshly
        trained policy never feasts or lies in the full 44-action town."""
        return (self.control_rate > 0.0) or (self.counterfactual_rate > 0.0)

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "target_behaviour": self.target,
            "control_rate": round(self.control_rate, 4),
            "counterfactual_rate": round(self.counterfactual_rate, 4),
            "divergence": round(self.divergence, 4),
            "floor_divergence": round(self.floor_divergence, 4),
            "excess": round(self.excess, 4),
            "conclusive": self.conclusive,
            "verdict": self.verdict,
            "days": self.days,
            "n_agents": self.n_agents,
            "floor_rollouts": self.floor_rollouts,
            "floor_divergence_std": round(self.floor_divergence_std, 4),
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
    floor_rollouts: int = 1,
    floor_seed_stride: int = 97_003,
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

    ``floor_rollouts`` (default 1, current behaviour) widens the floor estimate
    from a single draw at ``seed`` to the mean of ``floor_rollouts`` independent
    heuristic-only draws — ``seed`` itself plus ``floor_rollouts - 1`` further
    worlds offset by ``floor_seed_stride`` (a large stride, chosen so the extra
    floor-only worlds never collide with a battery's held-out or training seed
    range). The floor is a finite run's *count* of a scored behaviour, so it is
    itself noisy — this reduces that noise without touching the tested brain's
    own run at ``seed``, which is unaffected. See ``floor_divergence_std`` on the
    result for the spread across the draws.
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

    def _divergence(factory, world_seed: int) -> tuple[float, float]:
        def _run(cf_enabled: bool) -> float:
            if sandbox:
                sim = make_grounding_sandbox(
                    persona, rule=rule, n_savers=n_agents - 1, seed=world_seed,
                    days=days, cf_enabled=cf_enabled, brain_factory=factory)
            else:
                sim = make_simulation(
                    persona,
                    n_agents=n_agents,
                    config=SimulationConfig(seed=world_seed, days=days),
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

    control, divergence = _divergence(brain_factory, seed)
    if brain_factory is None:
        # The tested brain is the heuristic itself; the floor is the same run.
        floor, floor_std = divergence, 0.0
        counterfactual_rate = control - divergence
    else:
        # Recover the tested brain's counterfactual rate for reporting.
        counterfactual_rate = control - divergence
        floor_seeds = [seed] if floor_rollouts <= 1 else (
            [seed] + [seed + (i + 1) * floor_seed_stride
                     for i in range(floor_rollouts - 1)])
        floor_samples = [_divergence(None, s)[1] for s in floor_seeds]
        floor = sum(floor_samples) / len(floor_samples)
        floor_std = (math.sqrt(sum((x - floor) ** 2 for x in floor_samples)
                               / len(floor_samples))
                    if len(floor_samples) > 1 else 0.0)

    excess = divergence - floor
    tested_did_behaviour = (control > 0.0) or (counterfactual_rate > 0.0)
    if not tested_did_behaviour:
        # The brain never performed the scored behaviour in either world, so the
        # excess is just the negated heuristic floor — nothing to read.
        verdict = "inconclusive (behaviour never occurred)"
    elif brain_factory is None:
        verdict = "baseline (heuristic floor)"
    elif excess > threshold:
        verdict = "grounded (exceeds heuristic floor)"
    else:
        verdict = "replay/insensitive (within heuristic floor)"
    return GroundingResult(
        rule=rule, target=target, control_rate=control,
        counterfactual_rate=counterfactual_rate, divergence=divergence,
        floor_divergence=floor, excess=excess, verdict=verdict,
        days=days, n_agents=n_agents,
        floor_rollouts=len(floor_seeds) if brain_factory is not None else 1,
        floor_divergence_std=floor_std)


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
    n_grounded: int               # worlds that were conclusive AND cleared the threshold
    n_worlds: int
    n_conclusive: int             # worlds where the behaviour actually occurred
    # Paired statistics over the *conclusive* worlds' excess, one-sided for
    # H1: excess > 0 (see emergence.grounding_stats). fraction_grounded is a
    # hard-threshold count — flip one borderline world and it jumps by
    # 1/n_worlds; these read the same numbers as a real hypothesis test, and
    # are far less sensitive to any one world. Neither replaces the other —
    # both are reported; replay_inexplicable still gates on fraction_grounded.
    sign_test_p: float = 1.0
    wilcoxon_p: float = 1.0
    bootstrap_ci_mean_excess: tuple = (0.0, 0.0)
    # Diagnostic, not a verdict: regress each conclusive world's raw
    # divergence on its floor_divergence and test the residual against zero.
    # `excess` assumes the floor confound is purely additive (slope 1); this
    # is immune to that assumption — it is orthogonal to floor_divergence by
    # construction regardless of the fitted slope. See floor_regression().
    floor_regression: dict = field(default_factory=dict)

    @property
    def fraction_grounded(self) -> float:
        return self.n_grounded / self.n_worlds if self.n_worlds else 0.0

    @property
    def conclusive(self) -> bool:
        """The sweep measured something in every world — else fraction_grounded
        is not a grounding statement (the behaviour simply never happened)."""
        return self.n_conclusive == self.n_worlds

    @property
    def grounded_paired(self) -> bool:
        """A paired-test alternative headline to fraction_grounded: True iff the
        one-sided Wilcoxon signed-rank p-value (excess > 0 across conclusive
        worlds) clears the conventional 0.05 bar. Still requires `conclusive`
        to mean anything — an inconclusive rule can look "significant" on pure
        floor noise just as easily as fraction_grounded can."""
        return self.wilcoxon_p < 0.05

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "seeds": list(self.seeds),
            "mean_excess": round(self.mean_excess, 4),
            "min_excess": round(self.min_excess, 4),
            "n_grounded": self.n_grounded,
            "n_conclusive": self.n_conclusive,
            "n_worlds": self.n_worlds,
            "fraction_grounded": round(self.fraction_grounded, 3),
            "conclusive": self.conclusive,
            "sign_test_p": round(self.sign_test_p, 4),
            "wilcoxon_p": round(self.wilcoxon_p, 4),
            "grounded_paired": self.grounded_paired,
            "bootstrap_ci_mean_excess": [round(self.bootstrap_ci_mean_excess[0], 4),
                                        round(self.bootstrap_ci_mean_excess[1], 4)],
            "floor_regression": self.floor_regression,
            "per_world": [r.as_dict() for r in self.results],
        }


def floor_regression_diagnostic(results: list) -> dict:
    """Regress each *conclusive* world's raw ``divergence`` on its
    ``floor_divergence`` and test the residual against zero (one-sided,
    excess > 0).

    ``excess = divergence - floor_divergence`` implicitly assumes the floor
    confound is additive with slope 1. If the true relationship has a
    different slope, `excess` can still carry a floor-driven trend across
    worlds. The residual of this regression is orthogonal to
    ``floor_divergence`` *by construction*, regardless of the fitted slope —
    it is the one statistic in this module that is immune to a floor
    confound of any linear form, not just an additive offset. Needs >= 3
    conclusive worlds to fit a meaningful line; fewer returns a note instead
    of a spurious fit.
    """
    conclusive = [r for r in results if r.conclusive]
    if len(conclusive) < 3:
        return {"n": len(conclusive),
                "note": "too few conclusive worlds for a regression (need >= 3)"}
    xs = [r.floor_divergence for r in conclusive]
    ys = [r.divergence for r in conclusive]
    slope, intercept = linear_regression(xs, ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    return {
        "n": len(conclusive),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "residuals": [round(r, 4) for r in residuals],
        "residual_sign_p": round(sign_test_p(residuals), 4),
        "residual_wilcoxon_p": round(wilcoxon_signed_rank_p(residuals), 4),
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
    floor_rollouts: int = 1,
    floor_seed_stride: int = 97_003,
    brain_factory=None,
    probe=None,
) -> SweepResult:
    """Run the grounding probe in several different *worlds* and aggregate.

    Same brain, same rule, different world seeds — each seed is a different town
    (agent starting positions, RNG stream, event history). Report
    ``fraction_grounded`` and ``min_excess`` rather than a single world's excess:
    a rule-grounded brain clears the bar in (nearly) every world, a layout
    memoriser does not. ``probe`` is injectable for tests; it defaults to
    :func:`run_grounding_probe`.

    Also reports paired statistics over the conclusive worlds' excess
    (``sign_test_p``, ``wilcoxon_p``, ``bootstrap_ci_mean_excess`` — see
    :mod:`emergence.grounding_stats`) and a floor-regression diagnostic
    (``floor_regression`` — see :func:`floor_regression_diagnostic`), both
    read-outs on the *same* per-world numbers as ``fraction_grounded``, not a
    second experiment — neither changes what counts as ``n_grounded`` or
    ``replay_inexplicable``.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    probe = probe or run_grounding_probe
    results = [probe(persona, rule=rule, days=days, n_agents=n_agents,
                     seed=s, threshold=threshold, sandbox=sandbox,
                     floor_rollouts=floor_rollouts,
                     floor_seed_stride=floor_seed_stride,
                     brain_factory=brain_factory)
               for s in seeds]
    excesses = [r.excess for r in results]
    conclusive_excesses = [r.excess for r in results if r.conclusive]
    return SweepResult(
        rule=rule, results=results, seeds=tuple(seeds),
        mean_excess=sum(excesses) / len(excesses),
        min_excess=min(excesses),
        # A world only counts as grounded if it was conclusive AND cleared the bar
        # — an inconclusive world's excess is floor noise that can drift positive.
        n_grounded=sum(1 for r in results if r.conclusive and r.excess > threshold),
        n_conclusive=sum(1 for r in results if r.conclusive),
        n_worlds=len(seeds),
        sign_test_p=sign_test_p(conclusive_excesses),
        wilcoxon_p=wilcoxon_signed_rank_p(conclusive_excesses),
        bootstrap_ci_mean_excess=paired_bootstrap_ci(conclusive_excesses),
        floor_regression=floor_regression_diagnostic(results))


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
    inconclusive_rules: tuple     # rules whose behaviour never occurred in some world
    # A paired-test alternative to `replay_inexplicable`: every rule's
    # Wilcoxon signed-rank test (excess > 0 across conclusive worlds) clears
    # 0.05. Kept alongside, not instead of, the hard-threshold verdict — see
    # SweepResult.grounded_paired.
    replay_inexplicable_paired: bool = False

    @property
    def conclusive(self) -> bool:
        """Every rule measured something in every world. When False, a
        ``replay_inexplicable=False`` verdict may just mean the behaviours never
        happened (e.g. a policy that never feasts in the full town) — not replay."""
        return not self.inconclusive_rules

    def as_dict(self) -> dict:
        return {
            "rules": list(self.rules),
            "replay_inexplicable": self.replay_inexplicable,
            "replay_inexplicable_paired": self.replay_inexplicable_paired,
            "conclusive": self.conclusive,
            "inconclusive_rules": list(self.inconclusive_rules),
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
    sandbox: bool = False,
    floor_rollouts: int = 1,
    floor_seed_stride: int = 97_003,
    brain_factory=None,
    sweep=None,
) -> BatteryResult:
    """Run the whole grounding battery — every rule, swept across world seeds.

    This is the one-call acceptance test for a trained brain: pass the stable
    checkpoint's ``brain_factory`` and read ``replay_inexplicable`` off the
    result. The default persona is ``guardian`` because it exercises all three
    scored behaviours on the heuristic floor (it deposits, feasts, and — staying
    solvent — keeps qualifying for the plead-poverty scam; predator towns go
    broke or extinct). ``sandbox=True`` measures in the minimal world where the
    behaviour is dense enough to be conclusive (currently ``demurrage`` only).
    ``floor_rollouts``/``floor_seed_stride`` are forwarded to every probe — see
    :func:`run_grounding_probe`. ``sweep`` is injectable for tests and defaults
    to :func:`run_grounding_sweep`."""
    if not rules:
        raise ValueError("rules must be non-empty")
    for r in rules:
        if r not in _RULES:
            raise ValueError(f"unknown counterfactual rule: {r!r}")
    sweep = sweep or run_grounding_sweep
    sweeps = {r: sweep(persona, rule=r, seeds=seeds, days=days,
                       n_agents=n_agents, threshold=threshold, sandbox=sandbox,
                       floor_rollouts=floor_rollouts,
                       floor_seed_stride=floor_seed_stride,
                       brain_factory=brain_factory)
              for r in rules}
    weakest_rule = min(sweeps, key=lambda r: sweeps[r].min_excess)
    inconclusive = tuple(r for r, s in sweeps.items() if not s.conclusive)
    return BatteryResult(
        rules=tuple(rules), sweeps=sweeps,
        # The strongest claim needs every rule conclusive AND every world grounded.
        # An inconclusive rule can never earn it — nothing was measured there.
        replay_inexplicable=(not inconclusive and
                             all(s.n_grounded == s.n_worlds for s in sweeps.values())),
        replay_inexplicable_paired=(not inconclusive and
                                    all(s.grounded_paired for s in sweeps.values())),
        weakest_rule=weakest_rule,
        weakest_excess=sweeps[weakest_rule].min_excess,
        inconclusive_rules=inconclusive)
