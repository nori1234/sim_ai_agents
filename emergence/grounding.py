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
from typing import Optional

from .grounding_stats import (
    linear_regression,
    paired_bootstrap_ci,
    regression_slope_bootstrap_ci,
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
#
# Sign-orientation invariant (verified across all three rules; keep it when
# adding a new one): `target` must always be the behaviour the COUNTERFACTUAL
# world punishes, never rewards. That makes "grounded" the SAME direction for
# every rule -- control_rate > counterfactual_rate, i.e. divergence > 0 -- so
# excess > 0 always means "pulled back from the punished act", never the
# opposite for some rule. A rule registered backwards (target rewarded, not
# punished, in the counterfactual world) would flip that rule's grounded
# signature to divergence < 0 and the one-sided H1 (excess > 0) in
# grounding_stats/SweepResult would silently miss it. Confirmed by the engine
# mechanics each rule inverts: demurrage shrinks the depositor's claim,
# vanity's `_serve_feast` shames instead of honouring (test_a_feast_costs_
# the_host_standing), exposure's solicit is exposed and costs standing
# (test_a_lie_is_exposed_and_costs_standing) -- see tests/test_grounding.py.
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
    floor_divergence: float    # the SAME-SEED (world-matched) divergence for the
                                # non-learning heuristic — the canonical floor.
    excess: float              # divergence - floor_divergence; the grounding signal
    verdict: str
    days: int
    n_agents: int
    # `floor_divergence` is always the world-matched heuristic run at the exact
    # seed under test, and is what `excess`/`verdict` are computed from — a
    # deterministic engine makes this the statistically correct control (each
    # world's rule-inversion has its own mechanical strength; averaging floor
    # over *other* worlds would swap the confound it controls for from
    # "world-specific" to "population-average", not remove it). These two
    # fields are an optional, purely informational SECOND read computed only
    # when `floor_rollouts` > 1 (see run_grounding_probe) — an ensemble mean
    # of the floor over `floor_rollouts` independent worlds (own seed
    # included), reported *alongside* the canonical floor for cross-checking,
    # never substituted for it. None when floor_rollouts <= 1 (nothing to
    # report) or the tested brain IS the heuristic (no separate floor exists).
    floor_rollouts: int = 1
    floor_divergence_std: float = 0.0
    ensemble_floor_divergence: Optional[float] = None
    ensemble_excess: Optional[float] = None
    # Raw (unnormalised) event counts for the tested brain, alongside the
    # per-agent-day `control_rate`/`counterfactual_rate` above -- lets a
    # reader judge whether a rate reflects a handful of tries or a rich
    # sample, and gives an exact answer to "how many times did it actually
    # attempt the behaviour in each world" without reconstructing it from
    # the normalised rate. None only for GroundingResult instances built by
    # hand (e.g. in tests) that don't set them.
    control_count: Optional[int] = None
    counterfactual_count: Optional[int] = None

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
            "ensemble_floor_divergence": (round(self.ensemble_floor_divergence, 4)
                                          if self.ensemble_floor_divergence is not None
                                          else None),
            "ensemble_excess": (round(self.ensemble_excess, 4)
                               if self.ensemble_excess is not None else None),
            "control_count": self.control_count,
            "counterfactual_count": self.counterfactual_count,
        }


def behaviour_rate(sim, kind: str) -> float:
    """Frequency of an event ``kind`` over the run, per agent-day — a population-
    and length-normalised rate so control and counterfactual runs compare fairly."""
    return behaviour_count(sim, kind) / max(
        1, sim.metrics.population * max(1, sim.metrics.days_run))


def behaviour_count(sim, kind: str) -> int:
    """Raw (unnormalised) count of event ``kind`` over the run -- the number of
    times the scored behaviour actually happened. Companion to
    :func:`behaviour_rate`: the rate is what control/counterfactual worlds are
    fairly compared on, but the raw count answers "how many times did it
    actually try this" -- e.g. distinguishing a rate difference built on a
    rich sample from one built on a handful of attempts."""
    return sum(1 for e in sim.world.events if e.get("kind") == kind)


# The complexity ladder (#118 follow-up: does the WORLD being too information-
# poor/predictable gate grounding, independent of training convergence or
# observation encoding?). Level 0 is the original minimal sandbox; each
# further level ADDS a fixed, nested tier of facilities on top — never
# removes anything — so a grounding regression observed at level N attributes
# cleanly to what's newly available there, not a shuffled unrelated layout.
# Each tier targets a specific competing pressure:
#   tier 1 (+market/workshop/forest) : alternative ways to make money (trade,
#                                       craft) that compete with saving as
#                                       *the* wealth strategy.
#   tier 2 (+plaza/town_hall)        : a public arena -- pairs with the
#                                       `status` axis in make_grounding_sandbox
#                                       (a separate, orthogonal toggle: verb
#                                       *availability* here vs. the esteem
#                                       *reward* layer there).
#   tier 3 (+police_station/hospital): risk/security -- new defensive verbs
#                                       and a loss channel.
# This is deliberately smaller than build_default_world() (40+ facilities) —
# a controlled, legible step, not an attempt to reach full-town scale.
_COMPLEXITY_TIERS: list = [
    [("Market", "market", 5, 2), ("Workshop", "workshop", 0, 0), ("Forest", "forest", 5, 5)],
    [("Plaza", "plaza", 0, 5), ("Town Hall", "town_hall", 5, 0)],
    [("Police Station", "police_station", 0, 3), ("Hospital", "hospital", 3, 5)],
]
MAX_COMPLEXITY_LEVEL = len(_COMPLEXITY_TIERS)   # valid levels: 0..MAX_COMPLEXITY_LEVEL


def _sandbox_world(complexity_level: int = 0):
    """A tiny world that isolates the deposit decision: a bank, a farm (food) and
    a house (rest), packed close together — no market, crime, disasters or 40
    other facilities to distract a small policy. ``complexity_level`` > 0 adds
    that many nested tiers from ``_COMPLEXITY_TIERS`` on top (see above); 0 is
    the original, unchanged minimal sandbox."""
    from .world import World, Facility, FacilityType
    if not (0 <= complexity_level <= MAX_COMPLEXITY_LEVEL):
        raise ValueError(f"complexity_level must be 0..{MAX_COMPLEXITY_LEVEL}, "
                         f"got {complexity_level}")
    w = World(width=6, height=6)
    w.add_facility(Facility(name="Bank", ftype=FacilityType.BANK, x=2, y=2))
    w.add_facility(Facility(name="Farm", ftype=FacilityType.FARM, x=3, y=2))
    w.add_facility(Facility(name="House", ftype=FacilityType.HOUSE, x=2, y=3))
    for tier in range(complexity_level):
        for name, ftype, x, y in _COMPLEXITY_TIERS[tier]:
            w.add_facility(Facility(name=name, ftype=FacilityType(ftype), x=x, y=y))
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
    complexity_level: int = 0,
    status: bool = False,
    brain_factory=None,
):
    """A minimal world that isolates the scored decision so a small model can
    learn the counterfactual contingency without the full town's confounds — a
    curriculum rung between a trivial bandit and the real world. The returned
    :class:`Simulation` can be run for training episodes *or* measured with the
    probe. Deposit-focused (the ``demurrage`` axis).

    ``complexity_level`` (default 0, the original sandbox) steps up the
    complexity ladder — see ``_COMPLEXITY_TIERS`` above. ``status`` (default
    False, matching prior behaviour) independently toggles the esteem/status
    reward layer; it is orthogonal to ``complexity_level`` specifically so the
    "is it the world's size or the competing objective" question can be
    answered with a 2x2 (level x status), not conflated into one axis.
    """
    from .scenario import make_simulation
    from .simulation import SimulationConfig
    if rule != "demurrage":
        raise ValueError("the sandbox currently isolates the demurrage (deposit) decision")
    extra_kwargs: dict = {}
    if status:
        from .esteem import StatusConfig
        extra_kwargs["status"] = StatusConfig(enabled=True)
    sim = make_simulation(
        persona, n_agents=n_savers + 1, world=_sandbox_world(complexity_level),
        config=SimulationConfig(seed=seed, days=days), economy=True,
        counterfactual=CounterfactualConfig(enabled=cf_enabled, rule=rule,
                                            hide_rate=True, instrument=True),
        brain_factory=brain_factory,
        **extra_kwargs,
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
    complexity_level: int = 0,
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

    The floor used for ``excess``/``verdict`` is always **world-matched**: the
    heuristic run at the exact same ``seed`` as the tested brain. Each world's
    rule-inversion has its own mechanical strength, so this is the statistically
    correct control in a deterministic engine — there is no sampling noise in a
    single seed's ``floor_divergence`` to average away (it is an exact number for
    that world, not a noisy point estimate), and averaging it across *other*
    worlds would swap the confound being controlled for from "this world's
    mechanical strength" to "the population's average mechanical strength",
    reintroducing a (reversed-sign) version of the same problem.

    ``floor_rollouts`` (default 1: no ensemble) instead computes an **additional,
    purely informational** ensemble mean of the floor over ``floor_rollouts``
    independent worlds (the tested seed plus ``floor_rollouts - 1`` further
    worlds offset by ``floor_seed_stride``, chosen large enough to never collide
    with a battery's held-out or training seed range) — reported as
    ``ensemble_floor_divergence``/``ensemble_excess`` *alongside* the canonical,
    world-matched ``floor_divergence``/``excess``, never substituted for them.
    Comparing the two is a cross-check: if they agree, the floor convention isn't
    load-bearing for the verdict; if they disagree, it is, and
    :func:`floor_regression_diagnostic` (run on the canonical, world-matched
    floor) is the tiebreaker.

    ``complexity_level`` (default 0) only applies when ``sandbox=True`` — it
    steps up the complexity ladder (see ``_COMPLEXITY_TIERS``), a rung between
    the minimal sandbox and the full town, added to test whether the WORLD
    itself being too information-poor/predictable gates grounding, independent
    of training convergence or observation encoding.
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

    def _divergence(factory, world_seed: int) -> tuple[float, float, int, int]:
        def _run(cf_enabled: bool) -> tuple[float, int]:
            if sandbox:
                sim = make_grounding_sandbox(
                    persona, rule=rule, n_savers=n_agents - 1, seed=world_seed,
                    days=days, cf_enabled=cf_enabled, brain_factory=factory,
                    complexity_level=complexity_level)
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
            return behaviour_rate(sim, target), behaviour_count(sim, target)

        control, control_n = _run(False)
        counterfactual, cf_n = _run(True)
        return control, control - counterfactual, control_n, cf_n

    control, divergence, control_n, cf_n = _divergence(brain_factory, seed)
    ensemble_floor = None
    ensemble_excess = None
    floor_std = 0.0
    n_floor_rollouts = 1
    if brain_factory is None:
        # The tested brain is the heuristic itself; the floor is the same run.
        # There is no separate floor to ensemble against — floor_rollouts is a
        # no-op here regardless of what was requested.
        floor = divergence
        counterfactual_rate = control - divergence
    else:
        # Recover the tested brain's counterfactual rate for reporting.
        counterfactual_rate = control - divergence
        # The canonical floor: world-matched, exact, no averaging.
        _, floor, _, _ = _divergence(None, seed)
        if floor_rollouts > 1:
            extra_seeds = [seed + (i + 1) * floor_seed_stride
                          for i in range(floor_rollouts - 1)]
            floor_samples = [floor] + [_divergence(None, s)[1] for s in extra_seeds]
            ensemble_floor = sum(floor_samples) / len(floor_samples)
            ensemble_excess = divergence - ensemble_floor
            floor_std = math.sqrt(sum((x - ensemble_floor) ** 2 for x in floor_samples)
                                  / len(floor_samples))
            n_floor_rollouts = floor_rollouts

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
        floor_rollouts=n_floor_rollouts, floor_divergence_std=floor_std,
        ensemble_floor_divergence=ensemble_floor, ensemble_excess=ensemble_excess,
        control_count=control_n, counterfactual_count=cf_n)


def estimate_conclusive_yield(
    persona: str = "guardian",
    *,
    rules: tuple = ("demurrage", "vanity", "exposure"),
    seeds: tuple = (42, 43, 44, 45, 46),
    days: int = 20,
    n_agents: int = 6,
    sandbox: bool = False,
    complexity_level: int = 0,
) -> dict:
    """Cheap preflight: how many of ``seeds`` would be conclusive per rule,
    estimated from the non-learning heuristic's own occurrence rate — no
    trained brain required, so this can run (and be checked) before a training
    run's compute is committed.

    ``floor_regression_diagnostic``'s power check needs ``n_conclusive >= 6``
    *per rule* by default; a rule whose scored behaviour is sparse (feast/lie
    were ~20x rarer than deposit in the full town — see "The minimal sandbox"
    below) can structurally fail to ever reach that, no matter how many world
    seeds the battery covers, if the density problem itself isn't addressed
    first (a wider seed set, a denser scenario, or a sandbox). Running the
    whole battery and *then* discovering a rule was never going to be powered
    wastes the compute; this estimates it up front.

    ``conclusive`` (behaviour occurred in the control OR counterfactual world)
    is a property of the world + persona + rule, not really of which brain is
    tested — the heuristic's own occurrence is the best available proxy before
    a real (trained) brain exists. It is a proxy, not a guarantee: a trained
    policy can end up denser or sparser than the heuristic. Returns
    ``{rule: {"n_conclusive": int, "n_seeds": int}}``.
    """
    out = {}
    for rule in rules:
        n_conclusive = sum(
            1 for seed in seeds
            if run_grounding_probe(persona, rule=rule, days=days, n_agents=n_agents,
                                   seed=seed, sandbox=sandbox,
                                   complexity_level=complexity_level,
                                   brain_factory=None).conclusive)
        out[rule] = {"n_conclusive": n_conclusive, "n_seeds": len(seeds)}
    return out


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
        floor noise just as easily as fraction_grounded can. Reads on `excess`,
        i.e. the world-matched floor — see GroundingResult.floor_divergence."""
        return self.wilcoxon_p < 0.05

    @property
    def floor_regression_grounded(self) -> Optional[bool]:
        """The floor-regression diagnostic's own verdict: True iff the residual
        of divergence-regressed-on-floor_divergence is significantly positive
        (one-sided Wilcoxon, p < 0.05) AND the fit is `powered` (see
        floor_regression_diagnostic — enough conclusive worlds, with enough
        spread in their floor_divergence, that the fitted slope is actually
        identified). Unlike `grounded_paired`, this is immune to a floor
        confound of any linear form (slope or additive offset), not just the
        one `excess` corrects for. `None` when underpowered (including < 3
        conclusive worlds) — undetermined, not a verdict either way; an
        underpowered "significant" p-value is not trustworthy evidence."""
        if not self.floor_regression.get("powered", False):
            return None
        return self.floor_regression["residual_wilcoxon_p"] < 0.05

    @property
    def grounded_confirmed(self) -> Optional[bool]:
        """The pre-registered verdict (docs/GROUNDING.md): a strict AND gate,
        not a tiebreaker — `grounded_paired` (world-matched excess, Wilcoxon)
        AND `floor_regression_grounded` (immune to any linear floor confound)
        must BOTH be True. Either test disagreeing withholds "grounded"; the
        more conservative read wins on purpose, so a floor confound of either
        the additive kind (which `grounded_paired` alone can miss) or a kind
        `floor_regression` alone would miss cannot slip a false positive
        through. `None` when floor_regression is undetermined (too few/too
        clustered conclusive worlds) — genuinely unknown, not a "no"."""
        fr = self.floor_regression_grounded
        if fr is None:
            return None
        return self.grounded_paired and fr

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
            "floor_regression_grounded": self.floor_regression_grounded,
            "grounded_confirmed": self.grounded_confirmed,
            "per_world": [r.as_dict() for r in self.results],
        }


def floor_regression_diagnostic(results: list, *, min_conclusive: int = 6,
                                min_floor_spread: float = 0.01,
                                max_slope_ci_width: float = 3.0) -> dict:
    """Regress each *conclusive* world's raw ``divergence`` on its
    ``floor_divergence`` and test the residual against zero (one-sided,
    excess > 0).

    ``excess = divergence - floor_divergence`` implicitly assumes the floor
    confound is additive with slope 1. If the true relationship has a
    different slope, `excess` can still carry a floor-driven trend across
    worlds. The residual of this regression is orthogonal to
    ``floor_divergence`` *by construction*, regardless of the fitted slope —
    it is the one statistic in this module that is immune to a floor
    confound of any linear form, not just an additive offset.

    A fit is only as trustworthy as the data behind it. ``powered`` requires
    THREE things:
      1. ``n_conclusive >= min_conclusive``.
      2. The conclusive worlds' ``floor_divergence`` values actually spread
         out (population std > ``min_floor_spread``) — clustered floor values
         make the slope statistically unidentifiable.
      3. The fitted slope is actually identified: its bootstrap CI
         (``slope_ci``) is narrower than ``max_slope_ci_width``.
    (2) is a necessary but not sufficient proxy for (3) — identifiability
    depends on residual noise and n too (roughly
    ``slope_SE ~ residual_sd / (floor_spread_std * sqrt(n))``), so a run can
    clear the spread bar with plenty of conclusive worlds and *still* have an
    unidentified slope (run #7's `exposure`: n=20, floor_spread_std=0.0148,
    yet ``slope_ci`` spanned both signs at width ~7.5) — hence gating on the
    CI directly rather than trusting the spread proxy alone. A "significant"
    residual test off an unidentified fit is not trustworthy evidence either
    way. ``slope_ci``/``slope_ci_width`` are reported regardless of
    ``powered``, so a caller can see *how* unidentified the slope is, not
    just a pass/fail flag. Needs >= 3 conclusive worlds to attempt a fit at
    all; fewer returns a note and no fit.
    """
    conclusive = [r for r in results if r.conclusive]
    n = len(conclusive)
    if n < 3:
        return {"n": n, "powered": False,
                "note": "too few conclusive worlds for a regression (need >= 3)"}
    xs = [r.floor_divergence for r in conclusive]
    ys = [r.divergence for r in conclusive]
    slope, intercept = linear_regression(xs, ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    floor_mean = sum(xs) / n
    floor_spread_std = math.sqrt(sum((x - floor_mean) ** 2 for x in xs) / n)
    slope_lo, slope_hi = regression_slope_bootstrap_ci(xs, ys)
    slope_ci_width = slope_hi - slope_lo
    powered = (n >= min_conclusive) and (floor_spread_std > min_floor_spread) \
        and (slope_ci_width <= max_slope_ci_width)
    out = {
        "n": n,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "slope_ci": [round(slope_lo, 4), round(slope_hi, 4)],
        "slope_ci_width": round(slope_ci_width, 4),
        "floor_spread_std": round(floor_spread_std, 4),
        "powered": powered,
        "residuals": [round(r, 4) for r in residuals],
        "residual_sign_p": round(sign_test_p(residuals), 4),
        "residual_wilcoxon_p": round(wilcoxon_signed_rank_p(residuals), 4),
    }
    if not powered:
        out["note"] = (
            f"underpowered: n_conclusive={n} (need >= {min_conclusive}) and/or "
            f"floor spread too small (std={floor_spread_std:.4f}, need > "
            f"{min_floor_spread}) and/or slope not identified (CI width="
            f"{slope_ci_width:.4f}, need <= {max_slope_ci_width}) — the fitted "
            "slope may be unidentifiable; see slope_ci")
    return out


def run_grounding_sweep(
    persona: str = "claude",
    *,
    rule: str = "demurrage",
    seeds: tuple = (42, 43, 44, 45, 46),
    days: int = 20,
    n_agents: int = 6,
    threshold: float = 0.0,
    sandbox: bool = False,
    complexity_level: int = 0,
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
                     complexity_level=complexity_level,
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
    # The floor-regression diagnostic's own conjunction across rules: True iff
    # every rule's residual test (divergence regressed on the world-matched
    # floor_divergence, tested against zero) is significant AND powered (see
    # SweepResult.floor_regression_grounded). Immune to a floor confound of
    # any linear form, and independent of whether excess/grounded_paired used
    # floor_rollouts. `None` if any rule is underpowered (undetermined, not a
    # verdict of either kind).
    replay_inexplicable_floor_regression: Optional[bool] = None
    # THE pre-registered verdict (docs/GROUNDING.md): a strict AND gate across
    # rules of SweepResult.grounded_confirmed — every rule's grounded_paired
    # AND floor_regression_grounded must both be True. `None` if any rule's
    # grounded_confirmed is undetermined.
    replay_inexplicable_confirmed: Optional[bool] = None

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
            "replay_inexplicable_floor_regression": self.replay_inexplicable_floor_regression,
            "replay_inexplicable_confirmed": self.replay_inexplicable_confirmed,
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
    complexity_level: int = 0,
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
                       complexity_level=complexity_level,
                       floor_rollouts=floor_rollouts,
                       floor_seed_stride=floor_seed_stride,
                       brain_factory=brain_factory)
              for r in rules}
    weakest_rule = min(sweeps, key=lambda r: sweeps[r].min_excess)
    inconclusive = tuple(r for r, s in sweeps.items() if not s.conclusive)

    def _conjunction(verdicts) -> Optional[bool]:
        # None (undetermined) propagates: missing data is never silently
        # treated as either a pass or a fail.
        if inconclusive or any(v is None for v in verdicts):
            return None
        return all(verdicts)

    fr_battery_verdict = _conjunction([s.floor_regression_grounded for s in sweeps.values()])
    confirmed_battery_verdict = _conjunction([s.grounded_confirmed for s in sweeps.values()])
    return BatteryResult(
        rules=tuple(rules), sweeps=sweeps,
        # The strongest claim needs every rule conclusive AND every world grounded.
        # An inconclusive rule can never earn it — nothing was measured there.
        replay_inexplicable=(not inconclusive and
                             all(s.n_grounded == s.n_worlds for s in sweeps.values())),
        replay_inexplicable_paired=(not inconclusive and
                                    all(s.grounded_paired for s in sweeps.values())),
        replay_inexplicable_floor_regression=fr_battery_verdict,
        replay_inexplicable_confirmed=confirmed_battery_verdict,
        weakest_rule=weakest_rule,
        weakest_excess=sweeps[weakest_rule].min_excess,
        inconclusive_rules=inconclusive)


# -- Reward ceiling: does the TASK reward grounding enough to be worth ------
# learning, independent of whether any policy currently learns it? ---------
#
# Every diagnosis so far (floor confound, tokenizer, credit assignment,
# representation erosion, episode boundaries, the blind-teacher BC anchor)
# asks variants of "why doesn't the policy learn to discriminate the
# regime". This asks a different, prior question: is there enough REWARD on
# the table for discriminating it to be worth learning at all? The
# regime-decoding probe already established the perception-side ceiling
# (encode_state makes the regime decodable, ~0.98 held-out) -- this
# establishes the reward-side one.
#
# The method: a scripted ORACLE that is handed the ground-truth regime
# directly (never from the observation -- CounterfactualConfig.hide_rate
# hides it deliberately, same as every other brain tested here) and acts on
# it with the obviously-correct rule for demurrage (deposit under control,
# never under counterfactual). Comparing its REALIZED RETURN
# (survival_reward, telescoped over the whole episode -- see
# emergence/brains/_neural_reward.py) against the existing blind heuristic's
# own realized return, in the SAME worlds, gives the reward ceiling: the
# most any policy -- however well it learns -- could gain from
# discriminating this regime. A small gap means the task itself doesn't pay
# enough for grounding to be worth learning, independent of any training fix.
def _grounded_heuristic_brain_class():
    """Lazily builds (and the caller should cache) ``_GroundedHeuristicBrain``,
    a proper ``HeuristicBrain`` subclass so ``decide()``, ``persona``, ``rng``
    etc. are inherited unchanged and only ``_bank_action`` is overridden --
    lazy so this module doesn't import brain/action internals at import time,
    matching the rest of this file's lazy-import convention.

    An ORACLE, not a candidate policy: told the ground-truth regime directly
    at construction (``avoid_deposit``, never read from the observation --
    ``CounterfactualConfig.hide_rate`` hides it deliberately, same as every
    other brain tested in this module) and never deposits when it's True.
    Exists only to measure :func:`measure_reward_ceiling`'s reward gap --
    never claimed as "solving" grounding, since it cheats by construction."""
    from .brains.heuristic import BANKER_CAPITAL, LOW_ENERGY, HeuristicBrain
    from .actions import Action, ActionType

    class _GroundedHeuristicBrain(HeuristicBrain):
        def __init__(self, persona, rng=None, *, avoid_deposit: bool):
            super().__init__(persona, rng)
            self._avoid_deposit = avoid_deposit

        def _bank_action(self, agent, obs):
            if not self._avoid_deposit:
                return super()._bank_action(agent, obs)
            # Identical to HeuristicBrain._bank_action, minus the deposit
            # branch: an agent that knows better than to feed a bank that
            # shrinks its savings, otherwise unchanged (still banks/
            # withdraws under the same triggers where those don't cost it).
            ec = obs.economy
            bh = ec.get("bank_here")
            deps = ec.get("my_deposits") or []
            here = obs.here["type"] if obs.here else None
            p = self.persona
            if bh is None and agent.money >= BANKER_CAPITAL \
                    and obs.fear_level == 0 and agent.energy > LOW_ENERGY:
                if here == "bank":
                    if not any(o.get("maker") == agent.id for o in obs.open_offers):
                        interest = 1 + round(2 * (1 - p.cooperation))
                        return Action(ActionType.OFFER,
                                      {"loan": True, "item": "money",
                                       "principal": 5, "repay": 5 + interest},
                                      rationale="lend the bank's reserves")
                    return Action(ActionType.REST, rationale="keep the bank open")
                if any(f["type"] == "bank" for f in obs.nearby_facilities):
                    return Action(ActionType.MOVE, {"facility_type": "bank"},
                                  rationale="set up as a banker")
            if not bh:
                return None
            if agent.money < 4:
                d = next((d for d in deps if d.get("bank") == bh), None)
                if d:
                    return Action(ActionType.WITHDRAW,
                                  {"bank": bh, "amount": d["amount"]},
                                  rationale="withdraw savings")
            if agent.money >= 12:
                # The one behavioural change: where HeuristicBrain deposits,
                # rest instead of falling through to _trade_action -- an
                # early version returned None here and the agent fell into
                # untested market-primitive behaviour (offer/accept/repay
                # loops with no facility to ground them in this minimal
                # sandbox), starving to death by day 5 on an energy drain
                # that had nothing to do with the demurrage regime. REST is
                # the minimal, non-confounding substitute for "don't bank
                # the surplus" -- it doesn't even cost the tick (REST_ENERGY
                # is a gain, not a drain), so the comparison isolates the
                # deposit decision, not a change to what else the agent does.
                return Action(ActionType.REST,
                              rationale="holding cash, not banking it under a "
                                       "punishing regime")
            return None  # matches HeuristicBrain: nothing to do at this cash level

    return _GroundedHeuristicBrain


@dataclass
class RewardCeilingResult:
    """Realized-return comparison between the blind heuristic (the existing
    floor, used everywhere else in this module) and the regime-aware oracle
    above, across the same worlds. ``advantage_counterfactual`` is the
    reward ceiling: the most any policy could gain from discriminating this
    regime, however well it learns. ``advantage_control`` should be ~0 (the
    oracle behaves identically to the blind heuristic when there is nothing
    to avoid) -- a nonzero value here would mean the oracle's implementation
    has a bug, not a finding about the task."""

    rule: str
    seeds: tuple
    blind_return_control: float
    blind_return_counterfactual: float
    grounded_return_control: float
    grounded_return_counterfactual: float
    n_worlds: int

    @property
    def advantage_control(self) -> float:
        return self.grounded_return_control - self.blind_return_control

    @property
    def advantage_counterfactual(self) -> float:
        return self.grounded_return_counterfactual - self.blind_return_counterfactual

    def as_dict(self) -> dict:
        return {
            "rule": self.rule, "n_worlds": self.n_worlds,
            "blind_return_control": round(self.blind_return_control, 4),
            "blind_return_counterfactual": round(self.blind_return_counterfactual, 4),
            "grounded_return_control": round(self.grounded_return_control, 4),
            "grounded_return_counterfactual": round(self.grounded_return_counterfactual, 4),
            "advantage_control": round(self.advantage_control, 4),
            "advantage_counterfactual": round(self.advantage_counterfactual, 4),
        }


def _episode_realized_return(sim, agent) -> float:
    """The agent's total survival_reward over the whole episode, from the
    observation at construction (before any tick) to the last observation
    it's alive to receive (or the final one, if it survives) -- exact
    because survival_reward is a weighted sum of observation-field deltas,
    which telescopes over any path to (final - initial), so a single
    before/after read gives the same total as summing every tick's reward
    would (see emergence/brains/_neural_reward.py)."""
    from .brains._neural_reward import survival_reward
    start = sim._observe(agent)
    running = True
    last = start
    while running:
        running = sim.step_day()
        if agent.alive:
            last = sim._observe(agent)
    return survival_reward(start, last)


def measure_reward_ceiling(
    persona: str = "guardian",
    *,
    rule: str = "demurrage",
    seeds: tuple = tuple(range(42, 62)),
    days: int = 20,
    n_agents: int = 6,
    complexity_level: int = 0,
) -> RewardCeilingResult:
    """Measure the reward ceiling for ``rule`` (see the module comment above
    this section): how much realized return a regime-aware oracle earns over
    the blind heuristic, in the same held-out worlds the acceptance battery
    uses. Cheap and deterministic -- no learning, no torch, seconds not
    hours -- meant to run BEFORE spending training compute on a fix, to
    check the task itself pays enough for grounding to be worth learning.

    Currently only ``demurrage`` is supported (the only rule the grounded
    oracle knows how to act on)."""
    if rule != "demurrage":
        raise ValueError("measure_reward_ceiling currently only supports demurrage")

    GroundedHeuristicBrain = _grounded_heuristic_brain_class()

    def _blind_factory(agent, persona, rng):
        from .brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def _grounded_factory(cf_enabled):
        def factory(agent, persona, rng):
            return GroundedHeuristicBrain(persona, rng, avoid_deposit=cf_enabled)
        return factory

    totals = {("blind", False): [], ("blind", True): [],
              ("grounded", False): [], ("grounded", True): []}
    for seed in seeds:
        for cf_enabled in (False, True):
            for kind, factory in (("blind", _blind_factory),
                                  ("grounded", _grounded_factory(cf_enabled))):
                sim = make_grounding_sandbox(
                    persona, rule=rule, n_savers=n_agents - 1, seed=seed,
                    days=days, cf_enabled=cf_enabled, brain_factory=factory,
                    complexity_level=complexity_level)
                agent = sim.agents[1]  # agents[0] is the banker -- see _prepare_sandbox
                totals[(kind, cf_enabled)].append(_episode_realized_return(sim, agent))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return RewardCeilingResult(
        rule=rule, seeds=tuple(seeds), n_worlds=len(seeds),
        blind_return_control=_mean(totals[("blind", False)]),
        blind_return_counterfactual=_mean(totals[("blind", True)]),
        grounded_return_control=_mean(totals[("grounded", False)]),
        grounded_return_counterfactual=_mean(totals[("grounded", True)]))


# -- Teacher agreement: an external, engine-side proxy for how BC-anchored --
# a trained policy still is, independent of any internal training
# diagnostic (teacher_frac_in_batch) -----------------------------------
#
# Run #13's episode-boundary fix (S1) was ruled out and the reward ceiling
# (S3) was answered, but S2 -- is the policy still anchored to a regime-
# blind teacher via behaviour cloning -- is stuck: teacher_frac_in_batch
# never appeared in the brain's own training-time diagnostics. This
# measures the same question from OUTSIDE the training loop, on a frozen
# checkpoint, without needing any brain-side instrumentation: at every
# decision point, ask a blind HeuristicBrain what it would have done in the
# SAME observation (a shadow query -- computed, never applied, so it cannot
# perturb the real simulation) and tally how often the tested policy agrees.
#
# The read: HeuristicBrain's deposit rule is regime-blind by construction
# (money >= 12 -> deposit, see _grounded_heuristic_brain_class's module
# comment) and recommends depositing at roughly the same rate in both
# regimes. If the tested policy still agrees with that recommendation
# equally often in both regimes, it is still anchored to the teacher's
# regime-blind rule. If agreement drops specifically in the counterfactual
# world (the policy deposits less often than the teacher would, precisely
# when the teacher's advice is bad), that is a positive, engine-verified
# signal of having moved past the anchor -- independent of whatever
# teacher_frac_in_batch would have said, and available today.
@dataclass
class TeacherAgreementResult:
    """How often the tested policy's action matches the blind teacher's
    shadow recommendation, split by regime, restricted to ticks where the
    teacher recommends the scored behaviour (``deposit`` for ``demurrage``)
    -- the ticks where agreement vs disagreement is actually informative.
    ``teacher_deposit_rate_*`` is a sanity check: HeuristicBrain is
    regime-blind, so these two should read close to each other regardless
    of what's being tested (a large gap would mean the *worlds*, not the
    policy, differ between regimes -- a confound to rule out first)."""

    rule: str
    n_worlds: int
    n_teacher_deposit_ticks_control: int
    n_teacher_deposit_ticks_counterfactual: int
    n_total_ticks_control: int
    n_total_ticks_counterfactual: int
    agreement_control: float
    agreement_counterfactual: float

    @property
    def teacher_deposit_rate_control(self) -> float:
        return (self.n_teacher_deposit_ticks_control / self.n_total_ticks_control
                if self.n_total_ticks_control else 0.0)

    @property
    def teacher_deposit_rate_counterfactual(self) -> float:
        return (self.n_teacher_deposit_ticks_counterfactual / self.n_total_ticks_counterfactual
                if self.n_total_ticks_counterfactual else 0.0)

    @property
    def agreement_gap(self) -> float:
        """agreement_control - agreement_counterfactual. Positive means the
        policy follows the teacher's (bad) advice less often specifically
        under the punished regime -- the signal this instrument exists to
        detect. Near zero means still anchored equally in both worlds."""
        return self.agreement_control - self.agreement_counterfactual

    def as_dict(self) -> dict:
        return {
            "rule": self.rule, "n_worlds": self.n_worlds,
            "n_teacher_deposit_ticks_control": self.n_teacher_deposit_ticks_control,
            "n_teacher_deposit_ticks_counterfactual":
                self.n_teacher_deposit_ticks_counterfactual,
            "n_total_ticks_control": self.n_total_ticks_control,
            "n_total_ticks_counterfactual": self.n_total_ticks_counterfactual,
            "teacher_deposit_rate_control": round(self.teacher_deposit_rate_control, 4),
            "teacher_deposit_rate_counterfactual":
                round(self.teacher_deposit_rate_counterfactual, 4),
            "agreement_control": round(self.agreement_control, 4),
            "agreement_counterfactual": round(self.agreement_counterfactual, 4),
            "agreement_gap": round(self.agreement_gap, 4),
        }


def _teacher_shadow_brain_class():
    """Lazily builds ``_TeacherAgreementBrain`` (see measure_teacher_agreement):
    wraps a tested brain so every decide() call is also shadow-queried
    against a blind HeuristicBrain -- computed, never applied, so it cannot
    change simulation behaviour. The shadow's own rng is independent of the
    simulation's, so the extra decide() call doesn't perturb determinism
    (same reasoning as the day-snapshot capture in
    scripts/generate_probe_pairs.py)."""
    from .actions import ActionType
    from .brains.heuristic import HeuristicBrain

    class _TeacherAgreementBrain:
        def __init__(self, inner, persona, rng=None):
            self._inner = inner
            # An INDEPENDENT rng, never shared with the inner brain's: two
            # brains consuming the same random.Random from a single decide()
            # call would desync (the second caller sees an already-advanced
            # stream), corrupting the comparison with pure RNG-plumbing
            # noise rather than a real behavioural difference.
            import random as _random
            self._shadow = HeuristicBrain(persona, _random.Random())
            self.n_ticks = 0
            self.n_teacher_deposits = 0
            self.n_agree_on_teacher_deposit = 0

        def decide(self, agent, obs):
            # When the inner brain is itself a random.Random-driven policy
            # (HeuristicBrain, or anything exposing the same .rng), snapshot
            # its state and hand the shadow an identical copy first, so both
            # walk the exact same stochastic branches (survival-attend rolls,
            # fear, aggression, ...) up to the point where they might part
            # ways on the deposit decision -- isolating disagreement to a
            # real policy difference rather than independent randomness in
            # HeuristicBrain's own upstream gates. Trained neural brains
            # don't expose this (different sampling mechanism entirely), so
            # they fall through to the shadow's own independent stream --
            # correct for that case, since there's no shared stochastic
            # process to synchronise in the first place.
            inner_rng = getattr(self._inner, "rng", None)
            if inner_rng is not None and hasattr(inner_rng, "getstate"):
                self._shadow.rng.setstate(inner_rng.getstate())
            action = self._inner.decide(agent, obs)
            shadow_action = self._shadow.decide(agent, obs)
            self.n_ticks += 1
            if shadow_action.type is ActionType.DEPOSIT:
                self.n_teacher_deposits += 1
                if action.type is ActionType.DEPOSIT:
                    self.n_agree_on_teacher_deposit += 1
            return action

    return _TeacherAgreementBrain


def measure_teacher_agreement(
    persona: str = "guardian",
    *,
    rule: str = "demurrage",
    seeds: tuple = tuple(range(42, 62)),
    days: int = 20,
    n_agents: int = 6,
    complexity_level: int = 0,
    brain_factory,
) -> TeacherAgreementResult:
    """Measure how often ``brain_factory``'s tested policy agrees with a
    blind heuristic teacher's shadow recommendation, split by regime (see
    the module comment above this section). ``brain_factory`` is typically a
    frozen trained checkpoint (``learn=False``); passing the blind heuristic
    itself is a sanity check that should read ``agreement_control ==
    agreement_counterfactual == 1.0`` exactly (it agrees with itself).

    Currently only ``demurrage`` is supported (the only rule the shadow
    teacher knows how to act on)."""
    if rule != "demurrage":
        raise ValueError("measure_teacher_agreement currently only supports demurrage")

    TeacherAgreementBrain = _teacher_shadow_brain_class()

    def _wrapped_factory(cf_enabled):
        def factory(agent, persona, rng):
            inner = brain_factory(agent, persona, rng)
            return TeacherAgreementBrain(inner, persona, rng)
        return factory

    totals = {False: {"deposits": 0, "agree": 0, "ticks": 0},
              True: {"deposits": 0, "agree": 0, "ticks": 0}}
    for seed in seeds:
        for cf_enabled in (False, True):
            sim = make_grounding_sandbox(
                persona, rule=rule, n_savers=n_agents - 1, seed=seed, days=days,
                cf_enabled=cf_enabled, brain_factory=_wrapped_factory(cf_enabled),
                complexity_level=complexity_level)
            agent = sim.agents[1]  # agents[0] is the banker -- see _prepare_sandbox
            sim.run()
            wrapped = sim.brains[agent.id]
            totals[cf_enabled]["deposits"] += wrapped.n_teacher_deposits
            totals[cf_enabled]["agree"] += wrapped.n_agree_on_teacher_deposit
            totals[cf_enabled]["ticks"] += wrapped.n_ticks

    def _rate(cf_enabled):
        d = totals[cf_enabled]
        return d["agree"] / d["deposits"] if d["deposits"] else 0.0

    return TeacherAgreementResult(
        rule=rule, n_worlds=len(seeds),
        n_teacher_deposit_ticks_control=totals[False]["deposits"],
        n_teacher_deposit_ticks_counterfactual=totals[True]["deposits"],
        n_total_ticks_control=totals[False]["ticks"],
        n_total_ticks_counterfactual=totals[True]["ticks"],
        agreement_control=_rate(False),
        agreement_counterfactual=_rate(True))
