#!/usr/bin/env python3
"""Train a developmental brain IN the real engine, then run the acceptance battery.

This is the #130 milestone runner, following the brain side's confirmed procedure
(their mirror checkpoints are shape-incompatible with the real build_brain, and
their environment cannot reach this engine — so real-config training happens
here, where the engine lives):

  1. Run --neural towns for N days: NeuralDevelopmentalBrain learns inside
     decide(), so the run itself IS the training. Episodes rotate through a
     control world and each counterfactual world (demurrage / vanity / exposure)
     so the policy experiences both sides of every rule. A heuristic teacher
     provides the L0-L1 scaffolding. Episodes domain-randomize across a POOL of
     world seeds (--pool-size, rotating), disjoint from the battery's held-out
     seeds — see the note below.
  2. Periodically save a full checkpoint (dev.save_checkpoint — policy included)
     and probe it with GroundingMonitor (frozen: learn=False) *on the first pool
     seed* (a fixed held-in world — a training-health check, not a generalisation
     claim); stop early once every rule's monitor reports is_stable(window).
  3. Run run_grounding_battery on BATTERY_SEEDS (held-out) with the checkpointed
     brain and write battery.json — paste battery.as_dict() to issue #130. A
     negative result is a real, reportable result.

Train/eval separation (do not weaken this): run #5 trained on seeds
42,43,44,45,46,... (--seed + episode index) while the battery's default held-out
seeds are exactly (42,43,44,45,46) — the first 5 training episodes were literally
the battery's eval worlds, and GroundingMonitor's is_stable check used seed=42,
also a battery seed. That contamination is why fraction_grounded stayed flat
(0.4->0.4) even as is_stable was reached for the first time: the "improvement"
was convergence on training worlds, not evidence of generalisation. Fixed by
domain-randomizing training over a seed pool that is asserted disjoint from
BATTERY_SEEDS, and pointing the health-check monitor at a pool seed instead of a
battery seed. is_stable/GroundingMonitor's design does NOT change (still a fixed
held-in check) — only which seed it points at.

Floor confound (found inspecting run #6, after the leak fix above): even with
train/eval seeds properly disjoint, fraction_grounded stayed flat at 0.4 across
runs, and the SAME worlds (seeds 44, 45) were the ones that read "grounded" every
time — tracking floor_divergence, not the tested brain. `excess = divergence -
floor_divergence` only removes an *additive* floor confound; a *slope* confound
survives the subtraction untouched.

`floor_divergence`/`excess` remain the WORLD-MATCHED heuristic floor (same seed
as the tested brain) — this engine is deterministic, so a single seed's floor is
an exact number for that world, not a noisy estimate with sampling error to
average away. An earlier version of this response tried "--floor-rollouts
averages the floor to reduce its noise", which was wrong on both counts: there
is no such noise to reduce, and averaging the floor over *other* worlds swaps
the confound being controlled for from "this world's mechanical strength" to
"the population's average" — a reversed-sign version of the same problem
(caught before it shipped in a run). `--floor-rollouts` now computes that
ensemble ONLY as an additional, side-by-side cross-check
(`ensemble_floor_divergence`/`ensemble_excess` on GroundingResult) — it never
moves the canonical, world-matched fields, or what any verdict is based on.

Two responses that DO change what's reported (not what run_grounding_battery's
core excess means):
  1. The battery reports paired statistics (sign test, Wilcoxon signed-rank, a
     bootstrap CI) over the conclusive worlds' world-matched excess —
     SweepResult.sign_test_p/wilcoxon_p/bootstrap_ci_mean_excess — a harder-to-
     Goodhart read than the hard-threshold fraction_grounded, not a replacement
     for it. SweepResult.grounded_paired = wilcoxon_p < 0.05.
  2. SweepResult.floor_regression regresses each world's raw divergence on its
     (world-matched) floor_divergence and tests the residual against zero —
     immune to a floor confound of ANY linear form (slope or offset), not just
     the additive one `excess` assumes, and independent of the floor convention
     debate above. Only trustworthy when `powered`: >= 6 conclusive worlds,
     enough spread in their floor_divergence, AND the fitted slope's bootstrap
     CI is actually narrow (floor_regression_diagnostic's min_conclusive/
     min_floor_spread/max_slope_ci_width). The spread check alone is
     necessary but not sufficient -- run #7's `exposure` cleared n=20 and
     floor_spread_std=0.0148 comfortably, yet its slope_ci spanned both signs
     (width ~7.5); gating on CI width directly caught what the spread proxy
     alone would have missed. An underpowered fit's p-value is not evidence
     either way, so SweepResult.floor_regression_grounded is None rather than
     guessing.

THE pre-registered verdict (fixed before the next expanded run, per
docs/GROUNDING.md) is SweepResult.grounded_confirmed / BatteryResult.
replay_inexplicable_confirmed — a STRICT AND GATE: grounded_paired AND
floor_regression_grounded must BOTH be True, per rule. Either disagreeing
withholds "grounded"; this is not a tiebreaker where floor_regression
overrides grounded_paired (an earlier draft's wording implied that, which
contradicted the AND actually coded — floor_regression alone missing the
signal grounded_paired found is just as disqualifying as the reverse). None
if floor_regression is underpowered for any rule — genuinely undetermined,
not a "no". fraction_grounded/replay_inexplicable remain reported as
required context, never an alternate path to a pass.

BATTERY_SEEDS was also widened (5 -> 20 held-out worlds) for the statistical
power (1)-(2) need; by itself this does not de-bias anything.

What `None` means, fixed BEFORE looking at any run's numbers (the burden of
proof is on grounding, not on replay: until a powered confirmatory test says
"grounded", the reportable status is "not grounded / unconfirmed" -- None is
not a pass, and it must not become one after the fact). Two different Nones,
two different next steps:
  - UNDETERMINED (floor_regression unpowered for a rule): not a verdict either
    way -- the battery didn't measure enough to know. Next step: improve the
    battery itself (more conclusive worlds, or address behaviour density) and
    re-run, not conclude anything about the brain.
  - POWERED-NO (floor_regression was powered but grounded_confirmed is False
    because grounded_paired and/or floor_regression_grounded came back
    negative): a real negative result. Next step: stop tuning the metric and
    ask whether the rule is learnable from the observation as given
    (representation learnability), per docs/GROUNDING.md.
Which one occurred is visible directly off SweepResult.floor_regression["powered"]
per rule -- report it, don't infer it from grounded_confirmed alone.

Practical risk this creates: floor_regression's power check is PER RULE, and
feast/lie are far sparser than deposit (~20x, see "the minimal sandbox" in
docs/GROUNDING.md) -- a rule can be structurally unable to ever reach
n_conclusive>=6 no matter how many seeds the battery covers, if the density
problem isn't addressed first. Running the whole (expensive) battery and only
THEN discovering a rule was undetermined from the start wastes the run. Run
`--preflight-only` first (cheap: heuristic-only, no trained brain, no torch
even) to see the expected per-rule conclusive yield against BATTERY_SEEDS
before committing training compute.

Requires the [neural] extra (torch) plus the private llm_model_agi package; the
shared CI workflow (.github/workflows/neural-train-battery.yml) installs both.
Everything engine-side is stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.brains.heuristic import HeuristicBrain          # noqa: E402
from emergence.brains.neural import NeuralDevelopmentalBrain   # noqa: E402
from emergence.esteem import StatusConfig                      # noqa: E402
from emergence.grounding import (                              # noqa: E402
    CounterfactualConfig,
    MAX_COMPLEXITY_LEVEL,
    estimate_conclusive_yield,
    make_grounding_sandbox,
    run_grounding_battery,
)
from emergence.grounding_monitor import GroundingMonitor       # noqa: E402
from emergence.scenario import make_simulation                 # noqa: E402
from emergence.simulation import SimulationConfig              # noqa: E402

RULES = ("demurrage", "vanity", "exposure")
# Every rule is experienced from both sides: control, then each inverted world.
EPISODE_ROTATION = (None, "demurrage", None, "vanity", None, "exposure")

# The battery's held-out worlds. Declared here, not just inherited from
# run_grounding_battery's own (smaller) default, so the training pool can be
# asserted disjoint from it — the exact leak that flattened run #5's
# fraction_grounded — and so the acceptance battery gets the statistical power
# a paired test needs. Widened from the original 5 (42-46) to 20 for that
# reason; the default --seed (1000) and --pool-size (12) stay far clear of it.
BATTERY_SEEDS = tuple(range(42, 62))

# Mirrors floor_regression_diagnostic's own default (emergence/grounding.py) —
# kept as an explicit constant here so the preflight warning below and the
# regression's actual power check can never silently drift apart.
MIN_CONCLUSIVE_FOR_POWER = 6


def print_preflight(persona: str, rules: tuple, *, days: int, n_agents: int,
                    sandbox: bool, complexity_level: int = 0) -> bool:
    """Estimate each rule's conclusive yield against BATTERY_SEEDS using the
    heuristic only (no trained brain, no torch) and print it. Returns True iff
    every rule looks likely to power floor_regression. Purely advisory — it
    does not abort the run, since the heuristic's occurrence rate is a proxy
    for the trained brain's, not a guarantee — but printed loudly and early so
    a human can decide whether to fix behaviour density (more seeds, a denser
    scenario, or the sandbox) before committing training compute to a run that
    would come back UNDETERMINED for a sparse rule regardless of what the
    brain does (see the module docstring's "What None means")."""
    yields = estimate_conclusive_yield(persona, rules=rules, seeds=BATTERY_SEEDS,
                                       days=days, n_agents=n_agents, sandbox=sandbox,
                                       complexity_level=complexity_level)
    print(f"[preflight] estimated conclusive yield vs {len(BATTERY_SEEDS)} held-out "
          f"worlds (heuristic proxy, need >= {MIN_CONCLUSIVE_FOR_POWER} per rule for "
          "floor_regression to be powered):")
    all_ok = True
    for rule, y in yields.items():
        ok = y["n_conclusive"] >= MIN_CONCLUSIVE_FOR_POWER
        all_ok = all_ok and ok
        flag = "ok" if ok else "AT RISK -- likely UNDETERMINED, not a verdict"
        print(f"  {rule:>10}: {y['n_conclusive']}/{y['n_seeds']} conclusive  [{flag}]")
    if not all_ok:
        print("[preflight] WARNING: at least one rule is unlikely to reach "
              f"n_conclusive >= {MIN_CONCLUSIVE_FOR_POWER} -- floor_regression_grounded "
              "(and therefore grounded_confirmed) would likely be None for it "
              "regardless of what the trained brain does. Consider widening "
              "BATTERY_SEEDS, using --sandbox, or otherwise increasing the "
              "behaviour's density before spending training compute.", flush=True)
    return all_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train in the real engine, then run the battery (#130)")
    ap.add_argument("--episodes", type=int, default=60, help="max training episodes")
    ap.add_argument("--days", type=int, default=20, help="days per training episode")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1000,
                    help="first seed of the TRAINING pool (domain-randomized across "
                         "--pool-size consecutive seeds, rotated per episode). Must "
                         f"not overlap the battery's held-out seeds {BATTERY_SEEDS} "
                         "— asserted at startup.")
    ap.add_argument("--pool-size", type=int, default=12,
                    help="number of distinct world seeds to domain-randomize "
                         "training over; the brain side's diagnosis (run #5: "
                         "is_stable reached but fraction_grounded stayed flat) "
                         "was that a single training world can converge without "
                         "generalizing — a real fix, not a hparam tweak.")
    ap.add_argument("--persona", default="guardian",
                    help="floor persona; guardian exercises all three scored behaviours")
    ap.add_argument("--probe-every", type=int, default=5, help="probe cadence (episodes)")
    ap.add_argument("--window", type=int, default=3, help="is_stable window (probes)")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--floor-rollouts", type=int, default=1,
                    help="report an ADDITIONAL ensemble-mean floor read "
                         "(over this many independent worlds) alongside the "
                         "canonical world-matched floor, purely as a cross-check "
                         "-- never changes excess/verdict/fraction_grounded. "
                         "See the floor-confound note in this module's docstring.")
    ap.add_argument("--sandbox", action="store_true",
                    help="train + measure in the minimal sandbox (dense behaviour, "
                         "conclusive). demurrage only — the sandbox's supported rule.")
    ap.add_argument("--complexity-level", type=int, default=0,
                    help="step up the sandbox's complexity ladder (0..%d; only applies "
                         "with --sandbox). 0 is the original minimal sandbox; each "
                         "further level ADDS a nested tier of facilities (market/"
                         "workshop/forest, then plaza/town_hall, then police/hospital) "
                         "-- never removes anything. Added to test whether the WORLD "
                         "itself being too information-poor/predictable gates "
                         "grounding, independent of training convergence or "
                         "observation encoding (see docs/GROUNDING.md)." % MAX_COMPLEXITY_LEVEL)
    ap.add_argument("--status", action=argparse.BooleanOptionalAction, default=None,
                    help="enable the status/esteem layer during TRAINING episodes (a "
                         "competing reward objective). Default: on for full town "
                         "(existing behaviour), off for --sandbox (existing "
                         "behaviour) -- pass --status/--no-status to override either "
                         "way and build the sandbox x status 2x2 factorial that "
                         "isolates world size from this competing-objective axis, "
                         "independent of --complexity-level.")
    ap.add_argument("--regime-block-size", type=int, default=1,
                    help="only applies with --sandbox: hold the control/counterfactual "
                         "regime fixed for this many CONSECUTIVE episodes before "
                         "switching, instead of alternating every single episode "
                         "(default 1 = prior behaviour, alternate every episode). "
                         "The brain side confirmed the deployed policy is memoryless "
                         "step-to-step (no recurrent state persists even within an "
                         "episode) -- switching every episode may simply dilute the "
                         "per-regime gradient signal. If a larger block size measurably "
                         "improves grounding, that's the brain side's own pre-registered "
                         "'weak H3' outcome (regime info reaches the observation but "
                         "switching frequency was diluting it); no change, 'strong H3' "
                         "(their tokenizer isn't surfacing it) -- though engine-side "
                         "inspection already shows the regime IS distinguishable "
                         "snapshot-to-snapshot (economy.my_deposits[].amount trend, "
                         "self_view.money trend, memory text -- see docs/GROUNDING.md), "
                         "so 'no change' would point at their tokenizer specifically.")
    ap.add_argument("--out", default="grounding_out", help="output dir (ckpt, logs, battery.json)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="print the estimated per-rule conclusive yield against "
                         "BATTERY_SEEDS (heuristic only, no torch/training) and exit "
                         "-- run this before a real training run to catch a rule "
                         "that's structurally too sparse to ever power floor_regression.")
    ap.add_argument("--hparams", default=None,
                    help='JSON dict forwarded to build_brain, e.g. '
                         '\'{"batch_every": 64, "lr_decay_steps": 4000}\' — the '
                         "brain side's late-training-oscillation damper knobs "
                         "(batch_every/lr/lr_min/lr_decay_steps/entropy_weight/"
                         "self_attempt_base/bc_weight). Unset means their defaults.")
    args = ap.parse_args(argv)

    hparams = json.loads(args.hparams) if args.hparams else None

    train_pool = [args.seed + i for i in range(args.pool_size)]
    overlap = set(train_pool) & set(BATTERY_SEEDS)
    if overlap:
        sys.exit(f"[fatal] training pool {train_pool} overlaps the battery's "
                 f"held-out seeds {BATTERY_SEEDS} at {sorted(overlap)} — this is "
                 "exactly the train/eval leak that flattened run #5's "
                 "fraction_grounded. Pick a --seed that keeps the pool disjoint "
                 "(default 1000 already is).")

    # The sandbox isolates one decision; it currently supports demurrage only.
    rules = ("demurrage",) if args.sandbox else RULES

    if args.complexity_level != 0 and not args.sandbox:
        sys.exit("[fatal] --complexity-level only applies with --sandbox.")
    if args.regime_block_size != 1 and not args.sandbox:
        sys.exit("[fatal] --regime-block-size only applies with --sandbox.")
    if args.regime_block_size < 1:
        sys.exit("[fatal] --regime-block-size must be >= 1.")
    # Default matches prior behaviour exactly (full town always had status on,
    # --sandbox always had it off); --status/--no-status overrides either way
    # to build the sandbox x status 2x2 factorial (see --status's help).
    status_enabled = args.status if args.status is not None else (not args.sandbox)

    print_preflight(args.persona, rules, days=args.days, n_agents=args.agents,
                    sandbox=args.sandbox, complexity_level=args.complexity_level)
    if args.preflight_only:
        return 0

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "agent.pt")

    # -- persistent training brains: one per agent id, reused across episodes --
    brains: dict[str, NeuralDevelopmentalBrain] = {}

    def training_factory(agent, persona, rng):
        b = brains.get(agent.id)
        if b is None:
            b = NeuralDevelopmentalBrain(persona, teacher=HeuristicBrain(persona),
                                         hparams=hparams)
            brains[agent.id] = b
        else:
            b._prev_obs = None            # a new episode is a fresh trajectory
        return b

    # -- frozen evaluation factory: load the checkpoint, never learn ----------
    def probe_factory(agent, persona, rng):
        return NeuralDevelopmentalBrain(persona, learn=False, checkpoint=ckpt,
                                        hparams=hparams)

    # is_stable is a training-health check, not a generalisation claim: it stays
    # on a single fixed HELD-IN world (the pool's first seed) by design — this is
    # unchanged from before. What changed is that this seed is no longer also a
    # battery seed (see BATTERY_SEEDS / the disjointness assertion above).
    monitors = {
        r: GroundingMonitor(args.persona, rule=r, days=args.days,
                            n_agents=args.agents, seed=train_pool[0],
                            threshold=args.threshold, sandbox=args.sandbox,
                            complexity_level=args.complexity_level)
        for r in rules
    }

    def build_episode(ep: int):
        """One training episode. Domain-randomizes the world seed by rotating
        through train_pool (disjoint from BATTERY_SEEDS — see the module
        docstring): a policy that only ever sees one world can converge on that
        world's incidental layout rather than the rule, which is exactly what
        run #5 showed (is_stable reached, fraction_grounded unmoved). In sandbox
        mode, hold the control/counterfactual regime fixed for
        --regime-block-size consecutive episodes before switching (default 1 =
        alternate every episode); in full-town mode, rotate control + each of
        the three inverted worlds every episode (unaffected by the block size)."""
        seed = train_pool[ep % len(train_pool)]
        if args.sandbox:
            cf_enabled = (ep // args.regime_block_size) % 2 == 1
            return make_grounding_sandbox(
                args.persona, rule="demurrage", n_savers=args.agents - 1,
                seed=seed, days=args.days,
                cf_enabled=cf_enabled, brain_factory=training_factory,
                complexity_level=args.complexity_level, status=status_enabled)
        rule = EPISODE_ROTATION[ep % len(EPISODE_ROTATION)]
        return make_simulation(
            args.persona, n_agents=args.agents, economy=True,
            status=StatusConfig(enabled=status_enabled),
            config=SimulationConfig(seed=seed, days=args.days),
            counterfactual=CounterfactualConfig(
                enabled=rule is not None, rule=rule or "demurrage",
                hide_rate=True, instrument=True),
            brain_factory=training_factory,
        )

    where = "sandbox" if args.sandbox else "full town"
    level_str = (f", complexity_level={args.complexity_level}, "
                f"regime_block_size={args.regime_block_size}") if args.sandbox else ""
    print(f"[train] up to {args.episodes} episodes x {args.days} days, "
          f"{args.agents} agents, persona={args.persona}, {where}{level_str}, "
          f"status={status_enabled}, rules={','.join(rules)}, hparams={hparams}, "
          f"train_pool={train_pool} (battery held-out: {list(BATTERY_SEEDS)})",
          flush=True)
    stable = False
    for ep in range(args.episodes):
        sim = build_episode(ep)
        sim.run()

        first = brains[next(iter(brains))]
        if first._broken or first._dev is None:
            sys.exit("[fatal] the neural backend is not live (fell back to the "
                     "heuristic). Install torch + llm_model_agi and retry — a "
                     "heuristic-only run would train nothing.")

        # Surface the brain side's optional per-step diagnostics (grad_steps, lr,
        # ...) if learn() returns them — the calibration signal for hparams like
        # lr_decay_steps. None if their build doesn't report it.
        info = getattr(first, "last_learn_info", None)
        info_str = f" | {info}" if info else ""
        print(f"[train] episode {ep + 1}/{args.episodes} done{info_str}", flush=True)

        if (ep + 1) % args.probe_every != 0:
            continue

        # Save the full agent (policy included) and probe the frozen checkpoint.
        first._dev.save_checkpoint(ckpt)
        line = []
        for r, mon in monitors.items():
            res = mon.probe(ep, probe_factory)
            line.append(f"{r}={res.excess:+.4f}(streak {mon.streak_above_threshold()})")
        print(f"[probe] ep {ep + 1}: excess " + "  ".join(line), flush=True)
        if all(m.is_stable(args.window) for m in monitors.values()):
            stable = True
            print(f"[train] all rules stable over the last {args.window} probes — "
                  "stopping early.", flush=True)
            break

    # -- final checkpoint + the acceptance battery ----------------------------
    first = brains[next(iter(brains))]
    first._dev.save_checkpoint(ckpt)
    for r, mon in monitors.items():
        mon.to_jsonl(os.path.join(args.out, f"grounding_{r}.jsonl"))

    print(f"[battery] running the acceptance battery ({','.join(rules)} x "
          f"held-out worlds {list(BATTERY_SEEDS)}, {where}{level_str}, "
          f"floor_rollouts={args.floor_rollouts})...", flush=True)
    # Explicit, not the function's own default: this script's train/eval
    # separation is asserted against BATTERY_SEEDS specifically, so evaluation
    # must use exactly that set even if the library default ever changes.
    # complexity_level is a train/eval-MATCHED design (measure at the same
    # level trained at) -- the ladder asks "can grounding happen at all at
    # this complexity," not "does it transfer across complexity levels".
    battery = run_grounding_battery(args.persona, rules=rules,
                                    seeds=BATTERY_SEEDS,
                                    threshold=args.threshold,
                                    sandbox=args.sandbox,
                                    complexity_level=args.complexity_level,
                                    floor_rollouts=args.floor_rollouts,
                                    brain_factory=probe_factory)
    result = {"trained_stable": stable, "sandbox": args.sandbox,
              "complexity_level": args.complexity_level, "status": status_enabled,
              "regime_block_size": args.regime_block_size,
              **battery.as_dict()}
    with open(os.path.join(args.out, "battery.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))
    note = ""
    if battery.inconclusive_rules:
        note = (f"  [INCONCLUSIVE: {', '.join(battery.inconclusive_rules)} — the "
                "behaviour never occurred, so excess there is floor noise, not a "
                "verdict]")
    print(f"\n[done] replay_inexplicable={battery.replay_inexplicable}  "
          f"replay_inexplicable_paired={battery.replay_inexplicable_paired}  "
          f"replay_inexplicable_floor_regression={battery.replay_inexplicable_floor_regression}  "
          f"replay_inexplicable_CONFIRMED={battery.replay_inexplicable_confirmed}"
          f"  weakest={battery.weakest_rule} ({battery.weakest_excess:+.4f}){note}  "
          f"→ paste {args.out}/battery.json to issue #130")
    print("\n[paired stats + floor-regression diagnostic per rule] (one-sided "
          "H1: excess > 0; floor_divergence/excess are always the WORLD-MATCHED "
          "heuristic floor -- floor_rollouts only adds a side-by-side ensemble "
          "read, never moves these. The PRE-REGISTERED verdict is grounded_confirmed "
          "-- a strict AND of grounded_paired and floor_regression_grounded, per "
          "docs/GROUNDING.md; the two below it are context, not alternate passes.)")
    for r, sweep in battery.sweeps.items():
        fr = sweep.floor_regression
        if "slope" in fr:
            slope_lo, slope_hi = fr["slope_ci"]
            fr_str = (f"slope={fr['slope']:+.3f} "
                     f"(CI [{slope_lo:+.3f}, {slope_hi:+.3f}], width={fr['slope_ci_width']:.3f}) "
                     f"n={fr['n']} floor_spread_std={fr['floor_spread_std']:.4f} "
                     f"powered={fr['powered']} "
                     f"residual_sign_p={fr['residual_sign_p']:.4f} "
                     f"residual_wilcoxon_p={fr['residual_wilcoxon_p']:.4f} "
                     f"grounded={sweep.floor_regression_grounded}")
            if not fr["powered"]:
                fr_str += f"  [{fr['note']}]"
        else:
            fr_str = fr["note"]
        lo, hi = sweep.bootstrap_ci_mean_excess
        print(f"  {r:>10}: fraction_grounded={sweep.fraction_grounded:.2f}  "
              f"sign_p={sweep.sign_test_p:.4f}  wilcoxon_p={sweep.wilcoxon_p:.4f}  "
              f"grounded_paired={sweep.grounded_paired}  "
              f"bootstrap_ci=[{lo:+.4f}, {hi:+.4f}]")
        print(f"             floor_regression: {fr_str}")
        confirmed = sweep.grounded_confirmed
        if confirmed is None:
            reading = "UNDETERMINED (floor_regression unpowered -- not a verdict; " \
                      "improve the battery [more conclusive worlds / behaviour " \
                      "density] and re-run, don't conclude anything about the brain)"
        elif confirmed is False:
            reading = "POWERED-NO (a real negative result -- next: representation " \
                      "learnability, not more metric tuning)"
        else:
            reading = "CONFIRMED"
        print(f"             grounded_CONFIRMED (pre-registered verdict): "
              f"{confirmed}  [{reading}]")
        if args.floor_rollouts > 1:
            ens = [r2.ensemble_excess for r2 in sweep.results
                  if r2.ensemble_excess is not None]
            if ens:
                print(f"             ensemble-floor excess per world (cross-check "
                      f"only, not load-bearing): "
                      + ", ".join(f"{x:+.4f}" for x in ens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
