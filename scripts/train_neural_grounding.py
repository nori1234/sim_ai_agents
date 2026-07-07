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
floor_divergence` only removes an *additive* floor confound; it does nothing
about a *slope* confound, and the floor itself is a noisy point estimate (a
finite run's count of a scored behaviour). Three independent responses, none of
which changes what run_grounding_battery gates replay_inexplicable on:
  1. --floor-rollouts averages the floor over several independent worlds instead
     of trusting a single draw (emergence.grounding: run_grounding_probe).
  2. The battery now also reports paired statistics (sign test, Wilcoxon
     signed-rank, a bootstrap CI) over the conclusive worlds' excess —
     SweepResult.sign_test_p/wilcoxon_p/bootstrap_ci_mean_excess — a harder-to-
     Goodhart read than the hard-threshold fraction_grounded, not a replacement
     for it.
  3. SweepResult.floor_regression regresses each world's raw divergence on its
     floor_divergence and tests the residual against zero — immune to a floor
     confound of ANY linear form (slope or offset), not just the additive one
     `excess` assumes.
BATTERY_SEEDS was also widened (5 -> 20 held-out worlds) for statistical power;
by itself this does not de-bias anything, which is why (1)-(3) exist.

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
                    help="average each battery world's heuristic floor over this "
                         "many independent worlds instead of a single draw "
                         "(reduces the floor's own sampling noise — see the "
                         "floor-confound note in this module's docstring).")
    ap.add_argument("--sandbox", action="store_true",
                    help="train + measure in the minimal sandbox (dense behaviour, "
                         "conclusive). demurrage only — the sandbox's supported rule.")
    ap.add_argument("--out", default="grounding_out", help="output dir (ckpt, logs, battery.json)")
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
                            threshold=args.threshold, sandbox=args.sandbox)
        for r in rules
    }

    def build_episode(ep: int):
        """One training episode. Domain-randomizes the world seed by rotating
        through train_pool (disjoint from BATTERY_SEEDS — see the module
        docstring): a policy that only ever sees one world can converge on that
        world's incidental layout rather than the rule, which is exactly what
        run #5 showed (is_stable reached, fraction_grounded unmoved). In sandbox
        mode, alternate the control and demurrage worlds in the minimal
        deposit-decision town (dense behaviour); in full-town mode, rotate
        control + each of the three inverted worlds."""
        seed = train_pool[ep % len(train_pool)]
        if args.sandbox:
            return make_grounding_sandbox(
                args.persona, rule="demurrage", n_savers=args.agents - 1,
                seed=seed, days=args.days,
                cf_enabled=(ep % 2 == 1), brain_factory=training_factory)
        rule = EPISODE_ROTATION[ep % len(EPISODE_ROTATION)]
        return make_simulation(
            args.persona, n_agents=args.agents, economy=True,
            status=StatusConfig(enabled=True),
            config=SimulationConfig(seed=seed, days=args.days),
            counterfactual=CounterfactualConfig(
                enabled=rule is not None, rule=rule or "demurrage",
                hide_rate=True, instrument=True),
            brain_factory=training_factory,
        )

    where = "sandbox" if args.sandbox else "full town"
    print(f"[train] up to {args.episodes} episodes x {args.days} days, "
          f"{args.agents} agents, persona={args.persona}, {where}, "
          f"rules={','.join(rules)}, hparams={hparams}, "
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
          f"held-out worlds {list(BATTERY_SEEDS)}, {where}, "
          f"floor_rollouts={args.floor_rollouts})...", flush=True)
    # Explicit, not the function's own default: this script's train/eval
    # separation is asserted against BATTERY_SEEDS specifically, so evaluation
    # must use exactly that set even if the library default ever changes.
    battery = run_grounding_battery(args.persona, rules=rules,
                                    seeds=BATTERY_SEEDS,
                                    threshold=args.threshold,
                                    sandbox=args.sandbox,
                                    floor_rollouts=args.floor_rollouts,
                                    brain_factory=probe_factory)
    result = {"trained_stable": stable, **battery.as_dict()}
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
          f"weakest={battery.weakest_rule} ({battery.weakest_excess:+.4f}){note}  "
          f"→ paste {args.out}/battery.json to issue #130")
    print("\n[paired stats + floor-regression diagnostic per rule] "
          "(one-sided H1: excess > 0; see docs/GROUNDING.md)")
    for r, sweep in battery.sweeps.items():
        fr = sweep.floor_regression
        fr_str = (fr["note"] if "note" in fr else
                 f"slope={fr['slope']:+.3f} residual_sign_p={fr['residual_sign_p']:.4f} "
                 f"residual_wilcoxon_p={fr['residual_wilcoxon_p']:.4f}")
        lo, hi = sweep.bootstrap_ci_mean_excess
        print(f"  {r:>10}: fraction_grounded={sweep.fraction_grounded:.2f}  "
              f"sign_p={sweep.sign_test_p:.4f}  wilcoxon_p={sweep.wilcoxon_p:.4f}  "
              f"grounded_paired={sweep.grounded_paired}  "
              f"bootstrap_ci=[{lo:+.4f}, {hi:+.4f}]  floor_regression: {fr_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
