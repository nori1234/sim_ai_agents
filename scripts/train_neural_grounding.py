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
     provides the L0-L1 scaffolding.
  2. Periodically save a full checkpoint (dev.save_checkpoint — policy included)
     and probe it with GroundingMonitor (frozen: learn=False); stop early once
     every rule's monitor reports is_stable(window).
  3. Run run_grounding_battery with the checkpointed brain and write
     battery.json — paste battery.as_dict() to issue #130. A negative result is
     a real, reportable result.

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train in the real engine, then run the battery (#130)")
    ap.add_argument("--episodes", type=int, default=60, help="max training episodes")
    ap.add_argument("--days", type=int, default=20, help="days per training episode")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42, help="base seed (episode i uses seed+i)")
    ap.add_argument("--persona", default="guardian",
                    help="floor persona; guardian exercises all three scored behaviours")
    ap.add_argument("--probe-every", type=int, default=5, help="probe cadence (episodes)")
    ap.add_argument("--window", type=int, default=3, help="is_stable window (probes)")
    ap.add_argument("--threshold", type=float, default=0.0)
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

    monitors = {
        r: GroundingMonitor(args.persona, rule=r, days=args.days,
                            n_agents=args.agents, seed=args.seed,
                            threshold=args.threshold, sandbox=args.sandbox)
        for r in rules
    }

    def build_episode(ep: int):
        """One training episode. In sandbox mode, alternate the control and
        demurrage worlds in the minimal deposit-decision town (dense behaviour);
        in full-town mode, rotate control + each of the three inverted worlds."""
        if args.sandbox:
            return make_grounding_sandbox(
                args.persona, rule="demurrage", n_savers=args.agents - 1,
                seed=args.seed + ep, days=args.days,
                cf_enabled=(ep % 2 == 1), brain_factory=training_factory)
        rule = EPISODE_ROTATION[ep % len(EPISODE_ROTATION)]
        return make_simulation(
            args.persona, n_agents=args.agents, economy=True,
            status=StatusConfig(enabled=True),
            config=SimulationConfig(seed=args.seed + ep, days=args.days),
            counterfactual=CounterfactualConfig(
                enabled=rule is not None, rule=rule or "demurrage",
                hide_rate=True, instrument=True),
            brain_factory=training_factory,
        )

    where = "sandbox" if args.sandbox else "full town"
    print(f"[train] up to {args.episodes} episodes x {args.days} days, "
          f"{args.agents} agents, persona={args.persona}, {where}, "
          f"rules={','.join(rules)}, hparams={hparams}", flush=True)
    stable = False
    for ep in range(args.episodes):
        sim = build_episode(ep)
        sim.run()

        first = brains[next(iter(brains))]
        if first._broken or first._dev is None:
            sys.exit("[fatal] the neural backend is not live (fell back to the "
                     "heuristic). Install torch + llm_model_agi and retry — a "
                     "heuristic-only run would train nothing.")

        print(f"[train] episode {ep + 1}/{args.episodes} done", flush=True)

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

    print(f"[battery] running the acceptance battery ({','.join(rules)} x all "
          f"worlds, {where})...", flush=True)
    battery = run_grounding_battery(args.persona, rules=rules,
                                    threshold=args.threshold,
                                    sandbox=args.sandbox,
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
          f"weakest={battery.weakest_rule} ({battery.weakest_excess:+.4f}){note}  "
          f"→ paste {args.out}/battery.json to issue #130")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
