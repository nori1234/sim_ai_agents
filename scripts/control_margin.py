#!/usr/bin/env python3
"""Control-side deposit margin — the diagnostic run #14's collapse demanded.

Run #14 (the first fair-task training run, ``sole_banker=True``) came back
POWERED-NO: the trained policy collapsed to a regime-**independent**
never-deposit policy (15 control / 18 counterfactual deposit attempts summed
over 20 worlds, vs the heuristic floor's dense depositing). The engine-side
note on issue #130 conjectured a cause: "both arms' margins are deliberately
small (the +0.21 oracle advantage), so the control-side pull toward depositing
(interest) may be weak relative to reward noise."

The S6 deposit-only oracle (``scripts/deposit_oracle.py``) only ever measured
the **cf-side** margin — ``advantage_cf`` = the reward for HOLDING cash in the
counterfactual world (the transition from an always-deposit policy to a
grounded one), +0.21 / +0.20 sigma. The **control-side** pull — the reward for
depositing in control, i.e. the gradient a never-deposit policy must climb to
become regime-contingent — was never measured. This script measures it, plus
the full 2x2 of {deposit-per-rule (blind), never-deposit} x {control, cf}, so
the reward every simple policy sees is on the table at once.

It reuses ``measure_deposit_oracle``'s own internals (same sandbox, seeds,
brains, and the exact telescoped ``survival_reward`` the RL policy optimizes),
so the numbers are directly comparable to every earlier S6 result. Read-only:
it changes no task or reward term. No torch, no training, no CI — deterministic
and fast (seconds).

Usage
-----
    python3 scripts/control_margin.py --persona guardian --sole-banker
    python3 scripts/control_margin.py --persona guardian --sole-banker --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence import grounding as g  # noqa: E402


def _summary(xs):
    m = st.mean(xs)
    sd = st.pstdev(xs)
    return {
        "mean": m,
        "std": sd,
        "effect_size": (m / sd) if sd > 1e-9 else float("nan"),
        "worlds_positive": sum(1 for x in xs if x > 1e-9),
        "n": len(xs),
    }


def measure_control_margin(persona="guardian", *, seeds=tuple(range(42, 62)),
                            days=20, n_agents=6, sole_banker=False,
                            complexity_level=0):
    Oracle = g._deposit_only_oracle_brain_class()

    def blind_factory(agent, persona, rng):
        from emergence.brains.heuristic import HeuristicBrain
        return HeuristicBrain(persona, rng)

    def never_factory(agent, persona, rng):  # never deposits, in either regime
        return Oracle(persona, rng, skip_deposit=True)

    def run(cf_enabled, factory, seed):
        sim = g.make_grounding_sandbox(
            persona, rule="demurrage", n_savers=n_agents - 1, seed=seed,
            days=days, cf_enabled=cf_enabled, brain_factory=factory,
            complexity_level=complexity_level, sole_banker=sole_banker)
        agent = sim.agents[1]  # agents[0] is the banker
        ret, alive, _ = g._episode_outcome(sim, agent, None)
        return ret, alive

    per_world = []
    for s in seeds:
        bc, bc_alive = run(False, blind_factory, s)   # deposit in control
        nc, nc_alive = run(False, never_factory, s)   # hold cash in control
        bf, bf_alive = run(True, blind_factory, s)    # deposit in cf
        nf, nf_alive = run(True, never_factory, s)    # hold cash in cf
        per_world.append({
            "seed": s,
            "blind_control": bc, "never_control": nc,
            "blind_cf": bf, "never_cf": nf,
            "all_alive": all([bc_alive, nc_alive, bf_alive, nf_alive]),
        })

    def alive_only(key_pair):
        a, b = key_pair
        return [r[a] - r[b] for r in per_world if r["all_alive"]]

    control_pull = [r["blind_control"] - r["never_control"] for r in per_world]
    cf_advantage = [r["never_cf"] - r["blind_cf"] for r in per_world]

    grounded = [r["blind_control"] + r["never_cf"] for r in per_world]
    never = [r["never_control"] + r["never_cf"] for r in per_world]
    always = [r["blind_control"] + r["blind_cf"] for r in per_world]

    return {
        "persona": persona, "sole_banker": sole_banker, "n_worlds": len(seeds),
        "seeds": list(seeds),
        # the two margins to a grounded policy
        "control_pull": _summary(control_pull),          # from never-deposit
        "control_pull_survivors": _summary(alive_only(("blind_control", "never_control"))),
        "cf_advantage": _summary(cf_advantage),          # from always-deposit (== advantage_cf)
        "cf_advantage_survivors": _summary(alive_only(("never_cf", "blind_cf"))),
        # policy-level returns (per-world sum of the two regime cells)
        "return_grounded": _summary(grounded),
        "return_never_deposit": _summary(never),          # run #14's collapse
        "return_always_deposit": _summary(always),        # regime-blind, dominant-reward
        # the reward for regime-CONTINGENCY specifically (grounded vs always-deposit)
        "contingency_margin": _summary([gd - al for gd, al in zip(grounded, always)]),
        "per_world": per_world,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", default="guardian")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; default = battery's 20 held-out (42-61)")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--sole-banker", action="store_true",
                    help="the S6 task redesign run #14 trained on")
    ap.add_argument("--complexity-level", type=int, default=0)
    args = ap.parse_args()

    kwargs = {}
    if args.seeds:
        kwargs["seeds"] = tuple(int(s) for s in args.seeds.split(","))

    res = measure_control_margin(
        args.persona, days=args.days, n_agents=args.agents,
        sole_banker=args.sole_banker, complexity_level=args.complexity_level,
        **kwargs)

    print(json.dumps(res, indent=2))

    def line(label, s):
        return (f"{label:34s} {s['mean']:+8.4f}  (std {s['std']:6.4f}, "
                f"{s['effect_size']:+5.2f} sigma, {s['worlds_positive']}/{s['n']} worlds +)")

    print(f"\n[control-margin] persona={res['persona']} sole_banker={res['sole_banker']} "
          f"worlds={res['n_worlds']}")
    print(line("CONTROL pull (deposit in control)", res["control_pull"]))
    print(line("  survivors-only", res["control_pull_survivors"]))
    print(line("CF advantage (hold cash in cf)", res["cf_advantage"]))
    print(line("  survivors-only", res["cf_advantage_survivors"]))
    print(line("CONTINGENCY margin (grnd-always)", res["contingency_margin"]))
    print("  --- policy returns ---")
    print(line("grounded (dep C / hold CF)", res["return_grounded"]))
    print(line("never-deposit (run #14 collapse)", res["return_never_deposit"]))
    print(line("always-deposit (regime-blind)", res["return_always_deposit"]))


if __name__ == "__main__":
    main()
