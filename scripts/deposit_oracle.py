#!/usr/bin/env python3
"""Deposit-only oracle (S6) — the brain team's clean-spec reward-ceiling
variant, run to split "the task doesn't pay for grounding" from "the task
pays but variance kills the effective gradient".

The oracle is the blind heuristic with exactly one behavioural difference:
in the counterfactual world only, a DEPOSIT decision is dropped (the cash is
held) and control falls through to the blind heuristic's own next branch —
no REST substitute, and withdraw / lending / OFFER / REPAY / every other
branch identical to blind. See emergence/grounding.py's "Deposit-only
oracle (S6)" section for why the SIGN of advantage_cf decides between task
redesign and learning-side variance fixes, and why the output carries
per-world returns, survival, the blind heuristic's per-world return
variance, and the effect size (advantage / blind-cf std).

No torch, no training, no CI needed — deterministic and fast (seconds).

Usage
-----
    python3 scripts/deposit_oracle.py --persona guardian
    python3 scripts/deposit_oracle.py --persona guardian --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.grounding import measure_deposit_oracle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", default="guardian")
    ap.add_argument("--rule", default="demurrage")
    ap.add_argument("--seeds", default=None,
                     help="comma-separated seed list; default matches the "
                          "battery's 20 held-out worlds (42-61)")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--complexity-level", type=int, default=0)
    args = ap.parse_args()

    kwargs = {}
    if args.seeds:
        kwargs["seeds"] = tuple(int(s) for s in args.seeds.split(","))

    result = measure_deposit_oracle(
        args.persona, rule=args.rule, days=args.days, n_agents=args.agents,
        complexity_level=args.complexity_level, **kwargs)

    d = result.as_dict()
    print(json.dumps(d, indent=2))
    print(f"\n[deposit-oracle] advantage_cf (oracle - blind, counterfactual): "
          f"{d['advantage_counterfactual']:+.4f} "
          f"(control sanity check: {d['advantage_control']:+.4f}, must be 0)")
    print(f"[deposit-oracle] blind cf per-world std: {d['blind_cf_std']:.4f} "
          f"(variance {d['blind_cf_variance']:.4f}) -> effect size "
          f"{d['effect_size']:+.4f}")
    print(f"[deposit-oracle] oracle ahead in {d['worlds_oracle_ahead']}/"
          f"{d['n_worlds']} worlds; oracle deaths in cf: "
          f"{d['oracle_cf_deaths']}/{d['n_worlds']}")


if __name__ == "__main__":
    main()
