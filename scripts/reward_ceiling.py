#!/usr/bin/env python3
"""Reward ceiling — does the TASK pay enough for grounding to be worth
learning, independent of whether any policy currently learns it?

Compares the blind heuristic's own realized return (survival_reward, see
emergence/brains/_neural_reward.py) against a scripted oracle that is handed
the ground-truth regime directly and never deposits under the counterfactual
rule -- the most any policy, however well it learns, could gain from
discriminating this regime. See emergence/grounding.py's "Reward ceiling"
section for the full argument and docs/GROUNDING.md for how this fits with
the perception-side ceiling (the regime-decoding probe).

No torch, no training, no CI needed -- deterministic and fast (seconds).

Usage
-----
    python3 scripts/reward_ceiling.py --persona guardian
    python3 scripts/reward_ceiling.py --persona guardian --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.grounding import measure_reward_ceiling  # noqa: E402


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

    result = measure_reward_ceiling(
        args.persona, rule=args.rule, days=args.days, n_agents=args.agents,
        complexity_level=args.complexity_level, **kwargs)

    d = result.as_dict()
    print(json.dumps(d, indent=2))
    print(f"\n[reward-ceiling] blind's own within-policy regime gap "
          f"(control - counterfactual): "
          f"{d['blind_return_control'] - d['blind_return_counterfactual']:+.4f}")
    print(f"[reward-ceiling] grounded oracle's advantage over blind, "
          f"counterfactual: {d['advantage_counterfactual']:+.4f} "
          f"(control sanity check: {d['advantage_control']:+.4f}, must be 0)")


if __name__ == "__main__":
    main()
