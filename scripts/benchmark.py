#!/usr/bin/env python3
"""Engine performance benchmark (#40) -- an agents x days timing matrix for
the heuristic brain, so a regression on the hot path (_observe/_run_tick) is
visible instead of silently creeping in.

This is NOT part of the automated test suite: wall-clock time isn't
deterministic across machines/load, so it can't be asserted on the way
test_baseline_contract asserts on outcomes. Run it by hand before/after a
change to the hot path and compare.

Profiling finding (recorded here since this is the first perf pass on #40):
for a 40-agent/15-day/8-tick heuristic run, _observe accounts for ~90% of
wall time -- it is the O(N^2) cost the issue names (a per-other view built
fresh every tick for every agent). The `agents x ticks` matrix below is
chosen to make that scaling visible.

Usage
-----
    python3 scripts/benchmark.py
    python3 scripts/benchmark.py --agents 10,40,80 --days 10 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.scenario import make_simulation  # noqa: E402
from emergence.simulation import SimulationConfig  # noqa: E402


def _time_run(n_agents: int, days: int, ticks: int, seed: int) -> float:
    sim = make_simulation(
        "guardian", n_agents=n_agents,
        config=SimulationConfig(seed=seed, days=days, ticks_per_day=ticks),
    )
    start = time.perf_counter()
    sim.run()
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default="10,40,80",
                     help="comma-separated agent counts")
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    counts = [int(a) for a in args.agents.split(",")]
    rows = []
    for n in counts:
        elapsed = _time_run(n, args.days, args.ticks, args.seed)
        ticks_total = n * args.days * args.ticks
        rows.append({
            "agents": n, "days": args.days, "ticks_per_day": args.ticks,
            "seconds": round(elapsed, 3),
            "us_per_agent_tick": round(1_000_000 * elapsed / ticks_total, 1),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    header = f"{'agents':>7} {'days':>5} {'ticks/day':>10} {'seconds':>9} {'us/agent-tick':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['agents']:>7} {r['days']:>5} {r['ticks_per_day']:>10} "
              f"{r['seconds']:>9} {r['us_per_agent_tick']:>14}")


if __name__ == "__main__":
    main()
