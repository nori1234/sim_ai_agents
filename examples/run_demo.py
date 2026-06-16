"""Run each archetype and print a full report for one of them.

    python examples/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.report import format_report, one_line_verdict
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def main() -> None:
    config = SimulationConfig(days=15, ticks_per_day=8, seed=42)

    print("Running four 15-day towns, one per archetype...\n")
    for key in ("guardian", "philosopher", "idealist", "predator"):
        sim = make_simulation(key, config=config)
        sim.run()
        print(f"[{key:<11}] {one_line_verdict(sim)}")

    print("\nFull report for the Guardian (Claude-like) town:\n")
    sim = make_simulation("guardian", config=config)
    sim.run(verbose=True)
    print()
    print(format_report(sim, title="Emergence World [guardian]"))


if __name__ == "__main__":
    main()
