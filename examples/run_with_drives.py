"""Run the simulation with the three primal drives enabled.

Demonstrates how appetite (食欲), sleep (睡眠欲) and reproduction (性欲) reshape
each society. The Guardian town, with every need met, grows its population;
the neglectful Idealist town cannot even feed itself, let alone raise children.

    python examples/run_with_drives.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.drives import DrivesConfig
from emergence.report import format_report, one_line_verdict
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig


def main() -> None:
    config = SimulationConfig(days=15, ticks_per_day=8, seed=42)
    drives = DrivesConfig(enabled=True, reproduction=True)

    print("15-day towns with hunger + sleep + reproduction enabled:\n")
    for key in ("guardian", "philosopher", "idealist", "predator"):
        sim = make_simulation(key, config=config, drives=drives)
        sim.run()
        m = sim.metrics
        print(f"  [{key:<11}] start 10 -> {m.survivors:>2} alive, "
              f"{m.births:>2} born | {one_line_verdict(sim)}")

    print("\nFull report for the Guardian (Claude-like) town:\n")
    sim = make_simulation("guardian", config=config, drives=drives)
    sim.run(verbose=True)
    print()
    print(format_report(sim, title="Emergence World [guardian + drives]"))


if __name__ == "__main__":
    main()
