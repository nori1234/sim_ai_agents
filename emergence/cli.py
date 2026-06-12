"""Command-line entry point: run a simulation and print the report.

Examples
--------
Run the default 15-day "Claude-like" town offline::

    python -m emergence.cli --persona claude

Compare all four archetypes side by side::

    python -m emergence.cli --compare

Mix personas in one town and emit JSON metrics::

    python -m emergence.cli --persona guardian,predator --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .personas import ALIASES, PERSONAS
from .report import format_report, one_line_verdict
from .scenario import make_simulation
from .simulation import SimulationConfig


def _run_one(persona_mix, args) -> "tuple":
    config = SimulationConfig(days=args.days, ticks_per_day=args.ticks, seed=args.seed)
    sim = make_simulation(persona_mix, n_agents=args.agents, config=config)
    sim.run(verbose=args.verbose and not args.json)
    return sim


def _compare(args) -> int:
    rows = []
    for key in ("guardian", "philosopher", "idealist", "predator"):
        sim = _run_one(key, args)
        m = sim.metrics
        rows.append((PERSONAS[key].label, m, one_line_verdict(sim)))
    header = (
        f"{'Society':<12} {'Surv':>5} {'Crime':>6} {'Pass%':>6} "
        f"{'Fraud':>6} {'Collab':>7}  Verdict"
    )
    print(header)
    print("-" * len(header))
    for label, m, verdict in rows:
        print(
            f"{label:<12} {m.survivors:>2}/{m.population:<2} {m.crimes_total:>6} "
            f"{m.pass_rate:>5.0%} {m.frauds:>6} {m.collaborations:>7}  {verdict}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="emergence",
        description="Run the Emergence World multi-agent town simulation.",
    )
    parser.add_argument(
        "--persona",
        default="claude",
        help="persona key/alias, or a comma-separated mix "
        f"(keys: {sorted(PERSONAS)}; aliases: {sorted(ALIASES)})",
    )
    parser.add_argument("--agents", type=int, default=10, help="number of agents")
    parser.add_argument("--days", type=int, default=15, help="days to simulate")
    parser.add_argument("--ticks", type=int, default=8, help="turns per agent per day")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--json", action="store_true", help="emit JSON metrics only")
    parser.add_argument("--verbose", action="store_true", help="print daily summaries")
    parser.add_argument(
        "--html", metavar="PATH", default=None,
        help="write a self-contained HTML visualization to PATH",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run all four archetypes and print a comparison table",
    )
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(args)

    persona_mix = (
        [p.strip() for p in args.persona.split(",")]
        if "," in args.persona
        else args.persona
    )
    sim = _run_one(persona_mix, args)

    if args.html:
        from .viz import write_html
        write_html(sim, args.html, title=f"Emergence World [{args.persona}]")
        print(f"Wrote visualization to {args.html}")

    if args.json:
        print(json.dumps(sim.metrics.as_dict(), ensure_ascii=False, indent=2))
    elif not args.html:
        print(format_report(sim, title=f"Emergence World [{args.persona}]"))
        print()
        print("Verdict:", one_line_verdict(sim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
