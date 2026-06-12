"""Command-line entry point.

Examples
--------
Run the default 15-day "Claude-like" town offline::

    python -m emergence.cli --persona claude

Compare all four archetypes side by side::

    python -m emergence.cli --compare

Compare the same Philosopher population under four governance systems::

    python -m emergence.cli --compare-gov --persona gemini

Mix personas in one town and emit JSON metrics::

    python -m emergence.cli --persona guardian,predator --json

Run with constitutional governance and write an HTML report::

    python -m emergence.cli --persona philosopher --governance constitutional --html out.html
"""

from __future__ import annotations

import argparse
import json
import sys

from .drives import DrivesConfig
from .governance import GOVERNANCE_PRESETS
from .personas import ALIASES, PERSONAS
from .report import format_report, one_line_verdict
from .scenario import make_simulation
from .simulation import SimulationConfig


def _drives_from_args(args) -> DrivesConfig:
    """Build a DrivesConfig from CLI flags (disabled unless --drives)."""
    repro = getattr(args, "reproduction", False)
    if not getattr(args, "drives", False) and not repro:
        return DrivesConfig()
    return DrivesConfig(enabled=True, reproduction=repro)


def _run_one(persona_mix, args, governance: str = "direct"):
    config = SimulationConfig(days=args.days, ticks_per_day=args.ticks, seed=args.seed)
    sim = make_simulation(persona_mix, n_agents=args.agents, config=config,
                          governance=governance, drives=_drives_from_args(args))
    sim.run(verbose=args.verbose and not args.json)
    return sim


def _compare(args) -> int:
    rows = []
    for key in ("guardian", "philosopher", "idealist", "predator"):
        gov = getattr(args, "governance", "direct")
        sim = _run_one(key, args, governance=gov)
        m = sim.metrics
        rows.append((PERSONAS[key].label, m, one_line_verdict(sim)))
    header = (
        f"{'Society':<12} {'Surv':>5} {'Born':>5} {'Crime':>6} {'Pass%':>6} "
        f"{'Laws':>5} {'Fines':>6} {'Fraud':>6} {'Collab':>7}  Verdict"
    )
    print(header)
    print("-" * len(header))
    for label, m, verdict in rows:
        print(
            f"{label:<12} {m.survivors:>2}/{m.population:<2} {m.births:>5} "
            f"{m.crimes_total:>6} "
            f"{m.pass_rate:>5.0%} {m.laws_enacted:>5} {m.fines_collected:>6} "
            f"{m.frauds:>6} {m.collaborations:>7}  {verdict}"
        )
    return 0


def _compare_gov(args) -> int:
    """Same persona population, four governance forms — what society emerges?"""
    persona_mix = (
        [p.strip() for p in args.persona.split(",")]
        if "," in args.persona
        else args.persona
    )
    rows = []
    for gov_name in ("direct", "oligarchy", "constitutional", "anarchy"):
        sim = _run_one(persona_mix, args, governance=gov_name)
        m = sim.metrics
        rows.append((gov_name, m, one_line_verdict(sim)))

    header = (
        f"{'Governance':<14} {'Surv':>5} {'Crime':>6} {'Pass%':>6} "
        f"{'Laws':>5} {'Fines':>6} {'Tax':>4} {'Fraud':>6}  Verdict"
    )
    print(f"Persona: {args.persona}\n")
    print(header)
    print("-" * len(header))
    for gov_name, m, verdict in rows:
        print(
            f"{gov_name:<14} {m.survivors:>2}/{m.population:<2} {m.crimes_total:>6} "
            f"{m.pass_rate:>5.0%} {m.laws_enacted:>5} {m.fines_collected:>6} "
            f"{m.tax_days:>4} {m.frauds:>6}  {verdict}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    gov_choices = sorted(GOVERNANCE_PRESETS)
    parser = argparse.ArgumentParser(
        prog="emergence",
        description="Run the Emergence World multi-agent town simulation.",
    )
    parser.add_argument(
        "--persona", default="claude",
        help="persona key/alias, or a comma-separated mix "
        f"(keys: {sorted(PERSONAS)}; aliases: {sorted(ALIASES)})",
    )
    parser.add_argument(
        "--governance", default="direct", choices=gov_choices,
        help=f"governance form: {gov_choices}",
    )
    parser.add_argument("--agents", type=int, default=10, help="number of agents")
    parser.add_argument("--days", type=int, default=15, help="days to simulate")
    parser.add_argument("--ticks", type=int, default=8, help="turns per agent per day")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--drives", action="store_true",
                        help="enable the three primal drives (hunger + sleep)")
    parser.add_argument("--reproduction", action="store_true",
                        help="also enable reproduction (implies --drives)")
    parser.add_argument("--json", action="store_true", help="emit JSON metrics only")
    parser.add_argument("--verbose", action="store_true", help="print daily summaries")
    parser.add_argument("--html", metavar="PATH", help="write HTML visualization")
    parser.add_argument("--compare", action="store_true",
                        help="run all four archetypes and compare")
    parser.add_argument("--compare-gov", action="store_true",
                        help="run the same persona under all four governance forms")
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(args)
    if args.compare_gov:
        return _compare_gov(args)

    persona_mix = (
        [p.strip() for p in args.persona.split(",")]
        if "," in args.persona
        else args.persona
    )
    sim = _run_one(persona_mix, args, governance=args.governance)

    if args.html:
        from .viz import write_html
        title = f"Emergence World [{args.persona} / {args.governance}]"
        write_html(sim, args.html, title=title)
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
