"""Human-readable post-mortem of a finished run."""

from __future__ import annotations

from .simulation import Simulation


def format_report(sim: Simulation, title: str = "Emergence World") -> str:
    m = sim.metrics
    lines: list[str] = []
    lines.append(f"=== {title} — {m.days_run}-day report ===")
    lines.append("")
    lines.append(
        f"Survivors:   {m.survivors} living  "
        f"(started with {m.population}, survival rate {m.survival_rate:.0%})"
    )
    if m.births:
        lines.append(f"Births:      {m.births} children born during the run")
    lines.append(f"Crimes:      {m.crimes_total}")
    if m.crimes_by_type:
        for kind, count in sorted(m.crimes_by_type.items(), key=lambda kv: -kv[1]):
            lines.append(f"   - {kind:<10} {count}")
    lines.append(
        f"Proposals:   {m.proposals_total} total, "
        f"{m.proposals_passed} passed / {m.proposals_rejected} rejected "
        f"(pass rate {m.pass_rate:.0%})"
    )
    lines.append(f"Gifts:       {m.transfers} resource transfers given freely")
    lines.append(f"Fraud:       {m.frauds} fraudulent solicitations (\"I'm broke\" scams)")
    lines.append(
        f"Cooperation: {m.collaborations} collaborations, "
        f"{m.monuments_built} monuments, {m.granary_deposits} granary deposits"
    )
    lines.append("")

    # Cause-of-death roll call.
    dead = [a for a in sim.agents if not a.alive]
    if dead:
        lines.append("Fallen:")
        for a in sorted(dead, key=lambda x: (x.day_of_death or 0)):
            lines.append(
                f"   - {a.name} ({a.profession}) — {a.cause_of_death} "
                f"on day {a.day_of_death}"
            )
        lines.append("")

    lines.append("Citizens:")
    for a in sim.agents:
        status = "alive" if a.alive else f"died d{a.day_of_death}"
        lines.append(
            f"   - {a.name:<10} {a.profession:<12} [{a.persona:<11}] "
            f"{status:<11} crimes={a.crimes_committed} frauds={a.frauds_committed} "
            f"votes={a.votes_cast} collabs={a.collaborations}"
        )
    return "\n".join(lines)


def one_line_verdict(sim: Simulation) -> str:
    """A terse characterisation of the society that emerged."""
    m = sim.metrics
    if m.survivors == 0:
        return "COLLAPSE — the town died out."
    # Population growth (only possible with reproduction enabled).
    if m.births > 0 and m.survivors > m.population:
        if m.crimes_total == 0:
            return "FLOURISHING — a peaceful society that grew its population."
        if m.crimes_total >= 100:
            return "FERTILE CHAOS — the population grew amid pervasive crime."
        return "GROWING — an imperfect society that still expanded."
    if m.crimes_total == 0 and m.survival_rate >= 1.0:
        return "ORDER — peaceful, fully cooperative, but highly conformist."
    if m.crimes_total >= 100:
        return "CHAOS — pervasive crime and violence."
    if m.survival_rate < 0.5:
        return "FAILURE — the society failed to sustain its population."
    return "MIXED — a functioning but imperfect society."
