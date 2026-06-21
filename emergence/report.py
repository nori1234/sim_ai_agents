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
    if m.matings or m.births:
        avg_joy = m.total_pleasure / m.survivors if m.survivors else 0.0
        lines.append(
            f"Instinct:    {m.matings} matings -> {m.births} births  "
            f"(pleasure banked {m.total_pleasure:.0f}, ~{avg_joy:.0f}/survivor)"
        )
    if m.total_praise:
        top = max((a for a in sim.agents), key=lambda a: a.reputation, default=None)
        honoured = f"; most honoured: {top.name} ({top.reputation:.0f})" if top else ""
        lines.append(f"Esteem:      {m.total_praise} praises exchanged{honoured}")
    if m.works_created or m.peak_fear:
        lines.append(
            f"Psyche:      {m.works_created} works created "
            f"(fulfillment {m.total_fulfillment:.0f}); peak fear {m.peak_fear:.0f}/100"
        )
    if (m.weapons_crafted or m.gangs_formed or m.doses_taken or m.religions_founded):
        lines.append(
            f"Underworld:  {m.weapons_crafted} armed, {m.gangs_formed} gangs, "
            f"{m.rebellions} rebellions; {m.drug_deals} deals / "
            f"{m.doses_taken} doses ({m.addicts} addicts)"
        )
        lines.append(
            f"Culture:     {m.religions_founded} faiths, {m.conversions} converts, "
            f"{m.acts_of_worship} acts of worship"
        )
    if m.disasters_total or m.final_season:
        lines.append(
            f"Environment: {m.disasters_total} disasters; peak food price "
            f"{m.peak_food_price:.2f}x; ended in {m.final_season}"
        )
    if m.public_works_built or m.treasury_final:
        lines.append(
            f"Public works: {m.public_works_built} built by the council; "
            f"treasury {m.treasury_final}"
        )
    if m.prosperity:
        lines.append(f"Prosperity:  {m.prosperity:.0f}/100 (historical development index)")
    if m.trades or m.crafted:
        price = sim.emergent_price("food", "money")
        price_txt = f"; food≈{price} money" if price else ""
        lines.append(f"Economy:     {m.trades} trades, {m.crafted} crafted{price_txt}")
    if m.loans_made:
        lines.append(
            f"Credit:      {m.loans_made} loans, {m.loans_repaid} repaid, "
            f"{m.loan_defaults} defaulted"
        )
    lines.append(f"Crimes:      {m.crimes_total}")
    if m.arrests:
        lines.append(
            f"Enforcement: {m.arrests} arrests by guards "
            f"(the peace is kept by acts, not auras)"
        )
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


# Each verdict in English (the default / contract baseline) and Japanese.
_VERDICTS = {
    "collapse":     ("COLLAPSE — the town died out.",
                     "崩壊 — 町は滅びた。"),
    "flourishing":  ("FLOURISHING — a peaceful society that grew its population.",
                     "繁栄 — 平和なまま人口が増えた社会。"),
    "fertile_chaos":("FERTILE CHAOS — the population grew amid pervasive crime.",
                     "豊穣な混沌 — 犯罪の中でも人口が増えた。"),
    "growing":      ("GROWING — an imperfect society that still expanded.",
                     "成長 — 不完全だが拡大した社会。"),
    "order":        ("ORDER — peaceful, fully cooperative, but highly conformist.",
                     "秩序 — 平和・全面協調・だが強い同調圧力。"),
    "chaos":        ("CHAOS — pervasive crime and violence.",
                     "混沌 — 犯罪と暴力が蔓延。"),
    "failure":      ("FAILURE — the society failed to sustain its population.",
                     "失敗 — 人口を維持できなかった社会。"),
    "mixed":        ("MIXED — a functioning but imperfect society.",
                     "混合 — 機能はするが不完全な社会。"),
}


def _verdict_key(sim: Simulation) -> str:
    m = sim.metrics
    if m.survivors == 0:
        return "collapse"
    if m.births > 0 and m.survivors > m.population:
        if m.crimes_total == 0:
            return "flourishing"
        if m.crimes_total >= 100:
            return "fertile_chaos"
        return "growing"
    if m.crimes_total == 0 and m.survival_rate >= 1.0:
        return "order"
    if m.crimes_total >= 100:
        return "chaos"
    if m.survival_rate < 0.5:
        return "failure"
    return "mixed"


def one_line_verdict(sim: Simulation, lang: str = "en") -> str:
    """A terse characterisation of the society that emerged (English by default,
    Japanese with ``lang='ja'``). The English text is unchanged from before, so
    the four-society contract stays byte-identical."""
    en, ja = _VERDICTS[_verdict_key(sim)]
    return ja if lang == "ja" else en
