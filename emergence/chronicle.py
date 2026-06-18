"""Turn a run into a *story* — the heart of the observatory experience.

The engine emits a flat event log; this module curates it into something a
person wants to read: a **chronicle** of the town's days (the legislative spam
summarised, the dramatic beats surfaced) and a **life story** for any citizen
(their role versus what they actually did, their ties, their faith, their fate).

Pure stdlib and deterministic — the same run always tells the same story. A
future layer can hand these beats to an LLM to render flowing prose; this is the
grounded, free foundation it would build on.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# Event kinds that are dramatic enough to surface individually.
_HEADLINE = {
    "death", "gang_formed", "religion_founded", "rebellion", "monument",
    "conversion", "public_works",
}
# The roles an agent id can play in an event (for life-story extraction).
_ACTOR_FIELDS = ("offender", "victim", "leader", "prophet", "agent", "by",
                 "instigator", "author", "convert", "guard", "sender", "receiver")


def _names(sim) -> dict:
    return {a.id: a.name for a in sim.agents}


def _beat(e: dict, nm) -> str | None:
    """Render one headline event as a sentence (or None to skip)."""
    k = e["kind"]
    if k == "death":            return f"💀 {nm(e['agent'])} died — {e.get('cause','?')}"
    if k == "gang_formed":      return f"🩸 {nm(e['leader'])} founded the {e['gang']}"
    if k == "religion_founded": return f"🛐 {nm(e['prophet'])} founded the faith {e['faith']}"
    if k == "conversion":       return f"🛐 {nm(e['convert'])} joined {e['faith']}"
    if k == "rebellion":        return f"🚩 {nm(e['instigator'])} led a rebellion"
    if k == "monument":         return f"🏛 {nm(e['by'])} raised {e['name']}"
    if k == "public_works":     return f"🏗 the council built a {e.get('type','facility').replace('_',' ')}"
    return None


def chronicle(sim) -> list[dict]:
    """A curated, day-by-day story: ``[{"day": int, "beats": [str]}]``."""
    nm = (lambda i, m=_names(sim): m.get(i, i))
    by_day: dict[int, list[dict]] = defaultdict(list)
    for e in sim.world.events:
        by_day[e.get("day", 0)].append(e)

    out = []
    for day in sorted(by_day):
        evs = by_day[day]
        beats: list[str] = []
        # Headline drama, in order.
        for e in evs:
            line = _beat(e, nm)
            if line:
                beats.append(line)
        # Summarise the noisy machinery rather than listing each item.
        crimes = Counter(e["kind"] for e in evs if e["kind"] in ("violence", "theft", "arson"))
        if crimes:
            bits = ", ".join(f"{n} {k}" for k, n in crimes.items())
            beats.append(f"⚔ unrest: {bits}")
        passed = sum(1 for e in evs if e["kind"] == "proposal_resolved" and e.get("status") == "passed")
        laws = [e["effects"] for e in evs if e["kind"] == "law_enacted" and e.get("effects")]
        if passed or laws:
            effects = ", ".join(sorted({x for eff in laws for x in eff.split(", ")}))
            tail = f" — enacting {effects}" if effects else ""
            beats.append(f"🏛 the council passed {passed} bill(s){tail}")
        births = sum(1 for e in evs if e["kind"] == "birth")
        if births:
            beats.append(f"👶 {births} child(ren) born")
        if beats:
            out.append({"day": day, "beats": beats})
    return out


def chronicle_text(sim, title: str = "Town Chronicle") -> str:
    lines = [f"# {title}", ""]
    for entry in chronicle(sim):
        lines.append(f"**Day {entry['day']}**")
        lines += [f"- {b}" for b in entry["beats"]]
        lines.append("")
    return "\n".join(lines).strip()


def life_story(sim, agent_id: str) -> dict:
    """One citizen's life: their role vs their deeds, ties, beliefs, and fate."""
    a = sim._by_id.get(agent_id)
    if a is None:
        raise KeyError(agent_id)
    nm = (lambda i, m=_names(sim): m.get(i, i))

    # The acts of this life, in order — narrated from the citizen's vantage.
    beats: list[str] = []
    for e in sim.world.events:
        if not any(e.get(f) == agent_id for f in _ACTOR_FIELDS):
            continue
        k, d = e["kind"], e.get("day", 0)
        if k == "gang_formed":
            beats.append(f"D{d}: founded the {e['gang']}")
        elif k == "religion_founded":
            beats.append(f"D{d}: founded the faith {e['faith']} — a prophet now")
        elif k == "conversion" and e.get("convert") == agent_id:
            beats.append(f"D{d}: converted to {e['faith']}")
        elif k == "violence" and e.get("offender") == agent_id:
            beats.append(f"D{d}: attacked {nm(e['victim'])}")
        elif k == "violence" and e.get("victim") == agent_id:
            beats.append(f"D{d}: was attacked by {nm(e['offender'])}")
        elif k == "theft" and e.get("offender") == agent_id:
            beats.append(f"D{d}: stole from {nm(e['victim'])}")
        elif k == "arson" and e.get("offender") == agent_id:
            beats.append(f"D{d}: set fire to {e.get('facility','a building')}")
        elif k == "arrest" and e.get("offender") == agent_id:
            beats.append(f"D{d}: was arrested by {nm(e['guard'])}")
        elif k == "arrest" and e.get("guard") == agent_id:
            beats.append(f"D{d}: made an arrest — kept the peace")
        elif k == "monument" and e.get("by") == agent_id:
            beats.append(f"D{d}: raised the monument {e['name']}")
        elif k == "rebellion" and e.get("instigator") == agent_id:
            beats.append(f"D{d}: led a rebellion against those in power")
        elif k == "death" and e.get("agent") == agent_id:
            beats.append(f"D{d}: died — {e.get('cause','?')}")
        elif k == "birth" and e.get("child") == agent_id:
            beats.append(f"D{d}: was born")

    ties = sorted(a.trust.items(), key=lambda kv: -kv[1])
    allies = [f"{nm(i)} ({t:+.1f})" for i, t in ties if t > 0.3][:3]
    foes = [f"{nm(i)} ({t:+.1f})" for i, t in ties if t < -0.3][:3]
    beliefs = []
    if a.faith:
        beliefs.append("devout")
    if a.gang_id:
        beliefs.append("gang-sworn")
    if a.weapons:
        beliefs.append(f"armed ({a.weapons})")

    if a.alive:
        fate = f"Survived to the end, standing {a.reputation:.0f}."
    else:
        fate = f"Died on day {a.day_of_death} — {a.cause_of_death}."

    return {
        "id": a.id,
        "name": a.name,
        "subtitle": f"{a.profession} — {a.crimes_committed} crime(s), "
                    f"standing {a.reputation:.0f}",
        "beats": beats,
        "allies": allies,
        "foes": foes,
        "beliefs": beliefs,
        "fate": fate,
        "alive": a.alive,
    }


def life_story_text(sim, agent_id: str) -> str:
    s = life_story(sim, agent_id)
    lines = [f"# {s['name']}", f"*{s['subtitle']}*", ""]
    lines += [f"- {b}" for b in s["beats"]] or ["- (an uneventful life)"]
    lines.append("")
    if s["allies"]:
        lines.append(f"**Allies:** {', '.join(s['allies'])}")
    if s["foes"]:
        lines.append(f"**Foes:** {', '.join(s['foes'])}")
    if s["beliefs"]:
        lines.append(f"**Marks:** {', '.join(s['beliefs'])}")
    lines.append(f"**Fate:** {s['fate']}")
    return "\n".join(lines)


_NARRATE_SYSTEM = (
    "You are the chronicler of a small simulated town. Given a list of "
    "day-by-day events, write a vivid but FAITHFUL short chronicle — a few "
    "short paragraphs. Use only the events given; invent no new facts, and "
    "keep the names exactly. Aim for the feel of a brief history, not a list."
)


def narrate(chronicle_md: str, client) -> str | None:
    """Turn the deterministic chronicle into flowing prose via an LLM client
    (``client(system, user) -> str``). This is where 'story' meets
    'reproducibility': the narration call goes through the *same* recording
    client as the agents, so a narrated chronicle replays bit-exactly.

    Returns ``None`` when there is no client (heuristic worlds) or the call
    fails — callers then fall back to the curated `chronicle_text`."""
    if client is None:
        return None
    try:
        prose = client(_NARRATE_SYSTEM, chronicle_md)
    except Exception:
        return None
    prose = (prose or "").strip()
    return prose or None
