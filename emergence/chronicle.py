"""Turn a run into a *story* — the heart of the observatory experience.

The engine emits a flat event log; this module curates it into something a
person wants to read: a **chronicle** of the town's days (the legislative spam
summarised, the dramatic beats surfaced) and a **life story** for any citizen
(their role versus what they actually did, their ties, their faith, their fate).

Pure stdlib and deterministic — the same run always tells the same story. Each
renderer takes a ``lang`` ("en"/"ja"); English is the default and is byte-for-byte
unchanged, so existing behaviour and tests are unaffected. A future layer can
hand these beats to an LLM to render flowing prose in either language.
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

# Japanese glosses for engine-generated tokens (professions, facility types,
# death causes). Fall back to the raw token when unmapped.
_PROF_JA = {
    "farmer": "農夫", "builder": "大工", "teacher": "教師", "merchant": "商人",
    "doctor": "医者", "guard": "衛兵", "miner": "鉱夫", "librarian": "司書",
    "smith": "鍛冶", "council clerk": "書記", "child": "子供", "mayor": "市長",
    "priest": "司祭", "banker": "銀行家",
}
_FAC_JA = {
    "farm": "農場", "forest": "森", "mine": "鉱山", "workshop": "工房",
    "market": "市場", "bank": "銀行", "granary": "倉庫", "library": "図書館",
    "plaza": "広場", "house": "家", "town_hall": "役所", "hospital": "病院",
    "police_station": "警察署", "prison": "刑務所", "temple": "寺院",
    "monument": "記念碑",
}
_CAUSE_JA = {
    "violence": "暴力", "killed in violence": "暴力により死亡",
    "starvation": "餓死",
}
# Readable consequence of a law's machine effects (the "so what" of a bill).
_EFFECT = {
    "crime_deterrence": ("tougher policing", "取り締まり強化"),
    "food_redistribution": ("food for the needy", "困窮者への食料分配"),
    "tax": ("a wealth tax", "富裕税"),
    "punishment": ("fines for offenders", "違反者への罰金"),
    # Per-offence norms (#35): named specifically, instead of collapsing
    # into the same "tougher policing" every crime law reads as today.
    "violence_norm": ("a law against violence", "暴力を禁じる法"),
    "theft_norm": ("a law against theft", "窃盗を禁じる法"),
    "arson_norm": ("a law against arson", "放火を禁じる法"),
}


def _effect_name(k, lang):
    en, ja = _EFFECT.get(k, (k, k))
    return ja if lang == "ja" else en


def _L(lang, en, ja):
    return ja if lang == "ja" else en


def _prof(p, lang):
    return _PROF_JA.get(p, p) if lang == "ja" else p


def _cause(c, lang):
    return _CAUSE_JA.get(c, c) if lang == "ja" else (c or "?")


def _facility_type(t, lang):
    t = t or "facility"
    return _FAC_JA.get(t, t) if lang == "ja" else t.replace("_", " ")


def _names(sim) -> dict:
    return {a.id: a.name for a in sim.agents}


def _beat(e: dict, nm, lang: str) -> str | None:
    """Render one headline event as a sentence (or None to skip)."""
    k = e["kind"]
    if k == "death":
        return _L(lang, f"💀 {nm(e['agent'])} died — {e.get('cause','?')}",
                  f"💀 {nm(e['agent'])} が死亡 — {_cause(e.get('cause'), lang)}")
    if k == "gang_formed":
        return _L(lang, f"🩸 {nm(e['leader'])} founded the {e['gang']}",
                  f"🩸 {nm(e['leader'])} が{e['gang']}を結成")
    if k == "religion_founded":
        return _L(lang, f"🛐 {nm(e['prophet'])} founded the faith {e['faith']}",
                  f"🛐 {nm(e['prophet'])} が信仰「{e['faith']}」を創始")
    if k == "conversion":
        return _L(lang, f"🛐 {nm(e['convert'])} joined {e['faith']}",
                  f"🛐 {nm(e['convert'])} が{e['faith']}に入信")
    if k == "rebellion":
        return _L(lang, f"🚩 {nm(e['instigator'])} led a rebellion",
                  f"🚩 {nm(e['instigator'])} が反乱を主導")
    if k == "monument":
        return _L(lang, f"🏛 {nm(e['by'])} raised {e['name']}",
                  f"🏛 {nm(e['by'])} が{e['name']}を建立")
    if k == "public_works":
        return _L(lang, f"🏗 the council built a {_facility_type(e.get('type'), 'en')}",
                  f"🏗 議会が{_facility_type(e.get('type'), 'ja')}を建設")
    if k == "bribe":
        return _L(lang, f"🤝 {nm(e['briber'])} bribed {nm(e['guard'])} (a guard)",
                  f"🤝 {nm(e['briber'])} が衛兵 {nm(e['guard'])} に賄賂を渡した")
    return None


def chronicle(sim, lang: str = "en") -> list[dict]:
    """A curated, day-by-day story: ``[{"day": int, "beats": [str]}]``."""
    nm = (lambda i, m=_names(sim): m.get(i, i))
    by_day: dict[int, list[dict]] = defaultdict(list)
    for e in sim.world.events:
        by_day[e.get("day", 0)].append(e)

    out = []
    announced: set[str] = set()   # a law effect is news the first day it bites
    for day in sorted(by_day):
        evs = by_day[day]
        beats: list[str] = []
        # Headline drama, in order.
        for e in evs:
            line = _beat(e, nm, lang)
            if line:
                beats.append(line)
        # Summarise the noisy machinery rather than listing each item.
        crimes = Counter(e["kind"] for e in evs if e["kind"] in ("violence", "theft", "arson"))
        if crimes:
            _CR = {"violence": ("violence", "暴力"), "theft": ("theft", "窃盗"),
                   "arson": ("arson", "放火")}
            bits = ", ".join(f"{n} {_L(lang, *_CR[k])}" for k, n in crimes.items())
            beats.append(_L(lang, f"⚔ unrest: {bits}", f"⚔ 騒乱: {bits}"))
        passed = sum(1 for e in evs if e["kind"] == "proposal_resolved" and e.get("status") == "passed")
        laws = [e["effects"] for e in evs if e["kind"] == "law_enacted" and e.get("effects")]
        # Only surface legislation that *did* something — a law with real effects,
        # phrased as its consequence. Bare "N bills passed" with no teeth is
        # procedural noise, so it is dropped (it answered "so what?" with nothing).
        eff_keys = sorted({x for eff in laws for x in eff.split(", ")} - announced)
        if eff_keys:
            announced.update(eff_keys)
            effects = ", ".join(_effect_name(k, lang) for k in eff_keys)
            beats.append(_L(lang, f"🏛 a law took effect — {effects}",
                            f"🏛 法が発効 — {effects}"))
        births = sum(1 for e in evs if e["kind"] == "birth")
        if births:
            beats.append(_L(lang, f"👶 {births} child(ren) born", f"👶 {births}人の子が誕生"))
        if beats:
            out.append({"day": day, "beats": beats})
    return out


def chronicle_text(sim, title: str | None = None, lang: str = "en") -> str:
    if title is None:
        title = _L(lang, "Town Chronicle", "町の年代記")
    lines = [f"# {title}", ""]
    for entry in chronicle(sim, lang):
        lines.append(_L(lang, f"**Day {entry['day']}**", f"**{entry['day']}日目**"))
        lines += [f"- {b}" for b in entry["beats"]]
        lines.append("")
    return "\n".join(lines).strip()


def life_story(sim, agent_id: str, lang: str = "en") -> dict:
    """One citizen's life: their role vs their deeds, ties, beliefs, and fate."""
    a = sim._by_id.get(agent_id)
    if a is None:
        raise KeyError(agent_id)
    nm = (lambda i, m=_names(sim): m.get(i, i))
    dd = (lambda d: _L(lang, f"D{d}", f"{d}日目"))

    # The acts of this life, in order — narrated from the citizen's vantage.
    beats: list[str] = []
    for e in sim.world.events:
        if not any(e.get(f) == agent_id for f in _ACTOR_FIELDS):
            continue
        k, d = e["kind"], e.get("day", 0)
        if k == "gang_formed":
            beats.append(_L(lang, f"{dd(d)}: founded the {e['gang']}",
                            f"{dd(d)}: {e['gang']}を結成した"))
        elif k == "religion_founded":
            beats.append(_L(lang, f"{dd(d)}: founded the faith {e['faith']} — a prophet now",
                            f"{dd(d)}: 信仰「{e['faith']}」を創始 — 預言者となった"))
        elif k == "conversion" and e.get("convert") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: converted to {e['faith']}",
                            f"{dd(d)}: {e['faith']}に改宗した"))
        elif k == "violence" and e.get("offender") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: attacked {nm(e['victim'])}",
                            f"{dd(d)}: {nm(e['victim'])}を襲った"))
        elif k == "violence" and e.get("victim") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: was attacked by {nm(e['offender'])}",
                            f"{dd(d)}: {nm(e['offender'])}に襲われた"))
        elif k == "theft" and e.get("offender") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: stole from {nm(e['victim'])}",
                            f"{dd(d)}: {nm(e['victim'])}から盗んだ"))
        elif k == "arson" and e.get("offender") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: set fire to {e.get('facility','a building')}",
                            f"{dd(d)}: {e.get('facility','建物')}に放火した"))
        elif k == "arrest" and e.get("offender") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: was arrested by {nm(e['guard'])}",
                            f"{dd(d)}: {nm(e['guard'])}に逮捕された"))
        elif k == "arrest" and e.get("guard") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: made an arrest — kept the peace",
                            f"{dd(d)}: 逮捕を行い、治安を守った"))
        elif k == "monument" and e.get("by") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: raised the monument {e['name']}",
                            f"{dd(d)}: 記念碑「{e['name']}」を建立した"))
        elif k == "rebellion" and e.get("instigator") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: led a rebellion against those in power",
                            f"{dd(d)}: 権力者に対し反乱を主導した"))
        elif k == "death" and e.get("agent") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: died — {e.get('cause','?')}",
                            f"{dd(d)}: 死亡 — {_cause(e.get('cause'), lang)}"))
        elif k == "birth" and e.get("child") == agent_id:
            beats.append(_L(lang, f"{dd(d)}: was born", f"{dd(d)}: 生まれた"))

    ties = sorted(a.trust.items(), key=lambda kv: -kv[1])
    allies = [f"{nm(i)} ({t:+.1f})" for i, t in ties if t > 0.3][:3]
    foes = [f"{nm(i)} ({t:+.1f})" for i, t in ties if t < -0.3][:3]
    beliefs = []
    if a.faith:
        beliefs.append(_L(lang, "devout", "敬虔"))
    if a.gang_id:
        beliefs.append(_L(lang, "gang-sworn", "ギャングの一員"))
    if a.weapons:
        beliefs.append(_L(lang, f"armed ({a.weapons})", f"武装（{a.weapons}）"))

    if a.alive:
        fate = _L(lang, f"Survived to the end, standing {a.reputation:.0f}.",
                  f"最後まで生き延びた。名声 {a.reputation:.0f}。")
    else:
        fate = _L(lang, f"Died on day {a.day_of_death} — {a.cause_of_death}.",
                  f"{a.day_of_death}日目に死亡 — {_cause(a.cause_of_death, lang)}。")

    subtitle = _L(
        lang,
        f"{a.profession} — {a.crimes_committed} crime(s), standing {a.reputation:.0f}",
        f"{_prof(a.profession, lang)} — 犯罪{a.crimes_committed}件・名声{a.reputation:.0f}",
    )

    return {
        "id": a.id,
        "name": a.name,
        "subtitle": subtitle,
        "beats": beats,
        "allies": allies,
        "foes": foes,
        "beliefs": beliefs,
        "fate": fate,
        "alive": a.alive,
    }


def life_story_text(sim, agent_id: str, lang: str = "en") -> str:
    s = life_story(sim, agent_id, lang)
    lines = [f"# {s['name']}", f"*{s['subtitle']}*", ""]
    lines += [f"- {b}" for b in s["beats"]] or [_L(lang, "- (an uneventful life)", "- （平穏な一生）")]
    lines.append("")
    if s["allies"]:
        lines.append(_L(lang, f"**Allies:** {', '.join(s['allies'])}",
                        f"**盟友:** {', '.join(s['allies'])}"))
    if s["foes"]:
        lines.append(_L(lang, f"**Foes:** {', '.join(s['foes'])}",
                        f"**敵:** {', '.join(s['foes'])}"))
    if s["beliefs"]:
        lines.append(_L(lang, f"**Marks:** {', '.join(s['beliefs'])}",
                        f"**刻まれたもの:** {', '.join(s['beliefs'])}"))
    lines.append(_L(lang, f"**Fate:** {s['fate']}", f"**結末:** {s['fate']}"))
    return "\n".join(lines)


_NARRATE_SYSTEM = {
    "en": (
        "You are the chronicler of a small simulated town. Given a list of "
        "day-by-day events, write a vivid but FAITHFUL short chronicle — a few "
        "short paragraphs. Use only the events given; invent no new facts, and "
        "keep the names exactly. Aim for the feel of a brief history, not a list."
    ),
    "ja": (
        "あなたは小さなシミュレーション都市の年代記編者です。日ごとの出来事の一覧を基に、"
        "鮮やかでありながら**忠実**な短い年代記を、数段落の日本語で書いてください。与えられた"
        "出来事だけを使い、新たな事実を創作せず、名前は正確に保つこと。箇条書きではなく、"
        "短い歴史叙述の趣で。"
    ),
}


def narrate(chronicle_md: str, client, lang: str = "en") -> str | None:
    """Turn the deterministic chronicle into flowing prose via an LLM client
    (``client(system, user) -> str``). This is where 'story' meets
    'reproducibility': the narration call goes through the *same* recording
    client as the agents, so a narrated chronicle replays bit-exactly.

    Returns ``None`` when there is no client (heuristic worlds) or the call
    fails — callers then fall back to the curated `chronicle_text`."""
    if client is None:
        return None
    system = _NARRATE_SYSTEM.get(lang, _NARRATE_SYSTEM["en"])
    try:
        prose = client(system, chronicle_md)
    except Exception:
        return None
    prose = (prose or "").strip()
    return prose or None
