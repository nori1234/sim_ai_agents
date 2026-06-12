"""Render a finished simulation as a self-contained HTML page.

No third-party dependencies and no external assets: the output is a single
HTML file with inline CSS and hand-built SVG, so it opens in any browser
straight off disk. The views adapt to whichever needs layers a run enabled:

* metric cards (layer-aware) + one-line verdict
* a daily timeline (population, cumulative crime, fraud, and births)
* Maslow's needs pyramid — each tier brightened by its average satisfaction
* the town map with a spatial crime heatmap overlaid on the facilities
* the trust network between agents at the end of the run
* an honour/reputation ranking (with --status)
* a lineage / family tree (when children were born)
* a layer-aware citizens table
"""

from __future__ import annotations

import html
import math
from collections import Counter

from .report import one_line_verdict
from .simulation import Simulation
from .world import FacilityType

# Colour grouping for facilities on the map.
_FACILITY_GROUP = {
    FacilityType.FARM: ("food", "#2e7d32"),
    FacilityType.GRANARY: ("food", "#558b2f"),
    FacilityType.FOREST: ("material", "#6d4c41"),
    FacilityType.MINE: ("material", "#8d6e63"),
    FacilityType.WORKSHOP: ("work", "#1565c0"),
    FacilityType.MARKET: ("work", "#1976d2"),
    FacilityType.BANK: ("civic", "#7b1fa2"),
    FacilityType.TOWN_HALL: ("civic", "#5e35b1"),
    FacilityType.LIBRARY: ("civic", "#3949ab"),
    FacilityType.POLICE_STATION: ("civic", "#283593"),
    FacilityType.HOSPITAL: ("civic", "#00838f"),
    FacilityType.HOUSE: ("civic", "#90a4ae"),
    FacilityType.PLAZA: ("civic", "#9e9d24"),
    FacilityType.MONUMENT: ("civic", "#fbc02d"),
}


def _esc(s) -> str:
    return html.escape(str(s))


def _layers(sim: Simulation) -> dict:
    """Which optional needs layers were active in this run."""
    return {
        "drives": getattr(sim.drives, "enabled", False),
        "reproduction": getattr(sim.drives, "reproduction", False),
        "status": getattr(sim.status, "enabled", False),
        "psyche": getattr(sim.psyche, "enabled", False),
    }


# ----------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------
def _cards(sim: Simulation) -> str:
    m = sim.metrics
    L = _layers(sim)
    growth = f"+{m.births} born" if m.births else f"{m.survival_rate:.0%} survived"
    cards = [
        ("Survivors", f"{m.survivors}", f"from {m.population} — {growth}"),
        ("Crimes", f"{m.crimes_total}", "theft / violence / arson"),
        ("Pass rate", f"{m.pass_rate:.0%}", f"{m.proposals_passed}/{m.proposals_total} bills"),
    ]
    # Surface whichever advanced layers were switched on.
    if L["reproduction"]:
        cards.append(("Births", f"{m.births}", f"{m.matings} matings"))
    if L["drives"]:
        cards.append(("Pleasure", f"{m.total_pleasure:.0f}", "wellbeing banked"))
    if L["status"]:
        cards.append(("Praise", f"{m.total_praise}", "recognition exchanged"))
    if L["psyche"]:
        cards.append(("Works", f"{m.works_created}", "self-actualization"))
        cards.append(("Peak fear", f"{m.peak_fear:.0f}", "out of 100"))
    # Pad with the classic stats so the run always reads richly.
    for extra in (("Fraud", f"{m.frauds}", '"I\'m broke" scams'),
                  ("Collaboration", f"{m.collaborations}", f"{m.monuments_built} monuments"),
                  ("Days run", f"{m.days_run}", "of scheduled run")):
        if len(cards) % 3 == 0:
            break
        cards.append(extra)

    html_cards = "".join(
        f'<div class="card"><div class="card-val">{_esc(v)}</div>'
        f'<div class="card-key">{_esc(k)}</div>'
        f'<div class="card-sub">{_esc(s)}</div></div>'
        for k, v, s in cards
    )
    return f'<div class="cards">{html_cards}</div>'


def _timeline(sim: Simulation) -> str:
    log = sim.daily_log
    if not log:
        return ""
    W, H = 760, 280
    pad_l, pad_r, pad_t, pad_b = 44, 44, 20, 30
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    days = [d["day"] for d in log]
    alive = [d["alive"] for d in log]
    crimes = [d["crimes_total"] for d in log]
    frauds = [d["frauds"] for d in log]
    births = [d.get("births", 0) for d in log]
    show_births = _layers(sim)["reproduction"] and max(births) > 0
    n = len(days)
    # When the population can grow past its start, scale the left axis to it.
    pop = max(sim.metrics.population, max(alive), 1)
    max_crime = max(max(crimes), max(births) if show_births else 0, 1)

    def x(i: int) -> float:
        return pad_l + (plot_w * (i / (n - 1)) if n > 1 else plot_w / 2)

    def y_left(v: float) -> float:  # population scale
        return pad_t + plot_h * (1 - v / pop)

    def y_right(v: float) -> float:  # crime/fraud scale
        return pad_t + plot_h * (1 - v / max_crime)

    def poly(values, yfn, color) -> str:
        pts = " ".join(f"{x(i):.1f},{yfn(v):.1f}" for i, v in enumerate(values))
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" />'
        )

    # Axis gridlines / labels.
    grid = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + plot_h * frac
        grid.append(f'<line x1="{pad_l}" y1="{gy:.0f}" x2="{pad_l+plot_w}" '
                    f'y2="{gy:.0f}" stroke="#eceff1" />')
        grid.append(f'<text x="{pad_l-6}" y="{gy+3:.0f}" class="ax" '
                    f'text-anchor="end">{round(pop*(1-frac))}</text>')
        grid.append(f'<text x="{pad_l+plot_w+6}" y="{gy+3:.0f}" class="ax" '
                    f'text-anchor="start">{round(max_crime*(1-frac))}</text>')
    xticks = []
    for i, d in enumerate(days):
        if n <= 16 or i % 2 == 0 or i == n - 1:
            xticks.append(f'<text x="{x(i):.0f}" y="{H-pad_b+16:.0f}" class="ax" '
                          f'text-anchor="middle">{d}</text>')

    return f"""
    <div class="panel">
      <h2>Daily timeline</h2>
      <svg viewBox="0 0 {W} {H}" class="chart" role="img">
        {''.join(grid)}
        {poly(alive, y_left, '#2e7d32')}
        {poly(crimes, y_right, '#c62828')}
        {poly(frauds, y_right, '#ef6c00')}
        {poly(births, y_right, '#8e24aa') if show_births else ''}
        {''.join(xticks)}
        <text x="{pad_l}" y="14" class="ax">population (left)</text>
        <text x="{pad_l+plot_w}" y="14" class="ax" text-anchor="end">crime / fraud (right)</text>
      </svg>
      <div class="legend">
        <span><i style="background:#2e7d32"></i>alive</span>
        <span><i style="background:#c62828"></i>cumulative crime</span>
        <span><i style="background:#ef6c00"></i>cumulative fraud</span>
        {'<span><i style="background:#8e24aa"></i>cumulative births</span>' if show_births else ''}
      </div>
    </div>"""


def _town_map(sim: Simulation) -> str:
    world = sim.world
    cell = 22
    W, H = world.width * cell, world.height * cell

    # Aggregate crime counts per cell from the event log.
    crime_kinds = {"theft", "violence", "arson"}
    heat: Counter = Counter()
    for e in world.events:
        if e.get("kind") in crime_kinds and isinstance(e.get("pos"), (tuple, list)):
            px, py = e["pos"]
            heat[(int(px), int(py))] += 1
    max_heat = max(heat.values(), default=0)

    # Light grid.
    grid = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fafafa" '
            f'stroke="#e0e0e0" />']

    # Heat blobs under the facilities.
    blobs = []
    for (px, py), count in heat.items():
        cx, cy = px * cell + cell / 2, py * cell + cell / 2
        r = cell * (0.6 + 1.6 * (count / max_heat)) if max_heat else cell * 0.6
        op = 0.15 + 0.55 * (count / max_heat) if max_heat else 0.2
        blobs.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" '
                     f'fill="#e53935" opacity="{op:.2f}" />')

    # Facilities.
    dots = []
    for f in world.facilities:
        _, color = _FACILITY_GROUP.get(f.ftype, ("civic", "#90a4ae"))
        cx, cy = f.x * cell + cell / 2, f.y * cell + cell / 2
        dots.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="4.5" fill="{color}">'
                    f'<title>{_esc(f.name)} ({_esc(f.ftype.value)})</title></circle>')

    legend = "".join(
        f'<span><i style="background:{c}"></i>{g}</span>'
        for g, c in [("food", "#2e7d32"), ("material", "#6d4c41"),
                     ("work", "#1565c0"), ("civic", "#5e35b1"),
                     ("crime", "#e53935")]
    )
    note = (f"hottest cell: {max_heat} crimes" if max_heat
            else "no crime recorded")
    return f"""
    <div class="panel">
      <h2>Town map &amp; crime heatmap</h2>
      <svg viewBox="0 0 {W} {H}" class="map" role="img">
        {''.join(grid)}
        {''.join(blobs)}
        {''.join(dots)}
      </svg>
      <div class="legend">{legend}<span class="muted">{note}</span></div>
    </div>"""


def _trust_network(sim: Simulation) -> str:
    agents = sim.agents
    n = len(agents)
    if n == 0:
        return ""
    size = 520
    cx, cy, R = size / 2, size / 2, size / 2 - 60
    pos = {}
    for i, a in enumerate(agents):
        ang = 2 * math.pi * i / n - math.pi / 2
        pos[a.id] = (cx + R * math.cos(ang), cy + R * math.sin(ang))

    edges = []
    for a in agents:
        ax, ay = pos[a.id]
        for other_id, val in a.trust.items():
            if other_id not in pos or abs(val) < 0.25:
                continue
            bx, by = pos[other_id]
            color = "#2e7d32" if val > 0 else "#c62828"
            width = 0.8 + 2.6 * min(1.0, abs(val))
            op = 0.25 + 0.5 * min(1.0, abs(val))
            edges.append(f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" '
                         f'y2="{by:.0f}" stroke="{color}" stroke-width="{width:.1f}" '
                         f'opacity="{op:.2f}" />')

    nodes = []
    for a in agents:
        px, py = pos[a.id]
        fill = "#1565c0" if a.alive else "#b0bec5"
        ring = "#0d47a1" if a.alive else "#78909c"
        nodes.append(
            f'<circle cx="{px:.0f}" cy="{py:.0f}" r="9" fill="{fill}" '
            f'stroke="{ring}" stroke-width="2"><title>{_esc(a.name)} '
            f'({_esc(a.persona)}) — '
            f'{"alive" if a.alive else "dead"}</title></circle>'
            f'<text x="{px:.0f}" y="{py-14:.0f}" class="nodelbl" '
            f'text-anchor="middle">{_esc(a.name)}</text>'
        )

    return f"""
    <div class="panel">
      <h2>Trust network (end of run)</h2>
      <svg viewBox="0 0 {size} {size}" class="net" role="img">
        {''.join(edges)}
        {''.join(nodes)}
      </svg>
      <div class="legend">
        <span><i style="background:#2e7d32"></i>trusts</span>
        <span><i style="background:#c62828"></i>distrusts</span>
        <span><i style="background:#b0bec5"></i>deceased</span>
      </div>
    </div>"""


def _needs_pyramid(sim: Simulation) -> str:
    """Maslow's pyramid, each tier brightened by how satisfied that need is."""
    L = _layers(sim)
    if not (L["drives"] or L["status"] or L["psyche"]):
        return ""  # nothing to show on a bare run
    living = [a for a in sim.agents if a.alive] or sim.agents
    n = len(living) or 1

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    def avg(fn) -> float:
        return sum(fn(a) for a in living) / n

    def best_trust(a) -> float:
        vals = [t for t in a.trust.values()]
        return max(vals) if vals else 0.0

    # (label, satisfaction 0..1, active?, colour)
    tiers = [
        ("Physiological  食欲・睡眠・性欲",
         avg(lambda a: 1 - clamp((a.hunger + a.fatigue) / 200)), L["drives"], "#2e7d32"),
        ("Safety  安全（恐怖からの自由）",
         avg(lambda a: 1 - clamp(a.fear / 100)), L["psyche"], "#1565c0"),
        ("Belonging  社会・つがい・信頼",
         avg(lambda a: clamp(best_trust(a))), True, "#00838f"),
        ("Esteem  承認・名誉・権力",
         avg(lambda a: 1 - clamp(a.esteem / 100)), L["status"], "#8e24aa"),
        ("Self-actualization  自己実現・創造",
         avg(lambda a: clamp(a.fulfillment / 20)), L["psyche"], "#fbc02d"),
    ]

    W, H = 620, 320
    band_h = H / len(tiers)
    base_half = W / 2 * 0.92
    cx = W / 2
    bands, labels = [], []
    for i, (label, sat, active, color) in enumerate(tiers):
        # Bottom tier (i=0) is the widest; apex is the top.
        level = len(tiers) - 1 - i  # 0 at top, 4 at bottom for geometry
        y_bot = H - i * band_h
        y_top = H - (i + 1) * band_h
        hw_bot = base_half * (1 - (i) / len(tiers))
        hw_top = base_half * (1 - (i + 1) / len(tiers))
        poly = (f"{cx - hw_bot:.0f},{y_bot:.0f} {cx + hw_bot:.0f},{y_bot:.0f} "
                f"{cx + hw_top:.0f},{y_top:.0f} {cx - hw_top:.0f},{y_top:.0f}")
        if active:
            op = 0.18 + 0.72 * clamp(sat)
            fill = color
            pct = f"{sat * 100:.0f}%"
        else:
            op = 0.10
            fill = "#90a4ae"
            pct = "—"
        bands.append(f'<polygon points="{poly}" fill="{fill}" opacity="{op:.2f}" '
                     f'stroke="#fff" stroke-width="1.5" />')
        ty = (y_bot + y_top) / 2
        labels.append(
            f'<text x="{cx:.0f}" y="{ty - 2:.0f}" class="pyr" text-anchor="middle">'
            f'{_esc(label)}</text>'
            f'<text x="{cx:.0f}" y="{ty + 12:.0f}" class="pyrpct" text-anchor="middle">'
            f'{pct}</text>'
        )

    return f"""
    <div class="panel">
      <h2>Needs pyramid — average satisfaction</h2>
      <svg viewBox="0 0 {W} {H}" class="pyramid" role="img">
        {''.join(bands)}
        {''.join(labels)}
      </svg>
      <div class="legend"><span class="muted">brighter band = that need is
      better met across the living; grey = layer not enabled this run</span></div>
    </div>"""


def _reputation_ranking(sim: Simulation) -> str:
    """Horizontal bars of standing/honour — who the town admires."""
    if not _layers(sim)["status"]:
        return ""
    ranked = sorted(sim.agents, key=lambda a: a.reputation, reverse=True)
    ranked = [a for a in ranked if a.reputation > 0][:12]
    if not ranked:
        return ""
    top = ranked[0].reputation
    rows = []
    bar_w = 320
    for a in ranked:
        frac = a.reputation / top if top else 0
        w = max(2, bar_w * frac)
        fill = "#8e24aa" if a.alive else "#b39ddb"
        crown = " ♛" if a.times_mayor else ""
        rows.append(
            f'<div class="rep-row">'
            f'<span class="rep-name">{_esc(a.name)}{crown}</span>'
            f'<span class="rep-bar"><i style="width:{w:.0f}px;background:{fill}"></i></span>'
            f'<span class="rep-val">{a.reputation:.0f}</span>'
            f'</div>'
        )
    return f"""
    <div class="panel">
      <h2>Honour &amp; power — reputation ranking</h2>
      {''.join(rows)}
      <div class="legend"><span>♛ = held the mayoralty</span>
      <span class="muted">prestige from monuments, passed laws, office, praise</span></div>
    </div>"""


def _lineage(sim: Simulation) -> str:
    """A genealogy: founders and the children born during the run."""
    if sim.metrics.births <= 0:
        return ""
    agents = sim.agents
    by_id = {a.id: a for a in agents}
    # Generation = longest parent chain to a founder.
    gen: dict[str, int] = {}

    def resolve(a) -> int:
        if a.id in gen:
            return gen[a.id]
        if not a.parent_ids:
            gen[a.id] = 0
            return 0
        parents = [by_id[p] for p in a.parent_ids if p in by_id]
        g = 1 + max((resolve(p) for p in parents), default=0)
        gen[a.id] = g
        return g

    for a in agents:
        resolve(a)
    max_gen = max(gen.values(), default=0)
    if max_gen == 0:
        return ""

    # Lay out by generation column.
    cols: dict[int, list] = {}
    for a in agents:
        cols.setdefault(gen[a.id], []).append(a)
    col_w = 150
    row_h = 34
    W = (max_gen + 1) * col_w + 20
    H = max(len(v) for v in cols.values()) * row_h + 30
    pos = {}
    for g, members in cols.items():
        members.sort(key=lambda a: a.id)
        for j, a in enumerate(members):
            x = 20 + g * col_w
            y = 24 + j * row_h
            pos[a.id] = (x, y)

    edges, nodes = [], []
    for a in agents:
        if not a.parent_ids:
            continue
        cxv, cyv = pos[a.id]
        for p in a.parent_ids:
            if p in pos:
                px, py = pos[p]
                edges.append(f'<path d="M{px+70},{py} C{px+110},{py} {cxv-40},{cyv} '
                             f'{cxv},{cyv}" fill="none" stroke="#ce93d8" '
                             f'stroke-width="1.4" opacity="0.8" />')
    for a in agents:
        x, y = pos[a.id]
        founder = not a.parent_ids
        fill = "#5e35b1" if founder else ("#26a69a" if a.alive else "#cfd8dc")
        nodes.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" fill="{fill}">'
            f'<title>{_esc(a.name)} ({_esc(a.persona)})'
            f'{" — founder" if founder else ""}{"" if a.alive else " — deceased"}'
            f'</title></circle>'
            f'<text x="{x+10:.0f}" y="{y+4:.0f}" class="treelbl">{_esc(a.name)}</text>'
        )

    return f"""
    <div class="panel">
      <h2>Lineage — {sim.metrics.births} born across {max_gen} generation(s)</h2>
      <div class="scroll">
        <svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="tree" role="img">
          {''.join(edges)}
          {''.join(nodes)}
        </svg>
      </div>
      <div class="legend">
        <span><i style="background:#5e35b1"></i>founder</span>
        <span><i style="background:#26a69a"></i>born &amp; living</span>
        <span><i style="background:#cfd8dc"></i>born, deceased</span>
      </div>
    </div>"""


def _citizens(sim: Simulation) -> str:
    L = _layers(sim)
    # Build columns dynamically so advanced layers surface their own stats.
    cols = [("Name", lambda a: _esc(a.name)),
            ("Persona", lambda a: _esc(a.persona)),
            ("Status", lambda a: ("alive" if a.alive
                                  else f"died d{a.day_of_death} ({_esc(a.cause_of_death)})")),
            ("Crimes", lambda a: a.crimes_committed),
            ("Votes", lambda a: a.votes_cast)]
    if L["reproduction"]:
        cols.append(("Children", lambda a: a.children))
    if L["status"]:
        cols.append(("Honour", lambda a: f"{a.reputation:.0f}"))
        cols.append(("Praised", lambda a: a.praise_received))
    if L["psyche"]:
        cols.append(("Works", lambda a: a.works_created))
    if L["drives"]:
        cols.append(("Pleasure", lambda a: f"{a.pleasure:.0f}"))

    head = "".join(f"<th>{_esc(name)}</th>" for name, _ in cols)
    rows = []
    for a in sim.agents:
        cls = "" if a.alive else ' class="dead"'
        cells = "".join(f"<td>{fn(a)}</td>" for _, fn in cols)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"""
    <div class="panel">
      <h2>Citizens</h2>
      <div class="scroll">
        <table class="tbl">
          <tr>{head}</tr>
          {''.join(rows)}
        </table>
      </div>
    </div>"""


_CSS = """
:root { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
body { margin: 0; background: #f4f6f8; color: #263238; }
.wrap { max-width: 880px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.verdict { font-size: 15px; color: #455a64; margin: 0 0 20px; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 22px; }
.card { background: #fff; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card-val { font-size: 26px; font-weight: 700; }
.card-key { font-size: 13px; font-weight: 600; color: #37474f; margin-top: 2px; }
.card-sub { font-size: 11px; color: #90a4ae; }
.panel { background: #fff; border-radius: 10px; padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.panel h2 { font-size: 16px; margin: 0 0 12px; }
.chart, .map, .net { width: 100%; height: auto; }
.ax { font-size: 10px; fill: #78909c; }
.nodelbl { font-size: 10px; fill: #37474f; }
.legend { margin-top: 10px; font-size: 12px; color: #546e7a; display: flex; flex-wrap: wrap; gap: 14px; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.legend .muted { color: #b0bec5; }
.tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
.tbl th, .tbl td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eceff1; }
.tbl th { color: #607d8b; font-weight: 600; }
.tbl tr.dead td { color: #b0bec5; }
.scroll { overflow-x: auto; }
.pyramid { width: 100%; height: auto; }
.pyr { font-size: 11px; fill: #fff; font-weight: 600; }
.pyrpct { font-size: 11px; fill: #fff; font-weight: 700; }
.treelbl { font-size: 9px; fill: #455a64; }
.rep-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; font-size: 13px; }
.rep-name { width: 110px; flex: none; color: #37474f; }
.rep-bar { flex: 1; background: #f3e5f5; border-radius: 4px; height: 14px; }
.rep-bar i { display: block; height: 14px; border-radius: 4px; }
.rep-val { width: 40px; flex: none; text-align: right; color: #6a1b9a; font-weight: 600; }
.footer { font-size: 12px; color: #b0bec5; text-align: center; margin-top: 24px; }
"""


def render_html(sim: Simulation, title: str = "Emergence World") -> str:
    """Return a complete, self-contained HTML document for ``sim``."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>{_esc(title)}</h1>
  <p class="verdict">{_esc(one_line_verdict(sim))}</p>
  {_cards(sim)}
  {_timeline(sim)}
  {_needs_pyramid(sim)}
  <div class="panel" style="padding:0;background:none;box-shadow:none">
    {_town_map(sim)}
    {_trust_network(sim)}
  </div>
  {_reputation_ranking(sim)}
  {_lineage(sim)}
  {_citizens(sim)}
  <div class="footer">Generated by Emergence World — emergence.viz</div>
</div></body></html>"""


def write_html(sim: Simulation, path: str, title: str = "Emergence World") -> str:
    """Render ``sim`` and write it to ``path``; return ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(sim, title))
    return path
