"""Export a *standalone* observatory demo: a single HTML file that plays back a
real run with no server and no dependencies — open it in any browser.

It bakes a day-by-day timeline (citizen positions, inner lives, and the event
story) into the page, so it doubles as a shareable demo for showing the product
direction. The live experience is `python -m emergence.server`; this is the
offline twin for previews.

    python examples/export_demo.py [persona] [seed] [days] [out.html]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.api import EmergenceAPI

_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Emergence World — Observatory (demo)</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#272e38;--ink:#e6edf3;--dim:#8b98a5;}
 *{box-sizing:border-box;} body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink);}
 header{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;padding:.6rem .9rem;background:var(--panel);border-bottom:1px solid var(--line);}
 header h1{font-size:15px;margin:0 .6rem 0 0;font-weight:700;}
 button,input[type=range]{font:inherit;color:var(--ink);background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:.3rem .5rem;cursor:pointer;}
 #verdict{margin-left:auto;color:var(--dim);}
 main{display:grid;grid-template-columns:1fr 360px;gap:1px;background:var(--line);height:calc(100vh - 92px);}
 #stage{background:var(--bg);position:relative;overflow:hidden;} #town{display:block;width:100%;height:100%;}
 aside{background:var(--panel);display:flex;flex-direction:column;min-height:0;}
 .tabs{display:flex;border-bottom:1px solid var(--line);} .tabs button{flex:1;border:0;border-radius:0;background:transparent;color:var(--dim);padding:.5rem;border-bottom:2px solid transparent;}
 .tabs button.active{color:var(--ink);border-bottom-color:#1f6feb;}
 .scroll{overflow:auto;padding:.6rem .8rem;flex:1;min-height:0;} .feed div{padding:.15rem 0;border-bottom:1px solid #1d232c;}
 .feed .d{color:var(--dim);margin-right:.4rem;} .hidden{display:none;}
 .bar{height:8px;background:#0d1117;border-radius:5px;overflow:hidden;margin:.1rem 0 .5rem;} .bar>i{display:block;height:100%;}
 h3{margin:.4rem 0 .2rem;font-size:13px;} .muted{color:var(--dim);}
 .rel{display:flex;justify-content:space-between;border-bottom:1px solid #1d232c;padding:.1rem 0;}
 .timeline{display:flex;gap:.6rem;align-items:center;padding:.4rem .9rem;background:var(--panel);border-top:1px solid var(--line);}
 .timeline input{flex:1;} .legend{display:flex;gap:.8rem;font-size:12px;color:var(--dim);padding:.3rem .9rem;background:var(--panel);}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.25rem;vertical-align:middle;}
</style></head><body>
<header><h1>🌍 Emergence Observatory <span class="muted" style="font-weight:400">demo</span></h1>
 <span id="meta" class="muted"></span><span id="verdict"></span></header>
<main><div id="stage"><canvas id="town"></canvas></div>
 <aside><div class="tabs"><button data-tab="feed" class="active">Story</button><button data-tab="possess">Citizen</button></div>
  <div id="feed" class="scroll feed"></div>
  <div id="possess" class="scroll hidden"><p class="muted">Click a citizen in the town to possess them.</p></div>
 </aside></main>
<div class="timeline"><button id="play">▶ Play</button><span id="day" class="muted"></span>
 <input id="slider" type="range" min="0" value="0" /></div>
<div class="legend">
 <span><i class="dot" style="background:#4aa3ff"></i>Guardian</span>
 <span><i class="dot" style="background:#f0b429"></i>Philosopher</span>
 <span><i class="dot" style="background:#3fb950"></i>Idealist</span>
 <span><i class="dot" style="background:#f85149"></i>Predator</span>
 <span id="popline"></span></div>
<script>
const DATA = __DATA__;
const COLOR={guardian:"#4aa3ff",philosopher:"#f0b429",idealist:"#3fb950",predator:"#f85149"};
const $=s=>document.querySelector(s);
let i=0, selected=null, playing=false, timer=null;
const frames=DATA.frames, N=frames.length;
$("#slider").max=N-1;
$("#meta").textContent=`${DATA.persona} · seed ${DATA.seed} · ${DATA.width}×${DATA.height}`;
function esc(s){const d=document.createElement("div");d.textContent=s==null?"":String(s);return d.innerHTML;}
function nameMap(f){const m={};for(const a of f.agents)m[a.id]=a.name;return m;}
function story(e,n){const k=e.kind;
 if(k==="speech")return `🗣 <b>${n(e.agent)}</b>: “${esc(e.text)}”`;
 if(k==="violence")return `⚔ <b>${n(e.offender)}</b> attacked <b>${n(e.victim)}</b>`;
 if(k==="theft")return `🥷 <b>${n(e.offender)}</b> stole from <b>${n(e.victim)}</b>`;
 if(k==="arson")return `🔥 <b>${n(e.offender)}</b> set fire to ${esc(e.facility)}`;
 if(k==="arrest")return `👮 <b>${n(e.guard)}</b> arrested <b>${n(e.offender)}</b>`;
 if(k==="transfer")return `🎁 <b>${n(e.sender)}</b> gave ${esc(e.amount)} ${esc(e.resource)} to <b>${n(e.receiver)}</b>`;
 if(k==="praise")return `👏 <b>${n(e.by)}</b> praised <b>${n(e.of)}</b>`;
 if(k==="proposal")return `📜 <b>${n(e.author)}</b> proposed: “${esc(e.text)}”`;
 if(k==="proposal_resolved")return `🗳 proposal #${esc(e.id)} ${esc(e.status)} (${esc(e.yes)}–${esc(e.no)})`;
 if(k==="law_enacted")return `⚖ a law took effect: ${esc(e.effects)}`;
 if(k==="gang_formed")return `🩸 <b>${n(e.leader)}</b> founded the ${esc(e.gang)}`;
 if(k==="religion_founded")return `🛐 <b>${n(e.prophet)}</b> founded ${esc(e.faith)}`;
 if(k==="conversion")return `🛐 <b>${n(e.convert)}</b> joined ${esc(e.faith)}`;
 if(k==="monument")return `🏛 <b>${n(e.by)}</b> raised ${esc(e.name)}`;
 if(k==="death")return `💀 <b>${n(e.agent)}</b> died — ${esc(e.cause)}`;
 if(k==="rebellion")return `🚩 <b>${n(e.instigator)}</b> led a rebellion`;
 return `· ${esc(k)}`;}
const canvas=$("#town"),ctx=canvas.getContext("2d");
function fit(){const r=$("#stage").getBoundingClientRect();canvas.width=r.width;canvas.height=r.height;draw();}
function draw(){const f=frames[i];if(!f)return;const W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);
 const cw=W/DATA.width,ch=H/DATA.height;ctx.font="10px system-ui";ctx.textAlign="center";
 for(const fc of f.facilities){const x=(fc.x+0.5)*cw,y=(fc.y+0.5)*ch;ctx.fillStyle="#222b36";ctx.fillRect(x-cw*0.45,y-ch*0.45,cw*0.9,ch*0.9);ctx.fillStyle="#5b6775";ctx.fillText(fc.type.replace(/_/g," ").slice(0,10),x,y+3);}
 for(const a of f.agents){const x=(a.x+0.5)*cw,y=(a.y+0.5)*ch;ctx.beginPath();ctx.arc(x,y,Math.max(4,Math.min(cw,ch)*0.32),0,7);
  ctx.fillStyle=a.alive?(COLOR[a.persona]||"#9aa"):"#3a3f47";ctx.globalAlpha=a.alive?1:0.5;ctx.fill();ctx.globalAlpha=1;
  if(a.id===selected){ctx.lineWidth=2;ctx.strokeStyle="#fff";ctx.stroke();}}}
function renderFeed(){const feed=$("#feed");feed.innerHTML="";for(let k=0;k<=i;k++){const f=frames[k],n=id=>esc(nameMap(f)[id]||id);
 for(const e of f.events){const row=document.createElement("div");row.innerHTML=`<span class="d">D${f.day}</span>${story(e,n)}`;feed.prepend(row);}}}
function renderPossess(){if(!selected)return;const f=frames[i];const a=f.agents.find(x=>x.id===selected);if(!a){$("#possess").innerHTML=`<p class="muted">Gone.</p>`;return;}
 const bar=(l,v,mx,c)=>{v=Math.max(0,Math.min(mx,v||0));return `<h3>${l} <span class="muted">${Math.round(v)}</span></h3><div class="bar"><i style="width:${v/mx*100}%;background:${c}"></i></div>`;};
 const rels=(a.relationships||[]).map(r=>`<div class="rel"><span>${esc(r.name)}</span><span class="muted">${r.trust}</span></div>`).join("")||`<p class="muted">No ties yet.</p>`;
 const mem=(a.memory||[]).length?a.memory.slice().reverse().map(m=>`<div>· ${esc(m)}</div>`).join(""):`<p class="muted">No vivid memories.</p>`;
 $("#possess").innerHTML=`<h2 style="margin:.2rem 0">${esc(a.name)} ${a.alive?"":"<span class='muted'>(deceased)</span>"}</h2>
  <p class="muted">${esc(a.role)}</p>
  ${bar("Energy",a.energy,100,"#3fb950")}${a.hunger!=null?bar("Hunger",a.hunger,100,"#f0b429"):""}
  ${a.fatigue!=null?bar("Fatigue",a.fatigue,100,"#8957e5"):""}${a.fear!=null?bar("Fear",a.fear,100,"#f85149"):""}
  <h3>Wealth <span class="muted">${esc(a.money)} coin · ${esc(a.food)} food</span></h3>
  ${a.reputation!=null?`<h3>Standing <span class="muted">${esc(a.reputation)}</span></h3>`:""}
  <h3>Relationships</h3>${rels}<h3>Memories</h3>${mem}`;}
function render(){const f=frames[i];$("#day").textContent=`Day ${f.day}`;$("#slider").value=i;
 $("#popline").textContent=`${f.living}/${f.population} alive · crimes ${f.crimes}`;
 $("#verdict").textContent=(i===N-1&&frames[N-1].verdict)?frames[N-1].verdict:"(running…)";
 draw();renderFeed();if(selected)renderPossess();}
canvas.addEventListener("click",ev=>{const f=frames[i];const r=canvas.getBoundingClientRect();
 const cw=canvas.width/DATA.width,ch=canvas.height/DATA.height;const gx=(ev.clientX-r.left)/cw,gy=(ev.clientY-r.top)/ch;
 let best=null,bd=9;for(const a of f.agents){const d=Math.hypot(a.x+0.5-gx,a.y+0.5-gy);if(d<bd){bd=d;best=a;}}
 if(best&&bd<1.2){selected=best.id;showTab("possess");renderPossess();draw();}});
function showTab(name){document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("active",b.dataset.tab===name));
 $("#feed").classList.toggle("hidden",name!=="feed");$("#possess").classList.toggle("hidden",name!=="possess");}
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
$("#slider").oninput=e=>{i=+e.target.value;render();};
function play(){if(playing){clearInterval(timer);timer=null;playing=false;$("#play").textContent="▶ Play";return;}
 playing=true;$("#play").textContent="⏸ Pause";timer=setInterval(()=>{if(i>=N-1){play();return;}i++;render();},700);}
$("#play").onclick=play;window.addEventListener("resize",fit);
fit();render();
</script></body></html>"""


def build(persona="gemini", seed=42, days=15, out="emergence_observatory_demo.html"):
    api = EmergenceAPI()
    st = api.create_world(persona=persona, seed=seed, days=days, rich=True)
    wid = st["world_id"]
    frames = []

    def capture(state):
        agents = []
        for a in state["agents"]:
            v = api.agent_view(wid, a["id"])
            s = v["snapshot"]
            agents.append({
                "id": a["id"], "name": a["name"], "persona": a["persona"],
                "x": a["x"], "y": a["y"], "alive": a["alive"],
                "energy": s.get("energy"), "money": s.get("money"),
                "food": s.get("food"), "hunger": s.get("hunger"),
                "fatigue": s.get("fatigue"), "fear": s.get("fear"),
                "reputation": s.get("reputation"), "role": v["role"],
                "memory": v["memory"][-8:],
                "relationships": [{"name": r["name"], "trust": r["trust"]}
                                  for r in v["relationships"][:8]],
            })
        return {
            "day": state["day"], "living": state["living"],
            "population": state["population"],
            "crimes": state["metrics"]["crimes_total"],
            "facilities": [{"type": f["type"], "x": f["x"], "y": f["y"]}
                           for f in state["facilities"]],
            "agents": agents,
            "events": state.get("new_events", []),
            "verdict": None,
        }

    while True:
        out_state = api.step(wid, days=1)
        frames.append(capture(out_state))
        if out_state["finished"]:
            frames[-1]["verdict"] = out_state["verdict"]
            break

    data = {"width": st["width"], "height": st["height"], "persona": persona,
            "seed": seed, "days": days, "frames": frames}
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    kb = len(html.encode("utf-8")) / 1024
    print(f"wrote {out}  ({len(frames)} days, {kb:.0f} KB)")
    return out


if __name__ == "__main__":
    a = sys.argv
    build(a[1] if len(a) > 1 else "gemini",
          int(a[2]) if len(a) > 2 else 42,
          int(a[3]) if len(a) > 3 else 15,
          a[4] if len(a) > 4 else "emergence_observatory_demo.html")
