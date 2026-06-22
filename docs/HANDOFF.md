# Handoff — start here

A running multi-agent town sim ("Emergence World"). This doc lets the next
session pick up immediately. Skim it, then look at the open issues.

## What it is (one line)
A **single LLM individuated into citizens by external scaffolding** (a per-agent
character prompt + external per-agent memory), set loose in the engine's world so
a **society emerges**. You watch it from above and can **possess** any citizen.
The "4 personas" (Guardian/Philosopher/Idealist/Predator) are a **legacy
demo/benchmark**, not the concept (see `docs/OBSERVATORY.md`).

## Core philosophy (the spine of every decision)
**Institutions are not hardcoded — they emerge from primitives + agent choice.**
The engine provides the *alphabet* (primitive verbs `take/give/use/strike/say/
bond/make/move/mate` + a *norm channel* + enumerated motives/affordances); the
**LLM writes the sentences**. So: security = a norm + a guard's `arrest` *action*
(not a police aura); money = a normal inventory item; a law = published text that
agents choose to honour/enforce (the engine has no per-law mechanism);
tax/fines = conserved `take`s; corruption = a guard's choice not to arrest. Canon:
`docs/PRINCIPLED_MIGRATION.md`, `docs/VERB_PRIMITIVES.md`, `docs/ECONOMY.md`.

## Run & test
```bash
python3 -m emergence.server            # observatory UI → http://127.0.0.1:8800 (JA/English toggle top-right)
python3 -m emergence.cli --compare     # the 4-society demo/benchmark (heuristic, free)
python3 -m unittest discover -s tests  # full suite (~279 tests; keep green)
```

## Working conventions (please keep)
- **Branch → PR → squash-merge** (via the GitHub MCP tools). One change per PR.
  Don't push to `main` directly. Branch names like `feat/...` / `docs/...`.
- **Determinism is the contract.** `tests/test_baseline_contract.py` pins the
  four-society outcomes for the **heuristic brain, all layers OFF**. Every change
  must keep it **byte-identical**. New behaviour goes in **opt-in layers**
  (`--economy`, `--society`, `--environment`, `--memory`, `library=True`, …);
  the heuristic ignores the new `Observation` fields, so offline stays identical.
- **Reproducibility** = engine `seed` + **record/replay** of LLM calls
  (`emergence/replay.py`) — never `temperature=0`, never runtime code-gen.
- **Personality stays OUT of the engine** (lives in `brains/`/`personas`/
  `scenario`), like memory. The engine never knows an agent's persona knobs.
- **i18n**: UI strings + engine narrative are JA/EN (`?lang=ja`); English output
  is byte-identical to before localization. Keep both when adding text.
- End commit messages with the Co-Authored-By / Claude-Session trailer; PRs with
  the "Generated with Claude Code" footer. Don't put model IDs in artifacts.

## Architecture map
```
emergence/
  world.py simulation.py agent.py actions.py   # the engine (stdlib, deterministic)
  observation.py            # what a brain sees (add opt-in fields here; heuristic ignores them)
  affordances.py            # roles/production as DATA (possibility space)
  brains/heuristic.py llm.py # deterministic free tier / LLM (the "soul")
  personas.py               # 4 preset knob-vectors (legacy benchmark seeds)
  governance.py economy.py market.py drives.py esteem.py psyche.py society.py
  environment.py publicworks.py development.py memory_backend.py library.py
  chronicle.py              # the story (chronicle + life story; JA/EN)
  api.py server.py web/observatory.html   # product layer: API + stdlib HTTP + rich-2D UI
docs/  OBSERVATORY PRINCIPLED_MIGRATION VERB_PRIMITIVES ECONOMY LAYERS ARCHITECTURE LOCAL
```

## Shipped this session (all merged to main)
Rich-2D observatory (#18) · town library = cultural inheritance, burnable (#19) ·
concept reframe to "one LLM → a society" (#28) · economy MVP: specialisation of
supply gives money demand (#27) · Japanese UI (#29) + Japanese engine narrative
(#30) · UI: final report + facility labels + meaningful law beats (#33) ·
principled fiscality under `--economy` (#36) · publish every law as a norm so
LLMs act on novel laws (#39) · corruption (bribery) as emergence (#43) ·
route remaining trades/transfers through the primitive (#46).

## Open backlog (17 issues) — grouped
- **Economy:** #20 money consumption-demand · #21 richer professions/facilities ·
  #31 firms (事業主) + state · #32 economic failure → survival · #34 fiscal
  *collector acts to tax* (conserved fiscality already shipped #36) · #45 money
  supply (work mints money — design decision; see below).
- **Law/governance/corruption:** #35 law granularity (distinct offences) · #37
  law-as-norm (publish shipped #39; remaining: safe law-effect DSL) · #38
  corruption (bribery shipped #43; remaining: embezzlement — needs #34 collector,
  selective enforcement).
- **Library/knowledge:** #22 rot + scribe upkeep · #23 memory-agent recall
  adapter · #25 visualize the library in the UI.
- **Personality:** #24 per-agent character + heritable traits (= the *implementation*
  of "one LLM → individuals"; **high-value, product-defining**).
- **Cross-cutting quality:** #40 performance · #41 security (SSRF/keys/auth — for
  hosting) · #42 UI/UX polish & observability.

## Immediate pending decision — #45 (money supply)
Work pay mints money from nothing (`simulation.py` `_do_work`). It's the last
non-conserved "magic". **Recommended: do NOT do #45 standalone** — sourcing wages
needs an employer with capital, i.e. **#31 (firms pay wages)**. Two clean moves:
- **A (quick):** accept open-minting as intentional "labour creates value",
  document it in `docs/ECONOMY.md`, and close #45 referencing #31.
- **B (big):** tackle #31 (firms/wages) and source wages there (closing the loop
  sell→wage→consume→sell), opt-in under `--economy`, baseline byte-identical.

## Suggested next steps (pick one)
1. **#24 per-agent character generation** — the product concept ("one LLM →
   distinct individuals") still isn't in code: LLM agents share one of 4 preset
   prompts (`llm.py _build_system_prompt`). Generating a per-citizen character +
   using its own memory is the highest-leverage, on-brand move.
2. **#45 decision A** (quick doc close) then **#31 firms/wages** (economy's core).
3. **#42 UI** — surface `laws_in_force` / economy / library in the observatory.
