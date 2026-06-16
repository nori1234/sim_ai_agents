# Verb primitives — making the *action vocabulary* obey the engine principle

## The principle, applied one level deeper

We dissolved three baked-in institutions (money field, police aura, law magic)
into primitives (see `PRINCIPLED_MIGRATION.md`). But one institution remains
hiding in plain sight: **the verb vocabulary itself.**

`ActionType` has ~35 verbs. Most are not physical primitives — they are
institutions promoted to verbs. `steal`, `transfer`, `gather`, and
`deposit_granary` are *the same physical act* (move an item from A to B); they
differ only in **who** the counterparty is and **whether there was consent**.
Shipping them as separate verbs bakes the institution ("theft", "gift",
"commons") into the engine, exactly the thing we said the engine shouldn't do.

> The current `ActionType` list is a **standard library**. What the engine
> should expose is an **instruction set** — a small set of physical verbs the
> library compiles down to.

## `have` is a state, not a verb

A useful probe. `walk` is already a primitive (`move`). But `have` is **not a
verb at all** — it is a *state* ("I am holding X"), modelled by `inventory`.
The verbs that change that state are `take` (create having) and `give` (end
having). `be / own / have` are states; `take / give / make` are actions. The
engine should model possession as a state and ownership as a relation, and keep
the verb set to the acts that change them.

## The proposed instruction set (9 primitive verbs)

| primitive | physical meaning | current verbs it absorbs |
|-----------|------------------|--------------------------|
| `move` (walk) | change location | move |
| `take` | pull an item into own inventory | gather, steal, draw_granary, the receive half of a trade |
| `give` | push an item out of own inventory | transfer, deposit_granary, the hand-over half of a trade |
| `use` | apply an item to self or target, changing state | eat, take_drug, rest, sleep |
| `make` | transform inputs (+ place/tool) into an output | craft, build, create, craft_weapon |
| `strike` | apply force to damage an agent or structure | attack, arson |
| `say` | broadcast a signal/message (optional target) | speak, praise, preach, propose, report_crime |
| `bond` | commit to a relationship or agreement | vote, accept, join_gang, worship, lend, repay |
| `mate` | combine with a partner to produce offspring | mate |

(`idle`/`rest` can be `use` on self with no item, or a trivial `wait`. `arrest`
is composite enforcement — `take` a fine + `move` the target to prison — and is
discussed under the interpretation layer below.)

## Institutions become *interpretations*, not verbs

With primitives, an institution is a **label the world applies to a primitive
act in context**, not a distinct verb:

- `take(item, from=agent, consent=False)` → the world labels it **theft** (a
  crime), strikes fear, damages trust — exactly today's `steal` effects, but
  derived from the act + context rather than hard-coded into a verb.
- `take(item, from=agent, consent=True)` where the consent comes from a prior
  `offer`/`bond` → **trade**.
- `say(content)` that others come to believe and act on → a **religion**
  emerges; there is no `preach` verb, only speech that spreads.
- `give` to the commons → a **granary deposit**; `give` to a person → a **gift**.

This is the same move we made for money/police/law: the engine provides the
physics; the *meaning* (crime, trade, faith) is read off the physics + context.

## Why not just do it — the honest tradeoff

**For primitives:** maximal emergence (an LLM can compose `take`/`give`/`say`
into behaviours we never enumerated), real extensibility (a new institution
needs no new verb), and conceptual purity (the philosophy, all the way down).

**The cost:** we currently *count* `crimes`, `trades`, `births` **by verb**.
Primitives force a new **interpretation layer** that classifies acts
("consent-less `take` from an agent = theft") so the metrics, the report, and
the four-society contract still mean something. The heuristic brain also has to
move from "script named verbs" to "compose primitives", which is where the
contract is most at risk. None of this is impossible; it is just real work that
should not ride along silently.

## Recommended path — layer it, don't flip it

1. **Primitives as the substrate.** Add `take`/`give`/`use`/`make`/`say`/`bond`
   as real actions with consent/target parameters.
2. **Named verbs become thin macros** that lower to primitives
   (`steal` ≡ `take(from=agent, consent=False)`), so nothing visible changes
   for callers on day one.
3. **An interpretation layer** classifies primitive acts into institutions for
   metrics/fear/trust (the logic currently living inside `_do_steal` etc.,
   pulled out so it keys off the act, not the verb name).
4. **Heuristic brain stays on the macros** → determinism and the four-society
   contract are preserved byte-for-byte.
5. **LLM brain gets the primitives** in its action menu → it can improvise
   novel acts on the same physics.

This buys the open verb space and the emergence without putting the contract at
risk in a single big-bang refactor.

## Acceptance criterion (same as every prior phase)

`tests/test_baseline_contract.py` must stay green: the four societies
(guardian→ORDER, philosopher→CHAOS, idealist→COLLAPSE, predator→FAILURE) keep
emerging and stay distinct. The numeric snapshot may move deliberately when a
slice changes behaviour, never silently.

## Open questions

- **Granularity of `say`.** Does `propose`/`vote` (structured governance) lower
  cleanly to `say`/`bond`, or does structured deliberation deserve to stay a
  macro indefinitely? (Leaning: keep governance as a macro over `say`+`bond`.)
- **`arrest` as a primitive vs composite.** It is `take`(fine)+`move`(to prison)
  + a status change. Probably a macro, not a primitive.
- **Consent representation.** Is consent a flag on `take`, or is it an
  `offer`/`bond` the `take` must reference? (Leaning: reference a prior `bond`,
  so consent is itself a first-class, inspectable object.)
- **Energy/cost model.** Primitives need their own `ACTION_ENERGY_COST` entries;
  macros inherit from the primitive they lower to.

## Suggested first slice (when we implement)

Per the AskUserQuestion options, the smallest testable start is: add
`take`/`give`/`use`, route `steal`/`transfer`/`gather`/`eat` through them, and
introduce the consent→crime interpretation — one slice, contract green.

---

# Concrete shape — the layered architecture

## One tick, end to end

```
  Brain (heuristic | LLM) emits one Action
   ├── heuristic -> macro verbs only  (STEAL, TRANSFER, GATHER, EAT, ...)
   └── LLM       -> macros OR raw primitives (take, give, use, ...)
        │
        ▼
  Lowering: a macro is a thin handler that calls the shared physics helper
     steal    -> _move_items(actor, victim, {money:5, food:2}, kind=take, consent=False)
     transfer -> _move_items(actor, target, {resource:amount}, kind=give, consent=True)
        │
        ▼
  Physics helper (the ONLY code that moves items between holders)
     _move_items(...) -> mutates inventories, builds an Event, calls _interpret
        │   Event(kind=take, actor, other=victim, items={money:5,food:2}, consent=False)
        ▼
  Interpretation: institution is read off the act + context (not the verb name)
     take from agent, consent False -> theft  -> _register_crime (fear, trust--)
     give to agent,  consent True   -> gift   -> metrics.transfers, ledger, trust++
        ▼
  metrics / world.log / memory
```

## The split that makes it safe: physics vs accounting

Two responsibilities that are tangled inside today's `_do_steal` get separated:

* **Accounting (energy cost)** stays in the *entry handler*. `steal` costs 3.0,
  `transfer`/`eat` cost 0. If every verb routed through one spending primitive,
  those costs would collapse to a single value and the baseline would shift.
  So each entry handler spends its own `ACTION_ENERGY_COST`, then calls the
  physics helper, which **never** spends. Raw primitives (LLM) get their own
  modest cost entries; the heuristic never emits them, so the baseline is
  untouched.
* **Physics (item movement)** lives in `_move_items`, shared by every verb that
  moves goods. It returns an `Event` and hands it to `_interpret`.

## The Event object — the new connective tissue

```python
@dataclass
class Event:
    kind: str                 # "take" | "give" | "use" | ...
    actor: Agent
    other: Optional[Agent]    # counterparty, if any
    items: dict[str, int]     # what ACTUALLY moved (post-clamp)
    consent: Optional[bool]   # True | False | None
```

Metrics, fear, trust, and memory key off the `Event`, not the verb name. A new
institution becomes a new branch in `_interpret`, not a new verb.

## `_do_steal`: before → after

```python
# before — institution welded into the verb
def _do_steal(self, agent, action):
    victim = self._adjacent_or_targeted(agent, action)
    if victim is None: return
    self._spend(agent, ActionType.STEAL)
    agent.money += victim.take("money", 5)
    agent.add("food", victim.take("food", 2))
    self._register_crime(agent, "theft", victim)

# after — macro lowers to a primitive; meaning is interpreted
def _do_steal(self, agent, action):
    victim = self._adjacent_or_targeted(agent, action)
    if victim is None: return
    self._spend(agent, ActionType.STEAL)                 # accounting stays here
    self._move_items(agent, victim, {"money": 5, "food": 2},
                     kind="take", consent=False)         # physics + interpret
```

The order of mutations (`money` then `food`), the lack of RNG, and the
`_register_crime` effects are all preserved exactly — so the heuristic baseline
stays **byte-identical** and `test_baseline_contract` stays green.

## Slice plan (each slice keeps the contract green)

* **Slice 1 — `take` / `give` + interpretation. (DONE)** Added the two
  movement primitives, the `Event` record, the shared `_move_items` physics
  helper, and the `_interpret` layer. `steal` lowers to `take(consent=False)`
  and `transfer` to `give(consent=True)`; their theft/gift effects now live in
  `_interpret`, keyed off the act not the verb name. Raw `take`/`give` are in
  the LLM menu. The Phase-4 contract snapshot is **byte-identical** (the
  lowering preserved every mutation, RNG draw, and metric call), and
  `tests/test_primitives.py` covers consent-less take → theft, consensual
  give → gift, conservation, and the macros still behaving.
* **Slice 2 — `use`. (DONE, partial)** Added the `use` primitive and
  `_use_item`, the place item physics lives (food → energy + hunger relief).
  `eat` lowers to `use(food on self)`; the contract snapshot is byte-identical
  and `tests/test_primitives.py` checks `use` restores energy and that the
  `eat` macro is exactly `use 2 food`. Raw `use` is in the LLM menu.
  **Deferred from this slice with rationale:** `gather` is *extraction from the
  world* (a resource node with yield/regen), not a holder-to-holder move, so it
  does not fit `_move_items`; it belongs with a future world-sourced `take` /
  `harvest`. `take_drug` carries society-layer gating (materials cost, addiction
  thresholds, a drug-den role), so folding it into `use` is a behavioural change
  better done deliberately. Both keep their own handlers for now.
* **Slice 3a — `strike` / `make` (the physical ones). (DONE)** `strike` applies
  force; `_interpret` reads it as **violence** (vs a person → `_register_crime`)
  or **arson** (vs a structure → inline crime accounting + dread from the site),
  exactly parallel to take→theft. `attack` and `arson` lower to it, and the
  contract snapshot stays byte-identical (both occur in the baseline). `make`
  transforms effort into an output; `create` lowers to `_make_work`, and the
  raw `make` verb also routes recipe goods to the craft physics. Raw
  `strike`/`make` are in the LLM menu; tests cover violence/arson interpretation
  and macro-equivalence. **Deferred with rationale:** `build` (facility
  construction couples to the public-works treasury/voting) and `craft_weapon`
  (society layer) route through `make` in a later pass; `craft`/`offer`/`accept`
  already live in the economy-primitives layer.
* **Slice 3b — `say` / `bond` (the social ones). (DONE)** `say` broadcasts a
  signal: `speak` lowers to a public say, `report_crime` to a say aimed at the
  accused. `bond` commits to an agreement: `vote` lowers to `_bond_to_proposal`
  (assent to a collective decision), and the raw `bond` verb also forms a pact
  of mutual allegiance (trust) between two agents — a new affordance the LLM can
  improvise with. Raw `say`/`bond` are in the LLM menu; the contract is
  byte-identical. **Deferred with rationale:** `praise` (esteem layer),
  `propose` (governance-structured: text→law parsing, build inference), and
  `preach`/`worship`/`join_gang` (society layer) stay as structured macros for
  now; `accept`/`lend`/`repay` are already bond-family economy primitives.

## Where the instruction set stands

The substrate now exists: **move, take, give, use, strike, make, say, bond**
(plus mate). The everyday institutional verbs (steal, transfer, eat, attack,
arson, create, speak, report_crime, vote) are thin macros that lower to them,
and meaning (theft, gift, violence, arson) is read by `_interpret` from the act
plus context. The LLM brain can call the raw primitives to improvise; the
heuristic brain stays on the macros, so all four societies remain byte-identical.
What is intentionally left as structured macros — governance proposals, the
esteem/society/economy layers — is documented above, not hidden.

---

# Folding-in plan for the deferred verbs

The deferred verbs fall into a clean pattern: each lowers to a primitive, with
its layer-specific physics moving into the primitive's effect dispatch or into
an `_interpret` branch, and its preconditions (layer flag, location, cost)
staying in the macro. The work is grouped by target primitive below.

**The ordering principle is contract risk.** The four-society contract runs with
the esteem/society/economy layers *off*, so folding those verbs is low risk:
their interpretation branches simply never fire in the baseline, and their own
layer tests (`test_status`, `test_society`) guard them. The verbs that are
*active in the baseline* — `gather`, `build`, `propose` — must be lowered with
byte-identity diligence against `test_baseline_contract`.

## Into `take` (world-sourced)
- **`gather` → `take(from=world node)`.** Physics: extend movement so a source
  can be a Facility that *produces* via `gather_yield()` + `environment.gather()`
  rather than draining a holder's inventory; Event `other=None, site=node`. No
  institution to interpret (gathering is not a crime). **Risk: HIGH** (baseline-
  active, environment-coupled) — guard with the byte-identical snapshot.

## Into `use`
- **`take_drug` → `use(item="drug")`.** Physics: the `_dose` effect (energy
  spike, addiction, pleasure, `doses_taken`) becomes the drug branch of
  `_use_item`. Gating (society.drugs, materials self-supply, the `drug_den`
  role) stays in the macro. **Risk: LOW** (society layer off in baseline).

## Into `make`
- **`build` → `make(output=facility_type)`.** Physics: construction (materials
  spent, facility added or joined, monument honour, public-works treasury) routes
  through `make`; treasury/voting coupling stays. **Risk: MED** (monument builds
  occur in the baseline; public-works is opt-in).
- **`craft_weapon` → `make(output="weapon")`.** Physics: `weapons += 1`,
  `weapons_crafted`, the `weapons_factory` role. Gating (society.weapons,
  workplace, material cost) stays. **Risk: LOW**.

## Into `say`
- **`praise` → `say(to=target, kind="praise")`.** A clean fit: `_interpret`
  reads a praise-kind say as esteem relief + reputation + pleasure + mutual
  trust (the status effects). Gating (status.enabled) stays. **Risk: LOW**.
- **`propose` → `say(content, kind="proposal")`.** The hard one: `_interpret`
  must create a `Proposal` (legislature.propose + build inference). Plan A (do
  first): the say carries an explicit proposal payload (text/build), so the
  lowering is deterministic. Plan B (later): parse free-text `say` for proposal
  intent. **Risk: HIGH** (governance is baseline-active).
- **`preach` → `say(kind="sermon")`.** `_interpret` founds/spreads the faith
  (society.religion). Gating stays. **Risk: LOW**.

## Into `bond`
- **`worship` → `bond(to=faith, at=temple)`.** Fear/esteem relief, pleasure,
  `acts_of_worship`, communion trust (society.religion). **Risk: LOW**.
- **`join_gang` → `bond(to=gang)`.** Join/found a gang, arm the member,
  loyalty/suspicion trust (society.gangs). **Risk: LOW**.
- `accept`/`lend`/`repay` are already bond-family economy primitives — no work.

## Cross-cutting: the act gains an *intent* dimension
Several foldings (praise, propose, preach) need the act to carry intent — a
praise vs a plain statement, a proposal vs a speech. Two options:
1. **An explicit `kind`/`intent` param** on the primitive (structured,
   deterministic). The heuristic macros always pass it; recommended first.
2. **Parse free-text content** for intent (richer, LLM-flavoured) — only the
   free-form LLM `say` needs this; add it later.

`Event` already carries `kind`; the foldings extend the vocabulary
("praise", "proposal", "sermon") and add the matching `_interpret` branches.

## Suggested sequence
1. Low-risk layer verbs (contract trivially safe): `praise`, `take_drug`,
   `craft_weapon`, `preach`, `worship`, `join_gang`.
2. Baseline-active verbs with byte-identity diligence: `gather`, `build`,
   `propose` (the last also introduces the structured-intent `say`).

Each step keeps `test_baseline_contract` green and adds a primitive-level test,
exactly as slices 1–3 did.
* Each slice: heuristic stays on macros (contract byte-identical); the LLM menu
  gains the new primitive so it can improvise.

## Re-checked for breakdown — none found

* **Energy** — preserved by keeping the spend in the entry handler (see above).
* **RNG/determinism** — `steal`/`transfer` draw no randomness; movement order is
  preserved, so the deterministic stream is unchanged.
* **Double-counting** — verbs not yet lowered (e.g. `attack`) keep calling
  `_register_crime` directly; lowered verbs reach it once via `_interpret`. No
  overlap.
* **Money** — `take`/`add` operate on inventory-backed money, identical to the
  current `agent.money += ...` arithmetic.
* **Metric fidelity** — `transfer`'s ledger/trust/remember/log effects are
  replicated verbatim in `_interpret`'s gift branch.
