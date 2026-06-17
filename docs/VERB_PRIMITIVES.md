# Verb primitives — making the *action vocabulary* obey the engine principle

> **Base principle** (the engine provides physics, not institutions): see
> [`PRINCIPLED_MIGRATION.md` § The principle](PRINCIPLED_MIGRATION.md). This
> document applies that same principle one level deeper — to the verb
> vocabulary itself — and records the design, current status, and the plan for
> the verbs still left as macros.

## Why: the verbs were institutions in disguise

`ActionType` grew to ~35 verbs, but most are not physical primitives — they are
institutions promoted to verbs. `steal`, `transfer`, `gather`, and
`deposit_granary` are *the same physical act* (move an item from A to B); they
differ only in **who** the counterparty is and **whether there was consent**.
Shipping them as separate verbs bakes the institution ("theft", "gift",
"commons") into the engine — the very thing the engine shouldn't do.

> The full `ActionType` list is a **standard library**. What the engine should
> expose is an **instruction set**: a small set of physical verbs the library
> compiles down to.

`have` is a useful probe: `walk` is already a primitive (`move`), but `have` is
**not a verb** — it is a *state* ("I hold X"), modelled by `inventory`. The
verbs that change it are `take` (begin having) and `give` (end having).
`be / own / have` are states; `take / give / make` are actions.

## The instruction set (9 primitives)

| primitive | physical meaning | everyday verbs it absorbs |
|-----------|------------------|---------------------------|
| `move` (walk) | change location | move |
| `take` | pull an item into own inventory | gather, steal, draw_granary, a trade's receive |
| `give` | push an item out of own inventory | transfer, deposit_granary, a trade's hand-over |
| `use` | apply an item to self or target | eat, take_drug, rest, sleep |
| `make` | transform inputs/effort into an output | create, craft, build, craft_weapon |
| `strike` | apply force to damage an agent or structure | attack, arson |
| `say` | broadcast a signal/message | speak, praise, preach, propose, report_crime |
| `bond` | commit to a relationship or agreement | vote, accept, lend, repay, worship, join_gang |
| `mate` | combine with a partner to produce offspring | mate |

(`idle`/`rest` are `use` on self with no item, or a trivial wait. `arrest` is
composite enforcement — `take` a fine + `move` the target to prison — so it
stays a macro, not a primitive.)

## Institutions are interpretations, not verbs

With primitives, an institution is a **label the world applies to a primitive
act in context**, not a distinct verb:

- `take(from=agent, consent=False)` → **theft** (a crime): strikes fear, damages
  trust — today's `steal` effects, derived from the act not the verb name.
- `give(to=agent, consent=True)` → a **gift**; `give` to the commons → a
  **granary deposit**.
- `strike` a person → **violence**; `strike` a structure → **arson**.
- `say` content others come to believe → a **religion** spreads; no `preach`
  verb, only speech that catches on.

Same move as money/police/law: the engine provides the physics; the *meaning*
(crime, gift, violence) is read off physics + context by `_interpret`.

## The layered architecture

```
  Brain (heuristic | LLM) emits one Action
   ├── heuristic -> macro verbs only  (steal, transfer, eat, attack, ...)
   └── LLM       -> macros OR raw primitives (take, give, use, ...)
        │
        ▼  Lowering: a macro is a thin handler that calls a physics helper
        ▼  Physics helper (_move_items / _strike / _use_item / ...)
        │     the ONLY code that mutates state; returns an Event
        ▼  Interpretation (_interpret): act + context -> institution
        ▼  metrics / world.log / memory
```

**Physics vs accounting — the split that keeps it safe.** Energy cost stays in
the *entry handler* (`steal` costs 3.0, `transfer`/`eat` cost 0); the physics
helper never spends. If every verb routed through one spending primitive those
costs would collapse and the baseline would shift. Raw primitives get their own
modest costs; the heuristic never emits them, so the baseline is untouched.

**The `Event`** is the connective tissue — `kind`, `actor`, `other`, `items`,
`consent`, `site`. Metrics/fear/trust key off the Event, not the verb name; a
new institution is a new `_interpret` branch, not a new verb.

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

## Status — what is lowered today

The substrate exists and the everyday verbs lower to it. All of this is in
place; `tests/test_primitives.py` covers the interpretations and the
macro-equivalences.

| macro | lowers to | interpreted as |
|-------|-----------|----------------|
| steal | `take(consent=False)` | theft |
| gather | `take(from world node)` | (harvest) |
| transfer | `give(consent=True)` | gift |
| eat | `use(food on self)` | (metabolism) |
| take_drug | `use(drug on self)` | (dose: addiction) |
| attack | `strike(person)` | violence |
| arson | `strike(structure)` | arson |
| create | `make(work)` | (creation) |
| craft_weapon | `make(weapon)` | (arming) |
| build | `make(structure)` | (a monument earns honour) |
| speak | `say` | (public statement) |
| propose | `say(intent=proposal)` | a bill put to the legislature |
| report_crime | `say(intent=accusation)` | (accusation) |
| praise | `say(intent=praise)` | esteem grant (honour, relief) |
| preach | `say(intent=sermon)` | found / spread a faith |
| vote | `bond(proposal)` | (assent) |
| worship | `bond(intent=worship)` | prayer: relief + communion |
| join_gang | `bond(intent=gang)` | join / found a crew |

Raw `take/give/use/strike/make/say/bond` are in the LLM action menu, so an LLM
agent can improvise on the physics (a spontaneous gift, an offering, a pact)
while the heuristic brain stays on the macros. **Guarantee:** the four-society
contract (`tests/test_baseline_contract.py`) is byte-identical through every
slice — the lowerings preserved every mutation, RNG draw, and metric call.

**The folding is complete.** Every institutional verb now lowers to a primitive
and its meaning is read off the act by `_interpret`. `accept`/`lend`/`repay`/
`craft`/`offer` already lived in the economy-primitives layer. The four-society
contract (`tests/test_baseline_contract.py`) stayed byte-identical through all of
it.

## What's left as a structured macro — on purpose

A few things stay as macros because they are *compositions* or carry preconditions
the primitive shouldn't own:

- **`arrest`** — composite enforcement (`take` a fine + `move` to prison + a
  status change); a macro over primitives, not a primitive itself.
- **Layer gating** — each folded verb keeps its preconditions (the esteem/society
  layer flag, a workshop/temple, a material cost) in the macro; only the *effect*
  moved into `_interpret`.

## Possible future work

- **Free-text intent for `say`.** Today only the `propose`/`praise`/`preach`
  macros set an explicit `intent`; a raw LLM `say` is always plain speech. A
  later step could parse free-form content into intent (a speech that *is* a
  proposal), making the LLM's raw `say` fully expressive.
- **World-sourced raw `take`.** `gather` lowers to a world-take internally, but
  the raw `take` verb still only targets agents; it could be extended to harvest
  from a node directly.

**Cross-cutting: the *intent* dimension.** Folded say/bond verbs carry an
`Event.intent` (praise / sermon / worship / gang); `propose` will add
`intent="proposal"`. The heuristic macros always pass an explicit intent
(deterministic); free-text intent parsing for the LLM's free-form `say` can come
later. Each remaining step keeps `test_baseline_contract` green and adds a
primitive-level test.

## Appendix — rationale & history

**The tradeoff we accepted.** Primitives maximise emergence (an LLM composes
`take`/`give`/`say` into behaviours we never enumerated) and extensibility (a
new institution needs no new verb). The cost is that metrics, which counted
institutions *by verb*, now need the interpretation layer to classify acts. We
took that cost because it is exactly the engine principle, paid down in slices.

**Why layered, not a big-bang rewrite.** Named verbs became thin macros that
lower to primitives; the interpretation layer pulls the institution logic out of
`_do_steal` et al. so it keys off the act; the heuristic brain stays on the
macros (preserving determinism and the contract byte-for-byte); only the LLM
brain gets the raw primitives. This bought the open verb space without risking
the four societies in one step.

**Resolved design questions.** Consent is a flag on the act for now (a prior
`offer`/`bond` as first-class consent is a later refinement). `arrest` stays a
composite macro. Each primitive carries its own `ACTION_ENERGY_COST`; macros
spend their own cost and the physics helper never spends.

The per-slice history (Slices 1, 2, 3a, 3b) lives in the git log and in the
commit messages; this document tracks the settled design and the remaining plan.
