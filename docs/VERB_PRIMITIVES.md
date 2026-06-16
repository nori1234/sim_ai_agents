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
