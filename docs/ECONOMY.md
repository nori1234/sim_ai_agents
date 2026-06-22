# Economy & labour — design direction (plan of record)

Status: **plan, not yet implemented.** Tracks issues
[#20](https://github.com/nori1234/sim_ai_agents/issues/20) (money has no
consumption demand) and [#21](https://github.com/nori1234/sim_ai_agents/issues/21)
(professions & facilities are thin). They are two halves of one loop, so they are
designed and built together.

## The problem (from the code, today)

- **Survival is free.** `_do_gather` (food at a farm) + `_do_eat` + `_do_sleep`/
  `_do_rest` cost no money, so an agent never *needs* money to live.
- **No money → need-satisfaction path.** You can't buy better rest/healing, and
  you can't buy standing (`esteem.py` has no money→reputation route). The only
  genuine consumption sink is **drugs** (`--society`) — i.e. only the underworld
  currently gives money intrinsic pull.
- **Professions are flavour.** 10 jobs (`scenario.py`), but their role is a
  one-line hint (`affordances.py PROFESSION_ROLES`) the heuristic ignores; jobs
  grant no distinct capability or productivity, so there is no division of
  labour and no reason to trade.

Net: money is a hoardable token; the town is ~10 interchangeable agents.

## The core structure: produce because others need it

The thing that should emerge is simple to state:

> **You make something *because someone else needs it*. That is what lets you
> sell it, and that is what turns labour into money.**

So the world should hold **different producers** — a smith who forges tools, a
farmer who grows bread — each making what they're good at, *for others*. And
**services are labour too**: an inn's rest, a doctor's healing — someone *works*
to provide them, so they cost money like any other good. Money is wanted because
it is the claim on **everyone else's work**.

```
specialised work produces what OTHERS need → sell it for money → buy what you
DON'T produce (goods) and the services/needs you can't cheaply meet yourself
        ↑                                                        ↓
   division of labour  ←————  money gains real demand  ←——  food / rest / healing
                                                              / status are buyable
```

The two issues are the two arrows of the same wheel: demand for money (#20) only
becomes real once jobs specialise (#21), and specialisation only pays once money
buys what you lack (#20).

## Why money gains demand (the mechanism)

The root cause of "money has no demand" is the **free, universal subsistence
escape hatch**: if anyone can gather food cheaply, nobody buys, so producers have
no buyers and no reason to make a surplus. The fix is **specialisation of
supply** — each profession is good at producing *one* thing and **poor at the
rest** (a farmer's food, a miner's materials), so self-sufficiency is inefficient
*for everyone*. That makes the sensible path: specialise, sell your good, and buy
what you lack — which gives food/materials a **structural** demand, and gives the
farmer a buyer (hence a motive to produce).

Survival is **not** locked: an off-specialty agent can still gather, just
inefficiently (a fallback, never zero), so no one starves merely to create
demand. `--environment` (depletion, winter) sharpens scarcity further but is **not
required**. The LLM realises this as comparative advantage (it values its time);
the heuristic realises a minimal proxy (buy food when it's a poor food-gatherer
and an affordable offer exists, else gather).


## Design principles (keep the engine's stance)

1. **The engine provides the alphabet; the LLM writes the sentences.** The engine
   owns exactly two things — the **primitive verbs** (the instruction set:
   `take/give/use/strike/say/bond/make/move`, per `VERB_PRIMITIVES.md`) and the
   **enumerated menu of motives & affordances** (needs/drives, and what is
   possible *here*: buyable, sellable, hireable, taxable). It **never** scripts
   which motive wins or which act to take — that freedom is the LLM's and is
   non-negotiable. New economic acts are **interpretations of the primitives**,
   not new bespoke verbs: a purchase = `give(money)` + `take(good)`; employment =
   a `bond` + a wage `give`; tax = a coerced `give` (no consent) — the same
   interpretation layer that already reads "theft = take-without-consent" and
   "arson = strike-a-building".
2. **Emergence-first.** Don't script a "shop". Add **affordances** ("you can buy
   food here") and let brains choose. Prices are **not** a formula — they already
   emerge from accepted `OFFER`/`ACCEPT` swaps (`market.py`, `--economy`).
3. **Value is subjective and the LLM's; price is the public, emergent number.**
   An agent's *value* for a good forms from its own motives/needs (a starving
   agent prizes food) — the engine must **not** precompute or hand over a "value"
   number, because judging worth is the LLM's freedom (principle&nbsp;1). *Price*
   is only the emergent meeting-point where many private values settle into a
   trade ratio — a convenient public signal, not a truth. So surface the raw
   **motives** + the emergent **price**, and let the agent weigh its own value
   against the price.
4. **Money is not privileged** (per `PRINCIPLED_MIGRATION.md`). Consumption is
   just `money → good/service → use`, built from existing primitives.
5. **Institutions emerge, they are not hardcoded.** Firms and the state (below)
   are *emergent* the same way money, police and law are — we add the physics
   (employ, pay, levy) and let them form, not a built-in "company" or "tax
   office".
6. **Determinism.** The `--compare` baseline stays byte-identical. Everything
   here is **opt-in** (rides on `--economy`); `test_baseline_contract.py` is
   untouched. New signals ride on `Observation`; heuristic branches are gated.

## Value, items, and services (the conceptual model)

A question that keeps recurring — *"is a service an item?"* — has a precise
answer that shapes how we model doctors, banks, inns and the rest. Economic
history settles most of it.

**Value is subjective; an item is only its container.** Per the marginalist /
subjective-value tradition (Menger, Jevons — and principle&nbsp;3 above), value
does not live *inside* a thing; it arises because *someone needs it* at the
margin. A loaf is precious to the starving and worthless to the full. So value
rides on **a satisfied need**, and an item is merely a *vessel* that can hold and
move that value. A service satisfies a need just as a good does — in that sense a
service and a good are the **same kind of thing**: vessels of subjective value.

**The distinction that matters for the engine is not tangible vs intangible, but
storable/transferable vs not** — i.e. *is the result a `claim`-item or an
`effect`?*

| kind | examples | engine form | where value sits |
|---|---|---|---|
| **item** | food, materials, tools, **money**, **deposit-receipt / loan (a claim)** | an inventory token (conserved; ownable, movable, re-sellable) | in the need it can later meet |
| **effect-service** | healing, lodging, a feast → esteem | an instantaneous effect (energy / esteem); **not** storable or re-sellable | in the need satisfied *now* |

So **money is just the most tradable item** (historically: cattle, grain, salt,
shells, then metal — "money is the most salable good", Menger). It is not
privileged (principle&nbsp;4); demand for it comes from specialisation
(you can't cheaply self-produce every item) plus the fact that both items and
effect-services are claimed with it.

**Consumption services stay effects — don't reify them as fake items.** Healing
(shipped, #53), an inn's rest, a feast's honour: you cannot resell the care you
received. Modelling them as an applied *effect* is correct; the value is the need
met, not a token.

**Banking is the exception — and the most interesting case.** A bank's product
*is* a transferable **claim**, i.e. an item:

- a **deposit** = a claim-token on the bank (custody, maybe interest);
- a **loan** = a claim held by the creditor (already modelled as `market.Loan`).

This is not a modelling convenience — it is *how money was actually born*:
goldsmith/​money-changer **deposit receipts** circulated as paper, became
banknotes, then credit money (bills of exchange, the Templars' letters of
credit, fractional reserve). If agents come to **trust and trade a bank's
deposit-receipts**, that token becomes money **emergently** — exactly the
project's stance that institutions emerge from primitives, and a flagship
phenomenon only this kind of sim can show. The flip side is also emergent: a
banker holding others' money *can* embezzle or face a run (ties to #38), because
custody is a trusted promise, not an engine-guaranteed vault.

**How this rides on one substrate.** The service-offer mechanism (#53) carries
the **transaction** uniformly — a provider chooses to offer at a price it picks,
a taker consents and pays (conserved), the price emerges from accepted fees. The
only branch is the **result**: an effect fires (healing) *or* a new claim-item is
issued (a deposit-receipt / a loan). One agreement substrate, two result kinds.

### What counts as an item — and the scope we model
The test for "item" is **matter you can hold, own, move, and store** (a vessel of
value). By that test most of the natural world *qualifies* — wood, ore and other
**minerals**, stone, soil, water are all item-candidates. **Fire does not**: you
cannot hold fire; it is a *process/effect* (in the engine, arson is a `strike` on
a building, not a token). The holdable thing is the **fuel** (wood/charcoal — an
item); the fire is the *effect* of `use`-ing it — the same side of the line as
healing.

But the engine deliberately keeps the **alphabet of items small** — today just
`food`, `materials`, `tools`, `money` (+ `weapons`/`drugs` under `--society`).
Crucially, **`materials` folds together what are really distinct resources**:
forest→wood and mine→ore both yield the generic `materials`. Soil, stone, water
and specific minerals are not modelled at all. This is for determinism,
legibility and balance — every new item adds demand, recipes, prices and tuning.

So: enriching the natural-resource set (distinct **wood / ore / minerals /
stone / water**, with the recipes and professions they imply) is real and
on-brand, but a **deliberate, opt-in expansion — tracked under #21**, not a free
default. Fire and the like stay on the **effect/hazard** side, never items.

## Phased plan

### MVP — the thin vertical slice across #20 + #21 (do first)
The smallest change that makes the whole wheel turn once.

- **Jobs get distinct production.** farmer → more food/turn; miner → more
  materials; smith → materials→tools (value chain). Expressed as data
  (`affordances.py`) + a productivity factor in the gather/work handlers, gated
  by the economy layer.
- **Food becomes buyable.** A market sale converts `money → food`, which feeds
  through the existing `eat`. Supply = farmers' surplus offered for money; price
  emerges from swaps.
- **Brains choose buy-vs-make.** When busy / far from a farm / the farm is
  depleted, buying beats gathering. Heuristic: one simple opt-in rule (below);
  LLM: free choice via the affordance + price in `Observation`.
- **Observable emergence:** a smith stops farming, earns, and buys lunch; a
  farmer prospers by selling surplus → division of labour + money demand appear
  together.

### Phase 2 — services as labour, and depth of demand
Services are someone's *work*, so they are bought like goods — and, crucially,
**whether a service is free, fair, or extortionate is the provider's choice**,
not engine policy (the same stance as enforcement & corruption). So a service is
a **provider-posted offer on the existing OFFER/ACCEPT order book**:

> A provider *chooses* to post a service offer at a **price it picks** (0 =
> charity); a taker *accepts* (consent + pays). The fee moves to the provider
> (conserved); the service's **effect** is applied to the taker; the **price
> emerges** from accepted fees. Adding a service is a data entry in
> `market.SERVICES` + an effect handler in the simulation — **banks (deposit /
> loan), inns (lodging) and the like slot in the same way.**

- **[shipped]** Doctor **healing as a chosen service** — a doctor offers
  `{"service": "healing", "want_item": "money", "want_qty": P}`; whoever accepts
  pays `P` (→ the doctor, who earns; `P=0` is charity) and has their energy
  restored (produced like rest/food), more effectively at a hospital. Services
  are **local** (provider must be within reach). This is money's survival-grade
  demand: you can buy energy. Heuristic: a doctor offers care priced by
  temperament (cooperative → cheap/charitable, grasping → dear); a depleted,
  food-less patient accepts the cheapest affordable offer in reach. Richer play
  (haggling, refusing the poor, gouging the desperate) is the LLM's. Opt-in via
  `--economy`; baseline byte-identical.
- **Next services (same mechanism):** a **bank** (deposit for safe-keeping /
  interest; lending as a posted service) and an **inn** (paid rest → recovery).
- **Conspicuous consumption**: spend on a feast / patronage / commissioning →
  `esteem`/`reputation` (ties money to the dignity layer — the rich buy honour).

### Phase 3 — facility depth
- Market as a real buy/sell venue; **bank** as credit + storage; a **school**
  distinct from the library; etc. Each is a data edit in `affordances.py`
  (role + afforded actions), behaviour left to brains/council.

### Phase 4 — higher-order institutions (should emerge, not be hardcoded)
Once goods, services and a labour market exist, larger actors can form:

- **Firms / proprietors (事業主).** An agent with capital **hires** others' labour
  (pays a wage to have them gather/craft for it) and keeps the margin — a private
  organiser of production, distinct from the lone worker. Emerges from an
  *employ/pay* primitive, not a built-in "company".
- **The state (国家): taxation + labour levy.** The engine already has the seeds —
  `treasury`, a daily **civic levy** (tax), `--publicworks` construction, and an
  elected `mayor`/council. Extend toward the fuller picture: the state (or a
  dominant faction) **extracts taxes and labour** — a corvée / conscription that
  commands citizens' *turns*, not just their coin — to fund public works, war, or
  itself. Who is taxed, how hard, and whether it's legitimate or extractive
  becomes persona-/governance-differentiated (a Guardian commons vs a Predator
  shakedown).

These are large; they get their own issues once the MVP loop is proven.

## Decisions (resolved)

1. **Demand comes from specialisation of supply, on `--economy` alone.** Each
   profession produces its own good well and is **inefficient at off-specialty
   self-supply** (a non-farmer gathers food poorly), so buying from a specialist
   is the sensible path — giving food/materials a *structural* demand and giving
   producers buyers (hence a motive to make surplus). Survival is preserved:
   off-specialty gathering is a low-yield fallback, never zero, so no one starves
   to create demand. `--environment` amplifies but is not required. (This
   replaces the earlier "comparative advantage alone" framing, which left the
   free-gathering escape hatch open and so produced no real demand.)
2. **Heuristic stays minimal; rich economic play is the LLM's.** The heuristic
   gets one deterministic rule — *if hungry, not near a farm, and money ≥ meal
   price, buy; if you hold a surplus, sell it.* Arbitrage, stockpiling and price
   discovery beyond that are left to LLM brains (surfaced via affordances +
   prices in `Observation`). The heuristic exists for determinism + a free tier,
   not to be a clever trader.
3. **Tools are an exchange good in the MVP, not yet capital.** A smith makes
   tools and sells them; tools do **not** yet boost others' productivity. The
   capital-goods feedback (holding a tool makes you gather faster → investment,
   accumulation, inequality) is a named **Phase 2+** step, kept out of the MVP to
   bound scope and protect balance/determinism.
4. **No new flag — extend `--economy`.** Prices already emerge there; the labour/
   consumption layer rides on the same opt-in.

## Open questions (decide as we build)

- Exact productivity/price numbers so Decision&nbsp;1 holds (specialist wage ≫
  meal price) without making farming pointless for everyone.
- How firms (Phase 4) avoid degenerating into one agent owning everything — does
  competition for labour / wages emerge to balance it?
