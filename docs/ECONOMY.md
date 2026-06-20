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

## The loop we want to *emerge*

```
specialised work produces a surplus → sell it for money → buy what you DON'T
produce (and the needs you can't cheaply meet yourself)
        ↑                                                        ↓
   division of labour  ←————  money gains real demand  ←——  food / rest / healing
                                                              / status are buyable
```

The two issues are the two arrows of the same wheel: demand for money (#20) only
becomes real once jobs specialise (#21), and specialisation only pays once money
buys what you lack (#20).

## Why money gains demand (the mechanism)

Not "because we say so" — from **opportunity cost**. An agent has one action per
turn and must travel to a farm to gather. An agent doing high-value work (a smith
turning materials into tools worth more than a meal) is **better off buying food
than spending turns farming** — classic comparative advantage. So it specialises,
sells its surplus, and buys the rest. `--environment` (depletion, winter)
sharpens this: when free gathering fails, the market — stocked by farmers'
surplus — still sells, at an emergent price.

## Design principles (keep the engine's stance)

1. **Emergence-first.** Don't script a "shop". Add **affordances** ("you can buy
   food here") and let brains choose. Prices are **not** a formula — they already
   emerge from accepted `OFFER`/`ACCEPT` swaps (`market.py`, `--economy`).
2. **Money is not privileged** (per `PRINCIPLED_MIGRATION.md`). Consumption is
   just `money → good → use`, built from existing primitives.
3. **Determinism.** The `--compare` baseline stays byte-identical. Everything
   here is **opt-in** (rides on `--economy`); `test_baseline_contract.py` is
   untouched. New signals ride on `Observation`; heuristic branches are gated.

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
  depleted, buying beats gathering. Heuristic: an opt-in branch; LLM: free choice
  via the affordance + price in `Observation`.
- **Observable emergence:** a smith stops farming, earns, and buys lunch; a
  farmer prospers by selling surplus → division of labour + money demand appear
  together.

### Phase 2 — depth of demand
- Doctor **healing as a paid service** (`money → energy`).
- An inn / better rest (`money → recovery efficiency`).
- **Conspicuous consumption**: spend on a feast / patronage / commissioning →
  `esteem`/`reputation` (ties money to the dignity layer — the rich buy honour).

### Phase 3 — facility depth
- Market as a real buy/sell venue; **bank** as credit + storage; a **school**
  distinct from the library; etc. Each is a data edit in `affordances.py`
  (role + afforded actions), behaviour left to brains/council.

## Determinism guardrails

- All new logic lives under `--economy` (or a new `--market`); the default is
  unchanged.
- The four-society contract numbers stay fixed; add new tests for the new layer
  rather than perturbing the baseline (mirror `test_library.py`'s "on == off"
  guard where it applies).

## Open questions (decide as we build)

- Strength of the opportunity-cost lever without `--environment` — does free
  gathering still dominate when food is abundant? (May make the MVP recommend
  `--economy --environment` together.)
- How much should the heuristic brain participate vs. leaving rich economic play
  to LLM brains? (Keep heuristic minimal; it exists to keep determinism + a free
  tier, not to be a good trader.)
- Whether "tools" should feed back into productivity (a smith's tools make others
  gather faster) — a natural but larger capital-goods step; likely Phase 2+.
