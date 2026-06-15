# Principled Migration — making the baseline obey its own principle

## The principle

> We are building the **engine** (physical laws), not the **institutions**.
> Money, markets, police, religion, gangs — these should *emerge* from
> primitives, not be hardcoded as privileged objects.

The opt-in economy layer (`economy=True`) already follows this: it adds no
institutions, only physics — conserved items, `OFFER`/`ACCEPT` (barter →
emergent prices), `CRAFT` (recipes), `LEND`/`REPAY` (credit → trust). Prices
and credit are *consequences*, not constants.

The **baseline**, however, still ships several institutions baked in. This
document is the plan to dissolve them into primitives **without losing the
four societies** the baseline is famous for reproducing.

## The contract (what may not regress)

`tests/test_baseline_contract.py` pins the four qualitative endings:

| archetype   | model  | society it must keep producing |
|-------------|--------|--------------------------------|
| guardian    | Claude | **ORDER** — cooperative, conformist |
| philosopher | Gemini | **CHAOS** — pervasive crime |
| idealist    | GPT    | **COLLAPSE** — idealism that fails to sustain |
| predator    | Grok   | **FAILURE** — predation that eats its base |

The qualitative verdicts are durable. The numeric snapshot in the same file
tracks drift: a phase may move it, but only deliberately and with the change
committed alongside a justification.

**Acceptance criterion for every phase:** the four endings still emerge, and
they remain distinct from one another.

## The institutions to dissolve

| # | institution (hardcoded) | the magic | becomes (primitive) |
|---|-------------------------|-----------|---------------------|
| 1 | `Agent.money` — a privileged scalar field | money exists by fiat | an item in inventory (`coin`); a medium that *emerges* as the common want |
| 2 | police-aura (`_deterred`) — crime suppressed by proximity to a building | a building radiates lawfulness | enforcement is an *act* by an agent (a guard who can `ARREST`); buildings only host the role |
| 3 | law-keyword magic (`crime_deterrence_multiplier`) | passing a law named "police" globally lowers crime | a law is a *norm*: a published expectation + an enforcer who acts on it + compliance that agents weigh |

## Phases

- **Phase 0 — safety net (this commit).** Lock the contract: qualitative
  verdicts + numeric snapshot, before touching any mechanic. Purely additive.
- **Phase 1 — money → item.** Make `Agent.money` a property over
  `inventory["coin"]`. Coin stays conserved by the same transfer physics as
  food/materials. No agent gets coin for free that it didn't get before.
- **Phase 2 — police-aura → enforcement.** Introduce an `ARREST` action an
  agent in a guard role can take against a witnessed crime; remove the
  building-proximity `_deterred` aura. Order must now be *enforced*, not
  *radiated*.
- **Phase 3 — law-magic → norms.** Replace `crime_deterrence_multiplier`
  with a published norm that agents comply with in proportion to expected
  enforcement and their persona. Remove the global multiplier.
- **Phase 4 — re-tune & re-document.** Re-tune personas so the four endings
  re-emerge from the new primitives; rewrite the brittle numeric assertions
  as qualitative ones where appropriate; update the README's claims so they
  describe an engine, not a set of institutions.

## How to run the safety net

```
python -m unittest tests.test_baseline_contract -v
```

Run it before and after every phase. Green qualitative contract = the world
still produces its four societies. Snapshot diffs are reviewed, not feared.
