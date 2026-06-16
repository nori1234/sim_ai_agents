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
| 1 | `Agent.money` — a privileged scalar field | money exists by fiat | an item in inventory (`inventory["money"]`); a medium that *emerges* as the common want |
| 2 | police-aura (`_deterred`) — crime suppressed by proximity to a building | a building radiates lawfulness | enforcement is an *act* by an agent (a guard who can `ARREST`); buildings only host the role |
| 3 | law-keyword magic (`crime_deterrence_multiplier`) | passing a law named "police" globally lowers crime | a law is a *norm*: a published expectation + an enforcer who acts on it + compliance that agents weigh |

## Phases

- **Phase 0 — safety net (this commit).** Lock the contract: qualitative
  verdicts + numeric snapshot, before touching any mechanic. Purely additive.
- **Phase 1 — money → item. (DONE)** `Agent.money` is now a property over
  `inventory["money"]`, conserved by the same add/take physics as
  food/materials; the `money=` constructor argument still works and is folded
  into the inventory. Strictly representational — the Phase 0 snapshot is
  byte-identical. One latent quirk surfaced: theft's `take("money")` used to
  hit an empty, *separate* slot and net nothing, so theft only ever moved
  food. Now that money is in the inventory the same call would drain real
  coin, which tips the predator society from FAILURE into full COLLAPSE.
  Looting coin is therefore deferred to the re-tune phase, and theft keeps its
  historical (food-only) behaviour for now.
- **Phase 2 — police-aura → enforcement. (DONE)** The building-proximity
  `_deterred` aura (and `_nearest_deterrent`) are gone. Crime is now punished
  by an `ARREST` *act*: a recent offender is "wanted" for a short window, and
  an agent — the guard role, in offline runs — pursues and detains them
  (fine + energy cost + hauled to prison if one exists). Buildings only host
  the role; order is enforced, not radiated. The four endings hold and stay
  distinct; crime counts rose (e.g. gemini 133 -> 211) because crimes now
  happen and are punished reactively rather than being magically pre-empted,
  and the snapshot was updated to match. `crime_deterrence_multiplier` is now
  unused by the engine and falls in Phase 3.
- **Phase 3 — law-magic → norms. (DONE)** `crime_deterrence_multiplier` is
  gone. Enacting a crime law no longer flips a global knob; it publishes a
  *norm* (`PolicyEngine.has_crime_norm`), surfaced to agents via
  `Observation.norms` together with an *enforcement expectation* derived from
  real world state (living guards + the facilities that host them, not a
  constant). The brain's compliance check (`_norm_restrains`) then has each
  agent abstain from a crime in proportion to its conformity times that
  enforcement credibility: a law-abiding agent keeps the peace by choice, a
  low-conformity one flouts it, and a norm nobody enforces deters no one.
  Suppression thus emerges from agents weighing the rule, not from the engine
  rewriting crime probabilities. The four endings hold; gemini crime falls
  211 -> 151 as compliance re-enters, and the snapshot is updated to match.
  (The remaining law effects — punishment fines, tax, food redistribution —
  are economic-policy actions of the state rather than crime magic; they are
  out of this phase's scope.)
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
