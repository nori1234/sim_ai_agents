# Grounding probe — the counterfactual-world transfer test

> *Is the agent grounded in this world's consequences, or replaying its training?*

This is the project's first **validation instrument**, not a new mechanic. It
exists to answer the question the panel (#118) kept returning to: when an LLM
agent does something sensible here — saving in a bank, repaying a loan, taking
shelter — is that behaviour **grounded** in the consequences it has lived
through, or is it **replaying** a pattern memorised from training data? In a
world whose rules already match the training prior, the two are
indistinguishable, so success proves nothing.

## The design

We break the prior. A **counterfactual world** inverts one existing rule into
something the model has (almost) never read, and we never state the new rule in
the prompt — the agent can only discover it by living through its consequences.

The first rule is **demurrage**: money left in a bank *shrinks* (a negative
interest rate) instead of growing. This contradicts one of the strongest priors
in any text corpus ("saving grows your money").

We then run two otherwise-identical towns (same seed, persona, layers) and score
how often agents perform the punished behaviour (here, `deposit`), normalised per
agent-day:

| world | rule | a grounded agent | a replaying agent |
|---|---|---|---|
| **control** | savings grow | deposits normally | deposits normally |
| **counterfactual** | savings shrink | deposits **less** | deposits the same |

The **divergence** between the worlds (`control − counterfactual`) is the signal.
A grounded agent, having watched its coin evaporate, pulls back; a replaying
agent behaves the same in both.

## Why a raw divergence is not enough — the heuristic floor

A positive divergence is **not by itself** evidence of grounding. Even the
offline heuristic brain — which cannot learn anything — diverges, because
shrinking deposits mechanically feed back into later choices (smaller balances →
fewer follow-on deposits). Running the probe offline shows this floor directly:

```
$ python3 scripts/grounding_probe.py --persona guardian --days 20
  divergence       : +0.0583   (control - counterfactual)
  heuristic floor  : +0.0583   (mechanical, no learning)
  excess over floor: +0.0000   ← the grounding signal
  verdict          : baseline (heuristic floor)
```

So the probe measures the tested brain's divergence **against the heuristic
floor** and credits only the **excess**. Grounding is the excess of an LLM's
divergence over what the engine produces mechanically with no learning at all.

## Running it

```bash
# Offline floor (checks the instrument runs and conserves; excess is 0 by design)
python3 scripts/grounding_probe.py --persona guardian --days 20

# A real probe against a local Llama via Ollama
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1 \
    python3 scripts/grounding_probe.py --persona claude --llm --days 20

# Programmatic
from emergence.grounding import run_grounding_probe
result = run_grounding_probe("claude", rule="demurrage", days=20)
print(result.as_dict())   # control_rate, counterfactual_rate, divergence, floor, excess, verdict
```

## What it is and is not

* It **is** a falsifiable test: a model that only replays will score ~0 excess,
  and that is a real, reportable negative result.
* It is **inert when off**. `CounterfactualConfig` defaults to disabled with the
  rate advertised as usual, so the determinism baseline
  (`tests/test_baseline_contract.py`) is byte-identical.
* It is **conserved**: demurrage shrinks the depositor's *claim* and the bank's
  liability by the same amount — no coin is minted or burned.
* It is **one rule, one behaviour** so far. The natural next rules (each a new
  entry in `_TARGET_EVENT`) are ones equally absent from training: *gifts lower
  status*, *lying is visible*, *hoarding spoils*. Each gives an independent
  transfer test, and agreement across several is far stronger evidence than any
  one.

## Why this comes before 3D

The reframed north star is grounding and agency as a contribution to AGI. The
panel's consensus (#118) was that *embodiment in 3D is the weakest premise* —
the core of grounding is irreversible causal consequence, which the deterministic
engine already has. Before investing in a 3D client, we need to be able to
**measure** whether agents are grounded at all. This instrument is that
measurement; 3D can come later as a read-only view if the signal warrants it.
