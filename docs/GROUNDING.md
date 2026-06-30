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
The rules are registered in `_RULES` (each pairs the inverted law with the
behaviour whose frequency is scored, and the engine layers it needs):

| rule | inverted law (vs the prior) | scored behaviour | needs |
|---|---|---|---|
| **demurrage** | money left in a bank *shrinks* — negative interest — instead of growing (prior: "saving grows your money") | `deposit` | economy |
| **vanity** | hosting a lavish feast *lowers* the host's standing instead of buying honour (prior: conspicuous display buys status) | `feast` | status |

Agreement across several independent rules is far stronger evidence of grounding
than any single one — a replayer might fluke one, not all.

We then run two otherwise-identical towns (same seed, persona, layers) and score
how often agents perform the punished behaviour, normalised per agent-day. For
`demurrage`:

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

## Tracking grounding *over training* — `GroundingMonitor`

A single probe scores a snapshot. To watch grounding **emerge as a developmental
brain learns**, `emergence.grounding_monitor.GroundingMonitor` runs the probe on a
cadence and keeps the `excess` time series — the curve the `llm_model_agi`
learning loop logs:

```python
from emergence.grounding_monitor import GroundingMonitor

monitor = GroundingMonitor(persona="claude", every=10, on_result=trainer.log)
for epoch in range(n_epochs):
    train_one_epoch(...)
    # brain_factory must yield brains with the CURRENT trained weights
    monitor.maybe_probe(epoch, brain_factory=current_brain_factory)

monitor.to_jsonl("grounding.jsonl")
monitor.improving()    # did excess trend up over training?
```

The probe runs its own fresh simulations (it never touches the training run), and
the logged headline is `excess`, never the raw divergence.

## What it is and is not

* It **is** a falsifiable test: a model that only replays will score ~0 excess,
  and that is a real, reportable negative result.
* It is **inert when off**. `CounterfactualConfig` defaults to disabled with the
  rate advertised as usual, so the determinism baseline
  (`tests/test_baseline_contract.py`) is byte-identical.
* It **respects the engine's primitives**: `demurrage` is conserved (it shrinks
  the depositor's *claim* and the bank's liability by the same amount — no coin
  minted or burned); `vanity` adjusts the host's reputation (a non-conserved
  status score, like any honour change), it does not touch coin.
* It is **two rules so far** (`demurrage`, `vanity`), each a new entry in
  `_RULES`. Further rules equally absent from training — *lying is visible*,
  *hoarding spoils* — are cheap to add (invert one existing mechanic, register the
  scored behaviour and the layers it needs). Agreement across several independent
  rules is far stronger evidence of grounding than any one.

## Why this comes before 3D

The reframed north star is grounding and agency as a contribution to AGI. The
panel's consensus (#118) was that *embodiment in 3D is the weakest premise* —
the core of grounding is irreversible causal consequence, which the deterministic
engine already has. Before investing in a 3D client, we need to be able to
**measure** whether agents are grounded at all. This instrument is that
measurement; 3D can come later as a read-only view if the signal warrants it.
