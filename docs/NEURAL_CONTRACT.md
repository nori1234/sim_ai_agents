# Neural brain integration contract — v1.0

The versioned interface between the Emergence World engine (`sim_ai_agents`) and
an external developmental brain (`llm_model_agi`). This is the **single source of
truth** for the action vocabulary, observation schema and parameter conventions
the external adapter must target. The machine-readable form is
[`emergence/brains/neural_contract.py`](../emergence/brains/neural_contract.py)
— **import it, don't hard-code** — and the drift guards in
[`tests/test_neural_contract.py`](../tests/test_neural_contract.py) fail loudly if
the engine changes out from under it.

> **Versioning.** Bump `CONTRACT_VERSION` (minor = additive, major = breaking)
> whenever the vocabulary or schema changes. The adapter should record the
> version it was built against and check for a major mismatch.

## Who owns what

| Side | Owns |
|---|---|
| **Engine** (`sim_ai_agents`, this repo) | `NeuralDevelopmentalBrain` adapter, the survival reward (`_neural_reward.py`), this contract + its drift guards, the round-trip contract test |
| **Brain** (`llm_model_agi`) | the `agent` package: `DevelopmentalAgent` (HierMamba/policy/value/world-model/Titans/replay/L0–L3) **and** `agent.adapters.emergence`: `build_brain`, `to_engine_action`, `EmergenceObsTokenizer` |

The dependency points one way: the engine imports the brain (lazily, behind the
`[neural]` extra) and **never** the reverse. Absent the extra, the engine runs
unchanged on the heuristic.

## The interface the brain side provides

```
agent.adapters.emergence.build_brain(persona, teacher, checkpoint) -> DevelopmentalAgent
agent.adapters.emergence.to_engine_action(spec, agent, observation) -> emergence.actions.Action
agent.adapters.emergence.EmergenceObsTokenizer                       # observation -> token ids
```

`DevelopmentalAgent` must expose:

```
.act(observation) -> spec        # spec is the brain's own dict, mapped by to_engine_action
.learn(observation, reward)      # reward is a float from the engine side (see below)
```

Call order inside `NeuralDevelopmentalBrain.decide` (already implemented):

1. From turn 2 on: `reward = survival_reward(prev_obs, obs)` → `dev.learn(obs, reward)`
2. `spec = dev.act(obs)`
3. `action = to_engine_action(spec, agent, obs)` → returned to the engine

Any exception anywhere → the brain latches and degrades to the heuristic, so a
contract mismatch **cannot crash a run** (which is exactly why the round-trip
test below matters — silent fallback can otherwise mask a broken adapter).

## 1. The `Action` type — `target` lives in `params`

```python
Action(type: ActionType, params: dict = {}, rationale: str = "")
```

There is **no separate `target` argument**. `to_engine_action` builds
`Action(ActionType(spec_verb), params_dict)`. See `PARAM_SPEC` for each verb's
`params` shape; the engine clamps invalid/oversized params gracefully.

## 2. Action vocabulary (44)

The policy's output dimension maps onto `ACTION_VOCAB` (derived from the engine
enum, so it cannot drift). Categories:

* **survival/basics**: `idle move gather sow eat rest sleep work`
* **resources/commons**: `deposit_granary draw_granary transfer solicit`
* **governance**: `propose vote build collaborate speak praise create`
* **crime/enforcement**: `steal attack arson report_crime arrest`
* **society layer**: `craft_weapon deal_drug take_drug join_gang rebel preach worship`
* **economy**: `offer accept craft lend repay deposit withdraw endorse`
* **physical primitives** (institutions are read off these + context):
  `take give use strike make say bond`

## 3. Observation schema

* `observation.self_view` (= `Agent.snapshot()`) keys are frozen in
  `SELF_VIEW_KEYS`. The reward uses **`energy`** (0..100 float, survival),
  **`money`** (int, material) and **`reputation`** (social standing).
* **There is no "trust toward me" scalar.** `observation.others[i]["trust"]` is
  *this* agent's trust *of* neighbour `i`. The reward's social term therefore uses
  `reputation`, falling back to the mean of `others[*].trust` only when the status
  layer leaves reputation inert.
* Top-level fields are frozen in `OBSERVATION_FIELDS`. Layer dicts (`society`,
  `economy`, `environment`, …) are present but may be empty when the layer is off.

## 4. Reward (engine side, no torch)

`emergence/brains/_neural_reward.py`:

```
survival = wₑ·Δenergy + wₘ·Δmoney + w_s·Δsocial      # defaults: 0.02 / 0.05 / 0.10
```

Curiosity / prediction-error is added **inside** the brain against its
world-model; the engine supplies only this world-grounded extrinsic term. No
reward API is added to the engine — it is derived from the observation delta.

## 5. Persistence & memory

* One brain instance per agent, reused across **all** ticks of the agent's life
  (`Simulation.brains[agent_id]`), so learning state accumulates. Newborns get a
  fresh brain via `newborn_brain_factory`.
* The optional `memory_backend.TownMemory` only rewrites `observation.memory`.
  Keep Titans memory **internal** to the brain; don't write back to `TownMemory`
  (no double-management). They coexist without conflict.

## 6. Open question for the brain side — the teacher call

`build_brain` receives `teacher: AgentBrain` (e.g. an `LLMBrain`). To imitate it
the brain needs `teacher.decide(agent, obs) -> Action`, but `act(obs)` does not
receive `agent`. **Decide one:** (a) `act(observation, agent=None)` gains an
optional `agent`, or (b) the engine adapter holds `agent` and exposes a
`teacher_action(obs)` closure to the brain. The engine side can supply `agent`
either way — tell us which shape you want and we'll finalise the adapter.
