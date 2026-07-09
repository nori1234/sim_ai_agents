#!/usr/bin/env python3
"""Generate paired control/counterfactual Observation snapshots for the brain
team's supervised regime-decoding probe (``scripts/probe_regime_readout.py``
in ``llm_model_agi``, commit 7b40a93+ — see docs/GROUNDING.md, "Run #11").

The probe asks a narrower question than the acceptance battery: not "does RL
learn to act on the regime" but "does a single observation snapshot even make
the regime linearly decodable at all". Answering that needs raw, labelled
(obs, regime) pairs, not a trained policy's behaviour — so this script drives
the SAME sandbox worlds the battery uses with the non-learning heuristic
brain (which reliably deposits, per the preflight's 20/20 conclusive yield)
and records the exact ``Observation`` the agent's ``decide()`` would receive,
once per day (the heuristic/battery's own decision cadence), for both regimes
of every held-out battery world.

Rows are paired by ``(world_seed, obs["day"])``: the control and
counterfactual snapshot for the same world and day are adjacent in the
output, so the only thing that differs between a pair is the regime lived
through — not the world or the point in the episode. This is what lets the
probe attribute a readout difference to the regime rather than to a
confound.

``_observe`` (``emergence/simulation.py``) is a pure read of world state (no
RNG draws, no mutation — see its body) — calling it once more after each
``step_day()`` to snapshot the deposit-facing agent does not perturb the
run's determinism.

Usage
-----
    python3 scripts/generate_probe_pairs.py --out probe_pairs.jsonl

20 worlds (BATTERY_SEEDS) x 2 regimes x `--days` (default 20) timepoints
= 800 rows by default, matching the brain team's requested 400-1000 range.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emergence.grounding import make_grounding_sandbox  # noqa: E402
from train_neural_grounding import BATTERY_SEEDS  # noqa: E402


def _day_snapshots(sim, agent) -> list[dict]:
    """Run ``sim`` to completion, returning ``agent``'s Observation dict at
    the end of each day (after that day's demurrage/interest has posted —
    see ``_end_of_day`` -> ``_pay_deposit_interest``/``_apply_demurrage`` in
    ``emergence/simulation.py`` — so the snapshot reflects a consequence the
    agent has actually lived through, not just the regime label)."""
    snapshots = []
    running = True
    while running:
        running = sim.step_day()
        if agent.alive:
            snapshots.append(dataclasses.asdict(sim._observe(agent)))
    return snapshots


def generate(persona: str, seeds, days: int, n_agents: int, out_path: Path) -> int:
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            for regime, cf_enabled in (("control", False), ("counterfactual", True)):
                sim = make_grounding_sandbox(
                    persona, n_savers=n_agents - 1, seed=seed, days=days,
                    cf_enabled=cf_enabled, brain_factory=None)
                # agents[0] is the banker (never deposits — see
                # _prepare_sandbox); the first SAVER is the depositor whose
                # observation actually carries the scored decision.
                agent = sim.agents[1]
                for obs in _day_snapshots(sim, agent):
                    fh.write(json.dumps(
                        {"obs": obs, "regime": regime, "world_seed": seed}) + "\n")
                    n_rows += 1
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", default="guardian",
                     help="matches the battery's default persona (exercises all rules)")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("probe_pairs.jsonl"))
    args = ap.parse_args()

    n_rows = generate(args.persona, BATTERY_SEEDS, args.days, args.agents, args.out)
    print(f"[probe-data] wrote {n_rows} rows "
          f"({len(BATTERY_SEEDS)} worlds x 2 regimes x {args.days} days) to {args.out}")


if __name__ == "__main__":
    main()
