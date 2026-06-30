#!/usr/bin/env python3
"""Counterfactual-world transfer test — run the grounding falsification probe.

Runs two otherwise-identical towns: a CONTROL world where bank savings grow, and
a COUNTERFACTUAL world where they shrink (demurrage). Neither world states the
rule in the prompt, so an agent can only learn it by living through it. If the
behaviour (how often agents deposit) diverges between the worlds, that is
evidence the agent is *grounded* in consequence rather than replaying training.

See ``emergence/grounding.py`` and ``docs/GROUNDING.md`` for the full argument.

Examples
--------
Offline floor (heuristic brains — checks the instrument runs/conserves)::

    python3 scripts/grounding_probe.py --persona guardian --days 20

A real probe against a local Llama via Ollama::

    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1 \\
        python3 scripts/grounding_probe.py --persona claude --llm --days 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.grounding import run_grounding_probe  # noqa: E402


def _llm_factory(args):
    from emergence.brains.llm import LLMBrain

    base = args.llm_base or os.environ.get("LLM_BASE_URL")
    model = args.llm_model or os.environ.get("LLM_MODEL") or "llama3.1"
    key = args.llm_key or os.environ.get("LLM_API_KEY")
    provider = args.llm_provider

    def factory(agent, persona, rng):
        return LLMBrain(provider=provider, model=model, base_url=base,
                        api_key=key, persona=persona)

    return factory


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Counterfactual grounding probe")
    p.add_argument("--persona", default="claude", help="persona for the whole town")
    p.add_argument("--rule", default="demurrage", help="inverted world-rule to test")
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="divergence above this reads as grounded")
    p.add_argument("--llm", action="store_true", help="put agents on a real model")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base", default=None)
    p.add_argument("--llm-key", default=None)
    p.add_argument("--json", action="store_true", help="emit JSON only")
    args = p.parse_args(argv)

    brain_factory = _llm_factory(args) if args.llm else None
    result = run_grounding_probe(
        args.persona, rule=args.rule, days=args.days, n_agents=args.agents,
        seed=args.seed, threshold=args.threshold, brain_factory=brain_factory)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    d = result.as_dict()
    brain = "LLM" if args.llm else "heuristic (offline floor)"
    behaviour = d["target_behaviour"]
    print(f"Counterfactual transfer test — rule={d['rule']!r}  brain={brain}")
    print(f"  scored behaviour : {behaviour!r} events per agent-day")
    print(f"  control world    : {d['control_rate']:.4f}   (normal rule)")
    print(f"  counterfactual   : {d['counterfactual_rate']:.4f}   (rule inverted)")
    print(f"  divergence       : {d['divergence']:+.4f}   (control - counterfactual)")
    print(f"  heuristic floor  : {d['floor_divergence']:+.4f}   (mechanical, no learning)")
    print(f"  excess over floor: {d['excess']:+.4f}   ← the grounding signal")
    print(f"  verdict          : {d['verdict']}")
    if not args.llm:
        print("\nNote: with heuristic brains the tested brain IS the floor, so the "
              "excess is\nzero by construction. The non-zero divergence is purely "
              f"mechanical feedback\n(an inverted rule changes later {behaviour!r} "
              "choices). Run --llm for a real verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
