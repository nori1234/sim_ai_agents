#!/usr/bin/env python3
"""Teacher agreement — an external, engine-side proxy for how BC-anchored a
trained checkpoint still is to the blind teacher, independent of any
internal training diagnostic.

Run #13's episode-boundary fix (S1) was ruled out and the reward ceiling
(S3) was answered, but S2 (is the policy still anchored to a regime-blind
teacher via behaviour cloning) is stuck: teacher_frac_in_batch never
appeared in the brain's own training-time diagnostics. This measures the
same question from OUTSIDE the training loop, on a frozen checkpoint,
without needing any brain-side instrumentation -- see
emergence/grounding.py's "Teacher agreement" section for the full argument.

No training, just inference against a checkpoint -- needs torch +
llm_model_agi, but is fast relative to a training run.

Usage
-----
    python3 scripts/teacher_agreement.py --checkpoint grounding_out/agent.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.grounding import measure_teacher_agreement  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                     help="path to a checkpoint saved by train_neural_grounding.py "
                          "(dev._dev.save_checkpoint)")
    ap.add_argument("--persona", default="guardian")
    ap.add_argument("--rule", default="demurrage")
    ap.add_argument("--seeds", default=None,
                     help="comma-separated seed list; default matches the "
                          "battery's 20 held-out worlds (42-61)")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--complexity-level", type=int, default=0)
    args = ap.parse_args()

    from emergence.brains.neural import NeuralDevelopmentalBrain

    def brain_factory(agent, persona, rng):
        return NeuralDevelopmentalBrain(persona, learn=False, checkpoint=args.checkpoint)

    kwargs = {}
    if args.seeds:
        kwargs["seeds"] = tuple(int(s) for s in args.seeds.split(","))

    result = measure_teacher_agreement(
        args.persona, rule=args.rule, days=args.days, n_agents=args.agents,
        complexity_level=args.complexity_level, brain_factory=brain_factory,
        **kwargs)

    d = result.as_dict()
    print(json.dumps(d, indent=2))
    print(f"\n[teacher-agreement] teacher deposit-recommendation rate: "
          f"control={d['teacher_deposit_rate_control']:.4f} "
          f"counterfactual={d['teacher_deposit_rate_counterfactual']:.4f} "
          f"(sanity check -- should read close to each other)")
    print(f"[teacher-agreement] policy agrees with teacher's deposit call: "
          f"control={d['agreement_control']:.4f} "
          f"counterfactual={d['agreement_counterfactual']:.4f} "
          f"gap={d['agreement_gap']:+.4f}")
    if d["agreement_gap"] > 0.05:
        print("[teacher-agreement] positive gap: the policy follows the "
              "teacher's (bad, under demurrage) advice measurably less often "
              "in the counterfactual world -- evidence of moving past the "
              "regime-blind BC anchor, independent of teacher_frac_in_batch.")
    else:
        print("[teacher-agreement] gap near zero: the policy is still "
              "equally anchored to the teacher's regime-blind rule in both "
              "worlds -- consistent with S2 (BC anchor) still being live.")


if __name__ == "__main__":
    main()
