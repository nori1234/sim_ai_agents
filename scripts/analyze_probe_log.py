#!/usr/bin/env python3
"""Analyze the S4 probe fields in a neural-train-battery driver log.

Run #16 onward, the brain reports per-batch credit diagnostics for the DEPOSIT
action (`probe_*` fields in the learn-info dict the driver prints every
episode — see llm_model_agi `AgentConfig.probe_verb`). This script parses the
archived `full_log.txt`, reconstructs the trajectory, and joins the per-batch
episode SEGMENTS to regimes via the driver's own alternation rule
(`cf_enabled = (ep // block) % 2 == 1`, ep 0-based; the printed episode number
is ep+1, and `episodes_seen` in the same dict identifies the batch's last
segment's episode).

Usage:
    python3 scripts/analyze_probe_log.py docs/runs/run-16/full_log.txt
"""

from __future__ import annotations

import ast
import re
import sys


def parse(path: str, block: int = 1):
    updates = []   # one entry per line that carries a fresh batch update
    seen = set()
    pat = re.compile(r"\[train\] episode (\d+)/(\d+) done \| (\{.*\})\s*$")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        ep_printed = int(m.group(1))
        try:
            info = ast.literal_eval(m.group(3))
        except (ValueError, SyntaxError):
            continue
        gs = info.get("grad_steps")
        row = {"episode": ep_printed, **info}
        # the driver reprints the last batch's info between updates; keep the
        # FIRST line that shows each grad_steps value (the fresh update)
        if gs is not None and gs in seen:
            continue
        if gs is not None:
            seen.add(gs)
        updates.append(row)
    return updates


def regime_of_episode(ep_1based: int, block: int = 1) -> str:
    return "cf" if ((ep_1based - 1) // block) % 2 == 1 else "control"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/runs/run-16/full_log.txt"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ups = parse(path, block)
    has_probe = [u for u in ups if "probe_prob_mean" in u]
    print(f"parsed {len(ups)} fresh batch updates, {len(has_probe)} with probe fields\n")
    if not has_probe:
        print("no probe fields found (pre-instrumentation run?)")
        return

    print(f"{'ep':>4} {'grad':>4} {'tfrac':>6} {'p(dep)':>7} {'adv_dep_raw':>11} "
          f"{'adv_non_raw':>11} {'G_dep':>8} {'self_n':>6} {'teach_n':>7}")
    for u in has_probe:
        f = lambda k: ("None" if u.get(k) is None else f"{u[k]:+.4f}")
        print(f"{u['episode']:>4} {u.get('grad_steps',0):>4} "
              f"{u.get('teacher_frac_in_batch',0):>6.3f} {u['probe_prob_mean']:>7.4f} "
              f"{f('probe_adv_raw_mean'):>11} {f('nonprobe_adv_raw_mean'):>11} "
              f"{f('probe_G_mean'):>8} {u['probe_self_n']:>6} {u['probe_teacher_n']:>7}")

    # aggregates
    def vals(k):
        return [u[k] for u in has_probe if u.get(k) is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n== aggregates over all batches with data ==")
    print(f"probe_prob_mean:      first 5 avg {mean([u['probe_prob_mean'] for u in has_probe[:5]]):.4f}"
          f"  last 5 avg {mean([u['probe_prob_mean'] for u in has_probe[-5:]]):.4f}")
    print(f"probe_adv_raw_mean:   {mean(vals('probe_adv_raw_mean')):+.4f}  (n={len(vals('probe_adv_raw_mean'))})")
    print(f"probe_adv_used_mean:  {mean(vals('probe_adv_used_mean')):+.4f}  (n={len(vals('probe_adv_used_mean'))})")
    print(f"nonprobe_adv_raw_mean:{mean(vals('nonprobe_adv_raw_mean')):+.4f}  (n={len(vals('nonprobe_adv_raw_mean'))})")
    print(f"probe_self_n total:   {sum(u['probe_self_n'] for u in has_probe)}")
    print(f"probe_teacher_n total:{sum(u['probe_teacher_n'] for u in has_probe)}")

    # segment-level regime join: last segment belongs to episodes_seen, earlier
    # segments count backwards one episode each (regime alternates per `block`)
    by_regime = {"control": [], "cf": []}
    dep_credit = {"control": [], "cf": []}
    for u in has_probe:
        segs = u.get("probe_segments") or []
        last_ep = u.get("episodes_seen")
        if last_ep is None:
            continue
        for i, s in enumerate(segs):
            ep = last_ep - (len(segs) - 1 - i)
            if ep < 1:
                continue
            r = regime_of_episode(ep, block)
            by_regime[r].append(s["probe_prob_mean"])
            if s.get("probe_adv_raw_mean") is not None:
                dep_credit[r].append(s["probe_adv_raw_mean"])
    print("\n== segment-level, regime-joined ==")
    for r in ("control", "cf"):
        print(f"{r:>8}: propensity mean {mean(by_regime[r]):.4f} (n={len(by_regime[r])})"
              f" | deposit raw-credit mean "
              f"{mean(dep_credit[r]):+.4f} (n={len(dep_credit[r])})" if dep_credit[r] else
              f"{r:>8}: propensity mean {mean(by_regime[r]):.4f} (n={len(by_regime[r])})"
              f" | deposit raw-credit: no self-attempted deposits in these segments")


if __name__ == "__main__":
    main()
