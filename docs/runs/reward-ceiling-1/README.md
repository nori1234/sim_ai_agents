# reward-ceiling-1

Reproduce with:

    python3 scripts/reward_ceiling.py --persona guardian

Ran locally (no torch, no CI needed -- deterministic, ~6 seconds). See
`emergence/grounding.py`'s "Reward ceiling" section and `output.txt` for the
exact command output (`measure_reward_ceiling`'s `as_dict()` plus the two
summary lines).

Requested by the brain team ahead of run #13, as experiment "#9": before
spending training compute on episode-boundary + BC-anchor fixes, check
whether the task itself pays enough for grounding to be worth learning at
all.
