# Working notes for Claude

## Merge workflow — MUST keep the dev branch synced to main

We ship via: branch → PR → **squash-merge** → resync. A squash-merge makes
GitHub author a *new* commit (committer `noreply@github.com`) that is **not** a
descendant of our pushed branch tip. If only the local branch is moved onto
main and the **remote dev branch is left behind**, that GitHub-authored commit
appears in `origin/<branch>..HEAD` and the Stop hook flags it as "Unverified" —
**on every merge**. Pushing the branch up after the fact silences it but is a
band-aid.

**Rule: the merge is not done until the dev branch is fast-forwarded to the
merged commit.** Treat it as one atomic operation:

1. Squash-merge the PR (GitHub MCP `merge_pull_request`, `merge_method: squash`).
2. Immediately run `scripts/sync-branch.sh` on the dev branch.

The script fetches main, `reset --keep origin/main`, force-with-lease pushes the
dev branch to the merged commit, and verifies `origin/<branch>..HEAD` is empty.
Never `--amend --reset-author` the merged commit — that rewrites published main
history. The GitHub squash commit is GPG-signed and shows **Verified** on
GitHub; the only issue is the stale local/remote branch range, which this fixes.

## Determinism baseline (don't break it)

The offline baseline must stay **byte-identical**: all opt-in layers
(`--economy`, `--environment`, `--society`, drives, library, individuals) OFF →
`tests/test_baseline_contract.py` (guardian→ORDER, philosopher→CHAOS,
idealist→COLLAPSE, predator→FAILURE) must pass unchanged. New mechanics go
behind a flag and are inert when off.
