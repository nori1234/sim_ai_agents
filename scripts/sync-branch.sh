#!/bin/bash
# Post-merge branch sync — run IMMEDIATELY after squash-merging a PR.
#
# Why this exists:
#   A squash-merge makes GitHub author a brand-new commit (committer
#   noreply@github.com) that is NOT a descendant of our pushed branch tip.
#   If we resync only the local branch to main and leave the remote dev
#   branch behind, that GitHub-authored commit shows up in
#   `origin/<branch>..HEAD` and the Stop hook flags it as "Unverified" — on
#   EVERY merge. The fix is to keep the remote dev branch fast-forwarded to
#   the merged main, so that range is always empty. Do it as one atomic step
#   with the merge, never reactively after the hook complains.
#
# Usage: scripts/sync-branch.sh   (run on the dev branch, right after merge)
set -euo pipefail

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo "detached HEAD — checkout the dev branch first" >&2; exit 1
fi
if [[ "$branch" == "main" ]]; then
  echo "on main — nothing to sync" >&2; exit 0
fi

# Pull the just-merged main down and move the local branch onto it (keeps the
# working tree; aborts if there are uncommitted local changes).
git fetch origin main --quiet
git reset --keep origin/main

# Fast-forward the remote dev branch to the merged commit so
# origin/<branch>..HEAD is empty and the Stop hook has nothing to flag.
for i in 1 2 3 4; do
  git push --force-with-lease origin "HEAD:$branch" && break || sleep $((2**i))
done

# Verify the range is empty — the whole point of this script.
if [[ -n "$(git log --oneline "origin/$branch..HEAD")" ]]; then
  echo "WARNING: origin/$branch..HEAD is still non-empty after sync" >&2; exit 1
fi
echo "synced: origin/$branch == $(git rev-parse --short HEAD) (range clean)"
