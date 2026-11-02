#!/usr/bin/env bash
# ==============================================================================
# Shared state sync for every cloud job that writes back to the repository.
#
# The bot daemon, the self-diagnostics and the heartbeat all commit to the same
# branch, so a push can legitimately be rejected because another job pushed
# while this one was running. A plain `git push` would then fail the whole
# workflow even though the cycle itself succeeded.
#
# This helper rebases onto the remote and retries. If two jobs touched the same
# regenerated state file, the newest run wins: these files are snapshots, not
# hand-edited documents.
# ==============================================================================
set -uo pipefail

TARGET_BRANCH="${TARGET_BRANCH:-main}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
CONFLICT_FILES=(
  "active_models.json"
  "nim_history.log"
  "config.yaml"
  "diagnostics_report.json"
  "data/telegram_history.txt"
  "data/.bot_sessions.json"
  "data/.bot_settings.json"
  "data/fleet_report.json"
  "data/fleet_history.json"
  "data/model_sizes.json"
  "data/daily_digest.json"
  "data/duty_shift.json"
)

git config --global user.name "vasiliad"
git config --global user.email "vasiliadi@users.noreply.github.com"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if git push origin "HEAD:${TARGET_BRANCH}" 2>&1; then
    echo "[sync] state pushed to ${TARGET_BRANCH} (attempt ${attempt})."
    exit 0
  fi

  echo "[sync] push rejected (attempt ${attempt}); rebasing onto origin/${TARGET_BRANCH}..."
  git fetch origin "${TARGET_BRANCH}" || true

  # `git rebase` refuses to run with unstaged changes. The self-test writes
  # data/.bot_sessions.json (its health block) and leaves it uncommitted, and
  # that blocked the rebase — the transcript was then never pushed, and the
  # cycle looked broken for a reason that had nothing to do with the cycle.
  # So the local modifications are stashed around the rebase and restored
  # afterwards; the committed snapshot stays the newest one either way.
  stash_before=$(git stash list 2>/dev/null | wc -l | tr -d ' ')
  git stash push -m "sync-state-autostash" >/dev/null 2>&1 || true
  stash_after=$(git stash list 2>/dev/null | wc -l | tr -d ' ')
  stash_ref=""
  if [ "$stash_after" -gt "$stash_before" ]; then
    stash_ref=$(git stash list --format="%gd" 2>/dev/null | head -1)
  fi

  if ! git rebase "origin/${TARGET_BRANCH}"; then
    echo "[sync] conflict while rebasing — keeping this run's state files."
    for f in "${CONFLICT_FILES[@]}"; do
      git checkout --theirs -- "$f" 2>/dev/null || true
    done
    # Only the KNOWN state files are staged. A blanket `git add -A` would
    # publish anything that happened to be lying in the workspace of a
    # PUBLIC repository — a stray download, a temporary file, a local .env.
    for f in "${CONFLICT_FILES[@]}"; do
      [ -e "$f" ] && git add -- "$f" 2>/dev/null || true
    done
    GIT_EDITOR=true git rebase --continue 2>&1 || git rebase --abort 2>&1 || true
  fi

  if [ -n "$stash_ref" ]; then
    if git stash pop "$stash_ref" >/dev/null 2>&1; then
      echo "[sync] local changes restored after the rebase."
    else
      # The stash did not apply cleanly: the committed snapshot wins, and the
      # stash is kept so the next cycle can still read it.
      git diff --name-only --diff-filter=U 2>/dev/null | while read -r f; do
        git checkout HEAD -- "$f" 2>/dev/null || true
      done
      echo "[sync] local changes could not be re-applied; the committed snapshot wins."
    fi
  fi
done

# A failed state sync must not mark the whole cycle as failed: the next
# scheduled run re-syncs, and the cycle's own work already succeeded.
echo "[sync] WARNING: state could not be pushed after ${MAX_ATTEMPTS} attempts; the next cycle retries."
exit 0
