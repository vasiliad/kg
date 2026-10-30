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

  if ! git rebase "origin/${TARGET_BRANCH}"; then
    echo "[sync] conflict while rebasing — keeping this run's state files."
    for f in "${CONFLICT_FILES[@]}"; do
      git checkout --theirs -- "$f" 2>/dev/null || true
    done
    git add -A || true
    GIT_EDITOR=true git rebase --continue 2>&1 || git rebase --abort 2>&1 || true
  fi
done

# A failed state sync must not mark the whole cycle as failed: the next
# scheduled run re-syncs, and the cycle's own work already succeeded.
echo "[sync] WARNING: state could not be pushed after ${MAX_ATTEMPTS} attempts; the next cycle retries."
exit 0
