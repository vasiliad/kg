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
  # Written by forge_autopilot.yml — the complex's own verdict about
  # itself, and the patch it proposed (the operator reads both).
  "data/forge_autopilot.json"
  "data/forge_autopilot"
  # Written by bot_selftest.yml — without them a rebase conflict on a
  # self-test report could not be resolved and the push would be lost.
  "data/bot_selftest_report.json"
  "data/bot_transport_report.json"
  # Written by emergency_cron.yml — the supervisor's own memory of what it
  # dispatched, and the silence-alarm marker.
  "data/.cycle_supervisor.json"
  "data/.bot_silence_alarm.json"
  # Written by cloud_tests.yml — the proof that the verification cycle ran.
  "data/cloud_tests_report.json"
)

# The append-only logs are NOT snapshots: both writers' lines are true, so a
# conflict is MERGED, never resolved by taking one side whole.
#
# Measured on 2026-11-02, from the repository's own history: the report's
# proof of delivery was committed at 05:24 (72c2543) and ERASED six minutes
# later by the fleet tuner's commit (770dd9c), whose workspace held an older
# copy of the log. The supervisor then read the newest report as 22.7 hours
# old, decided the daily cycle was owed, and sent the letter again — four
# times in one day. One overwritten log line became a flood of mail.
MERGED_FILES=(
  "nim_history.log"
  "data/telegram_history.txt"
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
    echo "[sync] conflict while rebasing — merging this run's state with the other writer's."
    # The append-only logs first: their conflicted stages must be read BEFORE
    # any checkout resolves them.
    for f in "${MERGED_FILES[@]}"; do
      mode="events"
      [ "$f" = "data/telegram_history.txt" ] && mode="conversation"
      if ! python scripts/merge_state_log.py --file "$f" --mode "$mode"; then
        echo "[sync] WARNING: ${f} could not be merged — this run's version wins there."
      fi
    done
    # The snapshot-like state files: the newest run wins.
    for f in "${CONFLICT_FILES[@]}"; do
      case " ${MERGED_FILES[*]} " in *" $f "*) continue ;; esac
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

# A failed state sync is NOT a quiet warning. Measured on 2026-11-02: the
# state push was refused, the helper printed a warning and exited 0, and the
# workflow reported SUCCESS while the cycle's evidence never reached the
# repository. The supervisor then read the missing evidence as "this cycle
# never ran" and repeated the work — the operator got the same daily letter
# four times. A loss that nobody can see is a loss that happens again.
echo "[sync] ERROR: the state could not be pushed after ${MAX_ATTEMPTS} attempts; this cycle's evidence is NOT in the repository."
exit 1
