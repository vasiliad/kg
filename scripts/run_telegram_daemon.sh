#!/usr/bin/env bash
# ==============================================================================
# Telegram Bot Daemon Runner (cloud-native, GitHub Actions).
#
# Runs the bot as a background process for a bounded window, periodically
# persisting sessions/memory/state back into the repository so that the
# authorised chat, the conversation log and the memory vault survive the
# runner filesystem being wiped between jobs.
#
# THE DAEMON REPORTS THE BOT'S DEATH (audited 2026-11-04):
#   the bot's exit status used to be thrown away, so a bot that died on
#   startup — a revoked token, a missing secret, an import error — left a
#   GREEN run behind. Every window "succeeded" while the bot was dead, and
#   the chain kept dispatching dead windows. The daemon now waits for the
#   bot's real exit status and fails the window when the bot died on its own.
#   A window that merely reached its time limit is a NORMAL end: the bot is
#   still alive then, and the daemon stops it deliberately.
#
# Env:
#   DURATION_MINUTES  how long the bot stays alive (default 25)
#   COMMIT_INTERVAL   seconds between state syncs (default 180)
# ==============================================================================
set -uo pipefail

DURATION_MINUTES="${DURATION_MINUTES:-25}"
COMMIT_INTERVAL="${COMMIT_INTERVAL:-180}"

git config --global user.name "vasiliad"
git config --global user.email "vasiliadi@users.noreply.github.com"
git config --global --add safe.directory "${GITHUB_WORKSPACE:-$(pwd)}"

STATE_FILES=(
  "data/.bot_sessions.json"
  "data/telegram_history.txt"
  "active_models.json"
  "data/fleet_report.json"
  # The duty shift outlives the 50-minute runner window: without this file
  # the next window would pick a new model every time.
  "data/duty_shift.json"
  # The supervisor's own memory of what it dispatched. Without it in the
  # repository the next window cannot know that this one already tried —
  # and "GitHub did not answer" would be read as "idle" all over again.
  "data/.cycle_supervisor.json"
  "nim_history.log"
  # NOT `data/.bot_settings.json`: it holds the operator's address, so it
  # must NEVER be committed (the secret `DISPATCH_RECIPIENT_EMAIL` is the
  # source of truth). It used to be in this list — and `git add` with a
  # path that .gitignore rejects aborts the WHOLE command, so one private
  # file would have silently stopped every state sync.
)

sync_state() {
  local staged=0
  # Each path is added on its own: `git add` with several paths aborts
  # entirely when one pathspec does not match (measured 2026-11-03), and a
  # single rejected path would take the whole state sync with it.
  for f in "${STATE_FILES[@]}"; do
    [ -e "$f" ] || continue
    git add -- "$f" 2>/dev/null || true
    staged=1
  done
  [ "$staged" -eq 0 ] && return 0

  if ! git diff --staged --quiet 2>/dev/null; then
    if git commit -m "chore(bot): persist session and memory state [skip ci]" >/dev/null 2>&1; then
      # Shared helper: rebases and retries, because the heartbeat and the
      # diagnostics job push to the same branch concurrently.
      bash "${GITHUB_WORKSPACE:-$(pwd)}/scripts/sync_state.sh" 2>&1 | sed 's/^/[daemon] /'
    fi
  fi
}

# The bot's real exit status (bash keeps it until it is waited for).
# NOTE: `wait` must NOT be called inside a command substitution — that would
# run it in a subshell, where the bot is not a child and `wait` would answer
# 127 instead of the bot's real status.
BOT_PID=""
BOT_EXIT=0
BOT_DIED=0
cleanup() {
  echo "[daemon] shutdown requested — stopping bot..."
  if [ -n "$BOT_PID" ] && kill -0 "$BOT_PID" 2>/dev/null; then
    kill -TERM "$BOT_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$BOT_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$BOT_PID" 2>/dev/null || true
  fi
  sync_state
  echo "[daemon] final state sync complete."
}
trap cleanup EXIT TERM INT

echo "[daemon] Starting Telegram bot for ${DURATION_MINUTES} minutes..."
# Unbuffered: the runner's stdout is not a TTY, so Python would otherwise
# buffer the log until exit and hide the bot's live state.
python -u harness/telegram_bot.py &
BOT_PID=$!

# The first liveness probe is short: a bot that cannot even start (a revoked
# token, a missing secret, an import error) must be reported in seconds, not
# after the whole commit interval.
sleep 5
if ! kill -0 "$BOT_PID" 2>/dev/null; then
  wait "$BOT_PID" 2>/dev/null
  BOT_EXIT=$?
  BOT_DIED=1
  echo "[daemon] ERROR: the bot died immediately (exit status ${BOT_EXIT})."
else
  ELAPSED=0
  while kill -0 "$BOT_PID" 2>/dev/null && [ "$ELAPSED" -lt "$((DURATION_MINUTES * 60))" ]; do
    sleep "$COMMIT_INTERVAL"
    ELAPSED=$((ELAPSED + COMMIT_INTERVAL))
    sync_state
    echo "[daemon] heartbeat: ${ELAPSED}s / $((DURATION_MINUTES * 60))s"
  done

  if kill -0 "$BOT_PID" 2>/dev/null; then
    # The window reached its limit with the bot still running: this is the
    # NORMAL end. The bot is stopped deliberately by the cleanup trap.
    echo "[daemon] Window elapsed — cycling bot."
  else
    # The bot is gone before the window ended: it died on its own.
    wait "$BOT_PID" 2>/dev/null
    BOT_EXIT=$?
    BOT_DIED=1
    echo "[daemon] ERROR: the bot exited on its own (status ${BOT_EXIT}) before the window ended."
  fi
fi

if [ "$BOT_DIED" -eq 1 ]; then
  # A clean exit code still means a dead bot — the window promised a living
  # bot and did not deliver one. Never let it look green.
  if [ "$BOT_EXIT" -eq 0 ]; then
    BOT_EXIT=1
  fi
  echo "[daemon] the window is NOT healthy — failing the run."
  exit "$BOT_EXIT"
fi

exit 0
