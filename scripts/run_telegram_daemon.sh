#!/usr/bin/env bash
# ==============================================================================
# Telegram Bot Daemon Runner (cloud-native, GitHub Actions).
#
# Runs the bot as a background process for a bounded window, periodically
# persisting sessions/memory/state back into the repository so that the
# authorised chat, the conversation log and the memory vault survive the
# runner filesystem being wiped between jobs.
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
  "data/.bot_settings.json"
  "data/telegram_history.txt"
  "active_models.json"
  "nim_history.log"
)

sync_state() {
  local args=()
  for f in "${STATE_FILES[@]}"; do
    [ -e "$f" ] && args+=("$f")
  done
  [ ${#args[@]} -eq 0 ] && return 0

  git add -- "${args[@]}" 2>/dev/null || true
  if ! git diff --staged --quiet 2>/dev/null; then
    if git commit -m "chore(bot): persist session and memory state [skip ci]" >/dev/null 2>&1; then
      # Shared helper: rebases and retries, because the heartbeat and the
      # diagnostics job push to the same branch concurrently.
      bash "${GITHUB_WORKSPACE:-$(pwd)}/scripts/sync_state.sh" 2>&1 | sed 's/^/[daemon] /'
    fi
  fi
}

BOT_PID=""
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
python harness/telegram_bot.py &
BOT_PID=$!

ELAPSED=0
while kill -0 "$BOT_PID" 2>/dev/null && [ "$ELAPSED" -lt "$((DURATION_MINUTES * 60))" ]; do
  sleep "$COMMIT_INTERVAL"
  ELAPSED=$((ELAPSED + COMMIT_INTERVAL))
  sync_state
  echo "[daemon] heartbeat: ${ELAPSED}s / $((DURATION_MINUTES * 60))s"
done

echo "[daemon] Window elapsed — cycling bot."
