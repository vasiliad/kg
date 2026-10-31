#!/usr/bin/env bash
# ==============================================================================
# Bot Reviver (cloud-native, GitHub Actions).
#
# The Telegram bot normally lives in a self-chaining sequence of windows:
# every window dispatches its own successor. That chain is the primary
# mechanism and does not depend on GitHub's cron scheduler being punctual.
#
# This script is the safety net for the case where the chain itself breaks —
# a dispatch that failed three times, a cancelled run, a runner that died.
# It starts a new window ONLY when no window is running and none is waiting,
# so it can never create a second chain next to a live one.
#
# Usage: bash scripts/revive_bot.sh [repo] [workflow] [duration_minutes]
#
# Env:
#   GH_TOKEN     token with `actions: write` (github.token works)
#   GITHUB_REF_NAME  branch to dispatch on (default: main)
# ==============================================================================
set -uo pipefail

REPO="${1:-${GITHUB_REPOSITORY:-}}"
WORKFLOW="${2:-telegram_bot.yml}"
DURATION="${3:-50}"
REF="${GITHUB_REF_NAME:-main}"

if [ -z "$REPO" ]; then
  echo "[revive] no repository given — nothing to revive."
  exit 0
fi

API="repos/${REPO}/actions/workflows/${WORKFLOW}/runs"

active=$(gh api "${API}?status=in_progress" --jq '.total_count' 2>/dev/null || echo 0)
queued=$(gh api "${API}?status=queued" --jq '.total_count' 2>/dev/null || echo 0)
echo "[revive] active=${active} queued=${queued}"

if [ "${active}" != "0" ] || [ "${queued}" != "0" ]; then
  echo "[revive] the bot is alive — nothing to do."
  exit 0
fi

echo "[revive] no live bot window — dispatching a new one."
for attempt in 1 2 3; do
  output=$(gh workflow run "$WORKFLOW" --repo "$REPO" --ref "$REF" \
             -f "duration_minutes=${DURATION}" 2>&1)
  status=$?
  if [ "$status" -eq 0 ]; then
    [ -n "$output" ] && echo "$output"
    echo "[revive] new window dispatched (attempt ${attempt})."
    exit 0
  fi
  echo "[revive] dispatch failed (attempt ${attempt}): ${output}"
  sleep 10
done

# Never fail the surrounding job: the next tick will try again.
echo "[revive] WARNING: could not dispatch a window — the next tick will retry."
exit 0
