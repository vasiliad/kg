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
# THE SCRIPT FAILS LOUDLY (audited 2026-11-04):
#   * it used to exit 0 after three failed dispatches, so the watchdog run
#     went GREEN while the bot was DEAD — the exact "silent death" this
#     complex exists to prevent;
#   * it used to treat a failed state query as "no live window" and dispatch
#     anyway. The whole point of the query is to KNOW that no window is live;
#     a blind dispatch can start a SECOND chain next to a live one. A
#     reviver that cannot see must not act.
# The next tick still tries again: failing loudly does not stop the schedule,
# it makes the failure visible to whoever reads the run.
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
  echo "[revive] ERROR: no repository given — the bot cannot be revived."
  exit 1
fi

API="repos/${REPO}/actions/workflows/${WORKFLOW}/runs"

# THE FACTS FIRST. A reviver that cannot see the chain must not touch it.
active=$(gh api "${API}?status=in_progress" --jq '.total_count' 2>/dev/null) || active=""
queued=$(gh api "${API}?status=queued" --jq '.total_count' 2>/dev/null) || queued=""

if [ -z "$active" ] || [ -z "$queued" ]; then
  echo "[revive] ERROR: GitHub did not answer the chain query — refusing to dispatch blindly."
  echo "[revive] A blind dispatch can start a second chain next to a live one; the next tick will retry."
  exit 1
fi

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

# A failed revival is NOT a quiet warning. Measured on 2026-11-04: this
# script printed a WARNING and exited 0, the watchdog run went green, and
# the bot stayed dead until a human noticed the silence. The next tick
# still tries again — but this run says that the bot is NOT alive.
echo "[revive] ERROR: could not dispatch a window after 3 attempts — the bot is DEAD and this run says so."
exit 1
