#!/usr/bin/env python3
"""
Operational context builder for the Telegram Command Center.

The bot used to answer from the conversation log alone, so any question about
what the complex actually did ("were the reports sent today?", "which models
are active?", "is the key alive?") was answered from a log that does not
contain that information — which led the model to hedge and, worse, to invent
an operator role that does not exist.

This module assembles the verifiable facts from the artefacts the complex
really produces, so the Command Center can answer from evidence:

- data/.bot_settings.json  — the current dispatch recipient
- nim_history.log          — the operational log (cycles, key checks, dispatches)
- active_models.json       — the active fleet and the last catalog diff
- diagnostics_report.json  — the last cloud dependency health report
- data/telegram_history.txt — the conversation log tail

Everything here is read-only and never raises: a missing file simply yields
"no data", never an exception that would break a reply.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files worth reporting even when they are empty, so the model can tell
# "nothing happened" apart from "the log was never created".
_KEY_FILES = {
    "settings": PROJECT_ROOT / "data" / ".bot_settings.json",
    "history": PROJECT_ROOT / "nim_history.log",
    "state": PROJECT_ROOT / "active_models.json",
    "diagnostics": PROJECT_ROOT / "diagnostics_report.json",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _tail(path: Path, lines: int = 15) -> List[str]:
    try:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        return content.splitlines()[-lines:]
    except Exception:
        return []


def _recipient() -> str:
    try:
        from harness import runtime_settings
        return runtime_settings.get_recipient_email("не задан")
    except Exception:
        return os.getenv("DISPATCH_RECIPIENT_EMAIL", "не задан")


def _dispatch_events(history_lines: List[str]) -> List[str]:
    """Lines of the operational log that concern report dispatching."""
    return [
        ln for ln in history_lines
        if "ОТЧЁТ" in ln.upper() or "DISPATCH" in ln.upper()
    ]


def build_operational_context(
    query: str = "",
    event_memory: Any = None,
    max_history_lines: int = 15
) -> str:
    """
    Returns a compact, factual snapshot of the complex for the answering
    model. It is deliberately plain text: the model must quote these values
    instead of guessing.

    When a query and a semantic index over the operational log are supplied,
    the events that match the question are included as well — the same
    "durable text log + embedding index" protocol that serves the
    conversation memory, applied to events and actions.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts: List[str] = [
        "--- COMMAND CENTER: VERIFIED SYSTEM FACTS (read-only snapshot) ---",
        f"Snapshot taken at: {now}",
        f"Report recipient configured: {_recipient()}",
    ]

    # Operational log
    history = _tail(_KEY_FILES["history"], max_history_lines)
    dispatches = _dispatch_events(_tail(_KEY_FILES["history"], 1000))
    if history:
        parts.append("\nOperational log (nim_history.log, most recent last):")
        parts.extend(f"  {ln}" for ln in history)
    else:
        parts.append("\nOperational log (nim_history.log): EMPTY - no recorded activity.")

    if dispatches:
        parts.append("\nReport dispatch records (authoritative - newest last):")
        parts.extend(f"  {ln}" for ln in dispatches[-8:])
    else:
        parts.append(
            "\nReport dispatch records: NONE. "
            "If the user asks whether reports were sent, answer exactly that: "
            "there is no dispatch record, so the sending did not happen or was "
            "not logged - and offer to trigger the check now."
        )

    # Active fleet
    state = _read_json(_KEY_FILES["state"])
    if state:
        fleet = state.get("active_fleet") or []
        if isinstance(fleet, list):
            fleet_lines = []
            for m in fleet[:6]:
                if not isinstance(m, dict):
                    continue
                score = m.get("quality_score")
                score_text = "unranked" if score is None else str(score)
                fleet_lines.append(
                    f"  {m.get('id')} | tier={m.get('tier')} | score={score_text} "
                    f"| verified={m.get('score_verified')} | status={m.get('status')}"
                )
            parts.append("\nActive fleet (active_models.json):")
            parts.extend(fleet_lines or ["  (empty)"])
        diff = state.get("catalog_diff") or {}
        if isinstance(diff, dict) and (diff.get("new_count") or diff.get("removed_count")):
            parts.append(
                f"  Catalog diff since last scan: +{diff.get('new_count', 0)} / -{diff.get('removed_count', 0)}"
            )
        if state.get("last_emergency_run"):
            parts.append(f"  Last emergency run: {state.get('last_emergency_run')}")
        if state.get("key_health"):
            kh = state.get("key_health")
            if isinstance(kh, dict):
                parts.append(f"  NVIDIA key health: {kh.get('message')}")
    else:
        parts.append("\nActive fleet (active_models.json): NOT PRESENT.")

    # Semantically matched events: the operational log is indexed exactly like
    # the conversation, so "were the reports sent today?" is answered by
    # finding the matching DISPATCH record instead of guessing.
    if query and event_memory is not None:
        try:
            matches = event_memory.search_memories(query, top_k=6)
        except Exception:
            matches = []
        if matches:
            parts.append("\nEvents from the operational log that match the question:")
            for m in matches:
                parts.append(f"  [{m.get('timestamp')}] {m.get('text')[:300]}")
        else:
            parts.append(
                "\nEvents from the operational log that match the question: NONE FOUND. "
                "The semantic search over the event log returned no matching record."
            )

    # Last cloud diagnostics
    diag = _read_json(_KEY_FILES["diagnostics"])
    if diag:
        nvidia = diag.get("nvidia") or {}
        telegram = diag.get("telegram") or {}
        keenable = diag.get("keenable") or {}
        parts.append("\nLast cloud diagnostics (diagnostics_report.json):")
        parts.append(
            f"  taken at: {diag.get('timestamp')} | NVIDIA ok={nvidia.get('ok')} "
            f"| live models={len(nvidia.get('models') or [])}"
        )
        if telegram.get("username"):
            parts.append(f"  Telegram bot: {telegram.get('username')} (reachable={telegram.get('ok')})")
        if keenable:
            parts.append(f"  Keenable search: ok={keenable.get('ok')} reason={keenable.get('reason', 'n/a')}")
    else:
        parts.append("\nLast cloud diagnostics: NOT PRESENT.")

    parts.append(
        "\nRULES FOR ANSWERING FROM THESE FACTS:\n"
        "- You ARE the Command Center. There is no separate human operator, "
        "no 'duty operator' and no escalation chain. Never invent one and never "
        "tell the user to check something themselves.\n"
        "- If a fact is in this snapshot, state it plainly with its timestamp.\n"
        "- If a fact is missing, say so explicitly and offer the concrete action "
        "you can take right now (for example: run the mail check, re-scan the "
        "catalog, re-run diagnostics).\n"
        "- Never claim an action was performed when this snapshot does not show it."
    )
    return "\n".join(parts)
