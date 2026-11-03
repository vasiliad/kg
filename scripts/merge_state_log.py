#!/usr/bin/env python3
"""
Merges two versions of an append-only state log into ONE that keeps both.

Measured on 2026-11-02, in the cloud, from the repository's own history:

    72c2543  11-02 05:24  chore(dispatch): record report dispatch outcome
        + [2026-11-02 05:20:18 UTC] ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=Проверка связи ...
        + [2026-11-02 05:24:14 UTC] ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=Ежедневный дайджест ИИ ...

    770dd9c  11-02 05:30  feat(fleet): measured duty-team re-selection
        - [2026-11-02 05:20:18 UTC] ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=Проверка связи ...
        - [2026-11-02 05:24:14 UTC] ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=Ежедневный дайджест ИИ ...

The fleet tuner's workspace held an older copy of `nim_history.log`. When the
two writers met in the rebase, the conflict was resolved by taking one side
WHOLE — and the report's proof of delivery was erased from the repository.

The consequence was not cosmetic. At 10:31 the cycle supervisor read the log,
saw the newest report as 22.7 hours old, decided the daily cycle was owed,
and dispatched it again. The operator received the same letter four times in
one day (05:20, 10:31, 11:41, 12:26, 17:32 UTC — the first four of those
automatically). One erased log line became a flood of mail.

A log is not a snapshot. Both writers' lines are true. This helper is the
resolution the shared sync script was missing: the UNION, in chronological
order, with the other writer's lines restored.

Two formats are understood (see harness/memory_engine.py):
    events      — one "[timestamp] message" per line        (nim_history.log)
    conversation— blocks of "[timestamp] ROLE: text", separated by a blank
                  line                                       (telegram_history)

Usage (inside a conflicted rebase):
    python scripts/merge_state_log.py --file nim_history.log --mode events
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _blocks(text: str) -> List[str]:
    """The conversation's blocks: non-empty chunks separated by blank lines."""
    out: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            out.append("\n".join(current))
            current = []
    if current:
        out.append("\n".join(current))
    return out


def _sort_key(block: str) -> Tuple[str, str]:
    """Chronological order by the leading "[timestamp]" of a block/line."""
    first = block.splitlines()[0] if block.splitlines() else ""
    return (first[:32], block)


def merge_events(text_a: str, text_b: str) -> str:
    """
    The union of two event logs: every distinct line, oldest first.

    The log's lines are uniform ("[YYYY-MM-DD HH:MM:SS UTC] message"), so the
    byte order of the timestamp prefix IS the chronological order. Identical
    lines are collapsed: the same event recorded twice is one event.
    """
    lines = set()
    for text in (text_a or "", text_b or ""):
        for line in text.splitlines():
            if line.strip():
                lines.add(line)
    if not lines:
        return ""
    return "\n".join(sorted(lines)) + "\n"


def merge_conversation(text_a: str, text_b: str) -> str:
    """
    The union of two conversation logs: every distinct block, oldest first,
    still separated by a blank line (the format the memory engine reads).
    """
    blocks = set()
    for text in (text_a or "", text_b or ""):
        for block in _blocks(text):
            if block.strip():
                blocks.add(block)
    if not blocks:
        return ""
    return "\n\n".join(sorted(blocks, key=_sort_key)) + "\n"


def merge_log(text_a: str, text_b: str, mode: str = "events") -> str:
    """The merge, by format."""
    if mode == "conversation":
        return merge_conversation(text_a, text_b)
    return merge_events(text_a, text_b)


def _git_show(stage: str, path: str) -> Optional[str]:
    """
    One side of a conflicted file: stage 2 is the upstream ("ours" during a
    rebase), stage 3 is the commit being replayed ("theirs"). None when the
    stage does not exist — the file was added on one side only.
    """
    try:
        proc = subprocess.run(
            ["git", "show", f"{stage}:{path}"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def merge_conflicted_file(path: str, mode: str = "events") -> bool:
    """
    Resolves a conflicted state log in the working tree: both sides are read
    from the index, merged, written back and staged. True when the file is
    resolved (and staged) — false when it could not be merged, in which case
    the caller must not pretend the conflict is gone.
    """
    upstream = _git_show(":2", path)
    local = _git_show(":3", path)
    if upstream is None and local is None:
        return False
    merged = merge_log(upstream or "", local or "", mode=mode)
    target = PROJECT_ROOT / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merged, encoding="utf-8")
    except Exception as exc:
        print(f"[merge] {path}: could not be written: {exc}")
        return False
    try:
        proc = subprocess.run(
            ["git", "add", "--", path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        print(f"[merge] {path}: could not be staged: {exc}")
        return False
    if proc.returncode != 0:
        print(f"[merge] {path}: could not be staged: {(proc.stderr or '').strip()}")
        return False
    print(
        f"[merge] {path}: both writers kept "
        f"(upstream {len((upstream or '').splitlines())} lines, "
        f"local {len((local or '').splitlines())} lines, "
        f"merged {len(merged.splitlines())} lines)."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--mode", default="events", choices=["events", "conversation"])
    args = parser.parse_args()
    if not merge_conflicted_file(args.file, mode=args.mode):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
