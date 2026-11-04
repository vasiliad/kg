#!/usr/bin/env python3
"""
The supervisor of the cycles that still depend on GitHub's cron.

Measured on 2026-11-01: the scheduler skips slots. `bot_selftest.yml`'s daily
cron produced **zero** scheduled runs in the whole life of the repository;
the fleet tuner's 03:43Z slot produced no run; the diagnostics' 03:26Z slot
produced no run. A cycle that exists only in a cron is a cycle that does not
come back after a reboot — the exact thing the complex must not be.

The bot survived because its windows chain themselves (ADL-008). This module
gives the other cycles the same property. Each cycle has an ARTEFACT that
proves it ran:

    fleet          data/fleet_report.json        tuned_at
    diagnostics    diagnostics_report.json        timestamp
    self-test      data/bot_selftest_report.json ran_at
    daily dispatch nim_history.log               the dispatch record

The artefact — not the run history — is the signal: a run that crashed
produced nothing, so the cycle is still owed. The supervisor runs from the
two processes that demonstrably run (the bot window every ~50 minutes and
the heartbeat every few hours), dispatches a cycle whose artefact is stale,
and never dispatches a duplicate: a cycle with a run in progress or queued is
left alone, and a cycle that was attempted recently is left to its cooldown.

Three witnesses, and a silent one is not a negative answer:

    the artefact       when the cycle last proved it ran
    the repository     how long ago GitHub started the last run
    the ledger         when THIS supervisor last attempted the cycle

Measured in the cloud on 2026-11-02: the supervisor read "GitHub did not
answer" as "idle", skipped its own cooldown and dispatched on every tick.
Now an unanswered question only means the other witnesses are consulted —
and the attempt is written into the ledger, so the next tick remembers.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import atomic_io

# Each cycle: the workflow that runs it, the artefact that proves it ran,
# how old that artefact may get, and how long the supervisor must wait after
# an attempt before trying again.
CYCLES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "fleet",
        "workflow": "fleet_tuner.yml",
        "artefact": "data/fleet_report.json",
        "field": "tuned_at",
        "max_age_h": 7.0,
        "retry_h": 2.0,
    },
    {
        "name": "diagnostics",
        "workflow": "diagnostics.yml",
        "artefact": "diagnostics_report.json",
        "field": "timestamp",
        "max_age_h": 7.0,
        "retry_h": 2.0,
    },
    {
        "name": "selftest",
        "workflow": "bot_selftest.yml",
        "artefact": "data/bot_selftest_report.json",
        "field": "ran_at",
        "max_age_h": 26.0,
        "retry_h": 6.0,
    },
    {
        "name": "daily_dispatch",
        "workflow": "daily_dispatch_novt.yml",
        "artefact": "nim_history.log",
        # The marker must name the DIGEST, not "some report": the daily
        # cycle sends two letters, and a health letter on its own does not
        # fulfil the daily promise. Measured 2026-11-02: a generic marker
        # let one letter stand in for the whole report.
        "marker": "ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=Ежедневный дайджест ИИ",
        "max_age_h": 26.0,
        # A crashed dispatch must be retried long before the next report is
        # owed: the report is a daily promise, not a weekly one.
        "retry_h": 3.0,
        # The report is the 12:00 NOVT one (05:00 UTC). A daily cycle is not
        # merely stale — it is DUE: once a day, and not before its hour.
        "daily": True,
        "not_before_utc": 5,
    },
    {
        # The cloud verification — the cycle that notices when a SUPPLIER
        # changes something the complex depends on. It had no artefact and
        # no supervisor entry: the very check meant to catch drift was the
        # one cycle the scheduler could silently drop for weeks. The report
        # it now writes down is its proof of life.
        "name": "cloud_tests",
        "workflow": "cloud_tests.yml",
        "artefact": "data/cloud_tests_report.json",
        "field": "checked_at",
        "max_age_h": 7.0,
        "retry_h": 3.0,
    },
    {
        # The forge's own attempt to improve itself. The artefact's stamp is
        # `checked_at` — the cycle runs daily, but the ATTEMPT is rationed to
        # one per day inside the autopilot, so a re-dispatch after a forgotten
        # scheduler slot costs a check, not another model call.
        "name": "forge_autopilot",
        "workflow": "forge_autopilot.yml",
        "artefact": "data/forge_autopilot.json",
        "field": "checked_at",
        "max_age_h": 26.0,
        "retry_h": 6.0,
        "daily": True,
        "not_before_utc": 4,
    },
    {
        # The survival ladder — the complex's answer to «NVIDIA умерла».
        # Measured 2026-11-03: the module existed, was tested, and had NO
        # caller — the plan for surviving the cloud's death was itself
        # surviving only in the source code. Now the trigger measures the
        # cloud tier first (one catalog request) and walks the ladder only on
        # a measured death; the artefact is the proof that the check ran,
        # whatever the scheduler did with its cron slot.
        "name": "survival_protocol",
        "workflow": "survival_protocol.yml",
        "artefact": "data/survival_report.json",
        "field": "checked_at",
        "max_age_h": 26.0,
        "retry_h": 6.0,
        "daily": True,
        "not_before_utc": 3,
    },
    {
        # The third copy of the record — the one the complex owns. The account
        # was registered by the system itself (own mailbox → MEGA's client
        # protocol → verification link → login proved against the master key),
        # so this copy depends on neither the operator's inbox nor on GitHub
        # staying up. It is owed once a day, right after the digest.
        "name": "mega_archive",
        "workflow": "mega_archive.yml",
        "artefact": "data/mega_archive_report.json",
        "field": "checked_at",
        "max_age_h": 26.0,
        "retry_h": 6.0,
        "daily": True,
        "not_before_utc": 5,
    },
    {
        # The copy of the state that must survive THIS computer. The remote
        # is built by the complex itself — the credentials from the
        # environment or from the vault, no config file, no password in an
        # argument, and no mount (a mount dies with the runner). Measured
        # 2026-11-05: rclone refuses a plaintext password in the
        # environment, so the complex obscures it itself — rclone's own
        # AES-CTR format — and proves the copy by reading every file back.
        "name": "proton_drive",
        "workflow": "proton_drive_sync.yml",
        "artefact": "data/proton_drive_report.json",
        "field": "checked_at",
        "max_age_h": 26.0,
        "retry_h": 6.0,
        "daily": True,
        # From 06:00 UTC — AFTER the cycle's own 05:40 slot, so the daily
        # copy happens once and the supervisor only takes over when the
        # scheduler dropped the slot.
        "not_before_utc": 6,
    },
)

REPORT_PATH = PROJECT_ROOT / "data" / ".cycle_supervisor.json"

_LOG_LINE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def parse_timestamp(value: Any) -> Optional[datetime]:
    """
    A tolerant timestamp parser. The artefacts write three different
    shapes: ISO with a timezone, ISO without one, and «YYYY-MM-DD HH:MM:SS
    UTC». None of them may crash the supervisor.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("UTC"):
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S UTC")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _hours_since(ts: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if ts is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def read_dispatch_ledger(path: Path) -> Dict[str, Any]:
    """
    The supervisor's own memory of what it dispatched, and WHEN.

    GitHub is not always able to answer. Measured in the cloud: a tick that
    could not read the run history treated "unknown" as "idle" and dispatched
    the cycle again — and again, on every tick, because nothing in the
    complex remembered the attempt. This ledger is that memory: it lives in
    the supervisor's own report, which the heartbeat commits to the
    repository, so it survives the runner's filesystem being wiped.
    """
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    ledger = data.get("dispatch_log")
    return ledger if isinstance(ledger, dict) else {}


def dispatch_ledger_age_h(
    ledger: Dict[str, Any],
    cycle_name: str,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """How long ago THIS supervisor last attempted this cycle."""
    if not ledger:
        return None
    return _hours_since(parse_timestamp(ledger.get(cycle_name)), now)


def artefact_timestamp(
    path: Path,
    field: Optional[str] = None,
    marker: Optional[str] = None,
) -> Optional[datetime]:
    """
    When the artefact was produced. None means "unknown" — a missing file,
    an unreadable file or a timestamp that cannot be parsed.
    """
    if not path or not path.exists():
        return None
    if marker:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None
        for line in reversed(lines):
            if marker not in line:
                continue
            match = _LOG_LINE.match(line.strip())
            if not match:
                continue
            return parse_timestamp(match.group(1))
        return None
    if not field:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return parse_timestamp(data.get(field))


def artefact_age_h(
    path: Path,
    field: Optional[str] = None,
    marker: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """
    How old the artefact is, in hours. None means "unknown" — a missing
    file, an unreadable file or a timestamp that cannot be parsed. Unknown
    is treated as stale: a cycle that never ran is owed a run.
    """
    return _hours_since(artefact_timestamp(path, field=field, marker=marker), now)


def _gh(args: List[str], timeout: int = 60) -> Optional[str]:
    """Runs a gh command and returns its stdout, or None when it fails."""
    try:
        proc = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def workflow_busy(repo: str, workflow: str) -> Optional[bool]:
    """
    Whether a run of this workflow is in progress or queued. None when
    GitHub did not answer — and "unknown" must never be read as "idle".
    """
    if not repo:
        return None
    answered = False
    for status in ("in_progress", "queued"):
        out = _gh([
            "api",
            f"repos/{repo}/actions/workflows/{workflow}/runs?status={status}",
            "--jq", ".total_count",
        ])
        if out is None:
            continue
        answered = True
        try:
            if int(out or 0) > 0:
                return True
        except ValueError:
            continue
    return False if answered else None


def last_run_age_h(repo: str, workflow: str, now: Optional[datetime] = None) -> Optional[float]:
    """How long ago the newest run of this workflow was created."""
    if not repo:
        return None
    out = _gh([
        "api",
        f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=1",
        "--jq", ".workflow_runs[0].created_at",
    ])
    if not out:
        return None
    return _hours_since(parse_timestamp(out), now)


def dispatch_cycle(repo: str, workflow: str, ref: str = "") -> bool:
    """
    Starts a run of the workflow. True only when the dispatch worked — and a
    failed dispatch says WHY: a refusal that says nothing cannot be
    diagnosed (measured in the cloud on 2026-11-01, where the first dispatch
    attempt failed silently).
    """
    if not repo:
        return False
    # The "gh" prefix is part of the command: without it Python tries to
    # execute a program named "workflow" (measured in the cloud on
    # 2026-11-01, where the first dispatch died with FileNotFoundError).
    args = ["gh", "workflow", "run", workflow, "--repo", repo]
    if ref:
        args += ["--ref", ref]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        print(f"[supervisor] dispatch of {workflow} failed: {type(exc).__name__}: {exc}")
        return False
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:300]
        print(f"[supervisor] dispatch of {workflow} was refused: {reason}")
        return False
    return True


def supervise(
    repo: str = "",
    ref: str = "",
    now: Optional[datetime] = None,
    cycles: Tuple[Dict[str, Any], ...] = CYCLES,
    dry_run: bool = False,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Checks every cycle and dispatches the stale ones. The report says what
    was measured, so a quiet supervisor is distinguishable from a broken one.
    """
    repo = repo or os.getenv("GITHUB_REPOSITORY", "") or os.getenv("GITHUB_REPO", "")
    now = now or datetime.now(timezone.utc)
    root = Path(root) if root else PROJECT_ROOT
    # The ledger is read BEFORE the first dispatch: it is what turns
    # "GitHub did not answer" from "idle" into "we already tried".
    ledger = read_dispatch_ledger(root / "data" / ".cycle_supervisor.json")
    report: Dict[str, Any] = {
        "checked_at": now.isoformat(),
        "repo": repo,
        "dry_run": dry_run,
        "cycles": [],
        "dispatched": [],
        "dispatch_log": dict(ledger),
    }

    for cycle in cycles:
        artefact = root / cycle["artefact"]
        produced = artefact_timestamp(
            artefact,
            field=cycle.get("field"),
            marker=cycle.get("marker"),
        )
        age = _hours_since(produced, now)
        stale = age is None or age > float(cycle["max_age_h"])
        entry: Dict[str, Any] = {
            "name": cycle["name"],
            "workflow": cycle["workflow"],
            "artefact": cycle["artefact"],
            "age_h": None if age is None else round(age, 2),
            "max_age_h": cycle["max_age_h"],
            "stale": stale,
            "action": "",
        }

        # A daily cycle is DUE, not merely stale: once per calendar day, and
        # never before its own hour (the report goes out at 12:00 NOVT =
        # 05:00 UTC, not at midnight because the file is old).
        if cycle.get("daily"):
            not_before = int(cycle.get("not_before_utc", 0))
            now_utc = now.astimezone(timezone.utc)
            last_day = produced.astimezone(timezone.utc).date() if produced else None
            due_today = (
                last_day is None or last_day < now_utc.date()
            ) and now_utc.hour >= not_before
            entry["due_today"] = due_today
            stale = stale or due_today
            # The report must tell the truth: a cycle that is due TODAY is
            # stale even when its artefact is younger than the limit
            # (measured 2026-11-03: the entry said stale=false while the
            # supervisor was re-dispatching it).
            entry["stale"] = stale

        if not stale:
            entry["action"] = "fresh"
            report["cycles"].append(entry)
            continue
        if not repo:
            entry["action"] = "no_repository"
            report["cycles"].append(entry)
            continue

        busy = workflow_busy(repo, cycle["workflow"])
        entry["busy"] = busy
        # "Unknown" is NOT "idle". When GitHub cannot answer, the run
        # history below is the second witness — and the supervisor's own
        # ledger is the third. Only a confirmed "busy" stops the cycle.
        if busy is True:
            entry["action"] = "busy"
            report["cycles"].append(entry)
            continue

        last = last_run_age_h(repo, cycle["workflow"], now)
        local = dispatch_ledger_age_h(ledger, cycle["name"], now)
        entry["last_run_age_h"] = None if last is None else round(last, 2)
        entry["local_dispatch_age_h"] = None if local is None else round(local, 2)
        # The cooldown uses the YOUNGEST evidence: if either the repository
        # or the supervisor's own memory says we tried recently, the cycle
        # waits. Before this, a missing answer skipped the cooldown
        # entirely and the supervisor dispatched on every tick.
        known = [a for a in (last, local) if a is not None]
        effective = min(known) if known else None
        entry["effective_age_h"] = None if effective is None else round(effective, 2)
        if effective is not None and effective < float(cycle["retry_h"]):
            entry["action"] = "cooldown"
            report["cycles"].append(entry)
            continue

        if dry_run:
            entry["action"] = "would_dispatch"
            report["cycles"].append(entry)
            continue

        ok = dispatch_cycle(repo, cycle["workflow"], ref)
        entry["action"] = "dispatched" if ok else "dispatch_failed"
        # The ATTEMPT is remembered, not only the success: a dispatch that
        # was refused must not be retried on every tick either.
        report["dispatch_log"][cycle["name"]] = now.isoformat()
        if ok:
            report["dispatched"].append(cycle["workflow"])
        report["cycles"].append(entry)

    return report


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """Writes the report next to the other state files."""
    target = path or REPORT_PATH
    # Atomic: THIS file is the supervisor's own memory of what it dispatched.
    # A half-written report reads as "no cycles ran" and the next heartbeat
    # would re-dispatch everything.
    if not atomic_io.write_json_atomic(target, report):
        return None
    return target


def main() -> int:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    repo = ""
    ref = ""
    for arg in args:
        if arg.startswith("--repo="):
            repo = arg.split("=", 1)[1]
        elif arg.startswith("--ref="):
            ref = arg.split("=", 1)[1]

    try:
        report = supervise(repo=repo, ref=ref, dry_run=dry_run)
    except Exception as exc:
        # A broken supervisor must never break the process it watches.
        print(f"[supervisor] WARNING: the supervisor itself failed: {type(exc).__name__}: {exc}")
        return 0

    for entry in report["cycles"]:
        age = "unknown" if entry.get("age_h") is None else f"{entry['age_h']}h"
        print(f"[supervisor] {entry['name']:<16} age={age:<8} "
              f"limit={entry['max_age_h']}h  -> {entry['action']}")
    if report["dispatched"]:
        print(f"[supervisor] dispatched: {', '.join(report['dispatched'])}")
    else:
        print("[supervisor] nothing to dispatch.")

    write_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
