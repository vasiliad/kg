#!/usr/bin/env python3
"""
The alarm that says *the bot is quiet* — and why.

Silence has two very different causes: the daemon is not running, or the
daemon is running and failing. From the operator's chat they look exactly
the same — no answer. The daemon writes a health trace (harness/
telegram_bot.py) and the reviver can start a window (scripts/revive_bot.sh);
neither of them can tell the operator that silence is happening right now.

This is the third leg. It reads the trace, asks GitHub whether a bot window
is alive at this moment, and — when the silence is real — mails the measured
evidence to the address the operator configured.

Three rules keep it honest:
  * it never guesses: every line in the mail is a measured number;
  * it never floods: one mail per outage, the marker file remembers;
  * it never breaks the heartbeat: every failure is swallowed and reported.

The recipient is never hardcoded — it comes from the runtime settings
(data/.bot_settings.json, changeable from Telegram) and falls back to the
DISPATCH_RECIPIENT_EMAIL secret.

Why two signals and not one: the daemon pushes its trace every three
minutes, so a committed trace can lag behind a perfectly healthy bot by a
few minutes. Judging silence from the trace alone would fire false alarms.
GitHub can be asked directly whether a window is running — and a running
window is a live bot whatever the trace says.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import atomic_io

SESSIONS_FILE = PROJECT_ROOT / "data" / ".bot_sessions.json"
MARKER_FILE = PROJECT_ROOT / "data" / ".bot_silence_alarm.json"

# A poll older than this means "the bot is not talking to Telegram".
SILENCE_THRESHOLD_MIN = 15.0
# A live window that has not pushed a trace for this long is stuck, not slow.
WEDGE_THRESHOLD_MIN = 45.0
# A window that was just (re)started has not pushed its first trace yet.
GRACE_MIN = 12.0
# The gap between two chained windows — the successor is already queued.
HANDOFF_MIN = 3.0
# One mail per outage, not one per cycle.
ALERT_COOLDOWN_MIN = 360.0

REASONS = {
    "no_window": "ни одно окно бота не запущено и не стоит в очереди",
    "window_wedged": "окно бота запущено, но оно не опрашивает Telegram",
    "stale_trace": "нет свежих опросов Telegram, и GitHub не подтвердил живое окно",
    "no_health_block": "бот ни разу не записал след своей работы",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: Optional[datetime] = None) -> str:
    return (moment or _now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(stamp: Any) -> Optional[datetime]:
    """Parses the UTC stamps the bot writes. Returns None instead of raising."""
    if not stamp:
        return None
    text = str(stamp).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_min(stamp: Any, now: Optional[datetime] = None) -> Optional[float]:
    """How many minutes ago the stamp was, or None when it cannot be read."""
    moment = _parse_iso(stamp)
    if moment is None:
        return None
    now = now or _now()
    return round((now - moment).total_seconds() / 60.0, 1)


# --------------------------------------------------------------------------- #
# Signal 1: the bot's own trace
# --------------------------------------------------------------------------- #
def read_health(path: Optional[Any] = None) -> Dict[str, Any]:
    """
    Reads the bot's health block. Never raises: a broken file is a verdict,
    not a crash — an alarm that dies with the thing it watches is useless.
    """
    target = Path(path) if path else SESSIONS_FILE
    result: Dict[str, Any] = {
        "ok": False,
        "reason": "no_session_file",
        "health": {},
        "cursor": 0,
        "path": str(target),
    }
    if not target.exists():
        return result
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        result["reason"] = f"broken_session_file: {type(exc).__name__}"
        return result
    if not isinstance(data, dict):
        result["reason"] = "broken_session_file"
        return result

    health = data.get("health")
    result["health"] = health if isinstance(health, dict) else {}
    try:
        result["cursor"] = int(data.get("last_update_id") or 0)
    except Exception:
        result["cursor"] = 0
    if result["health"]:
        result["ok"] = True
        result["reason"] = "ok"
    else:
        result["reason"] = "no_health_block"
    return result


# --------------------------------------------------------------------------- #
# Signal 2: is a bot window alive right now?
# --------------------------------------------------------------------------- #
def _query_runs(repo: str, workflow: str, status: str) -> Tuple[bool, int, Optional[datetime]]:
    """Returns (query_ok, count, newest_start) for the windows in `status`."""
    url = f"repos/{repo}/actions/workflows/{workflow}/runs?status={status}&per_page=10"
    try:
        proc = subprocess.run(
            ["gh", "api", url], capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return False, 0, None
    if proc.returncode != 0:
        return False, 0, None
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return False, 0, None
    runs = payload.get("workflow_runs") or []
    try:
        count = int(payload.get("total_count") or 0)
    except Exception:
        count = len(runs)
    newest: Optional[datetime] = None
    for run in runs:
        if not isinstance(run, dict):
            continue
        moment = _parse_iso(run.get("run_started_at") or run.get("created_at"))
        if moment and (newest is None or moment > newest):
            newest = moment
    return True, count, newest


def live_window(repo: Optional[str] = None, workflow: str = "telegram_bot.yml") -> Dict[str, Any]:
    """
    Asks GitHub whether a bot window is running at this moment.

    A window that is running or waiting in the queue IS a live bot, whatever
    the committed trace says. When GitHub cannot be asked, the verdict is
    "unknown" and the trace alone has to decide.
    """
    repo = repo or os.getenv("GITHUB_REPOSITORY", "")
    state: Dict[str, Any] = {
        "state": "unknown",
        "running": 0,
        "queued": 0,
        "started_at": None,
        "age_min": None,
        "detail": "",
    }
    if not repo:
        state["detail"] = "no repository given"
        return state
    if not (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")):
        state["detail"] = "no GitHub token in the environment"
        return state

    queries: List[Tuple[bool, int, Optional[datetime]]] = []
    for status in ("in_progress", "queued"):
        queries.append(_query_runs(repo, workflow, status))

    ok, running, started_running = queries[0]
    ok_queued, queued, started_queued = queries[1]
    state["running"] = running
    state["queued"] = queued
    starts = [s for s in (started_running, started_queued) if s]
    if starts:
        newest = max(starts)
        state["started_at"] = _iso(newest)
        state["age_min"] = _age_min(_iso(newest))

    if running or queued:
        state["state"] = "alive"
        state["detail"] = f"running={running} queued={queued}"
        return state

    # A partial answer is never proof of death: if one query failed, a window
    # may still be running and we simply cannot see it. Only two agreeing
    # answers may call the bot dead.
    if ok and ok_queued:
        state["state"] = "dead"
        state["detail"] = "running=0 queued=0"
    else:
        state["state"] = "unknown"
        state["detail"] = "GitHub did not answer both queries"
    return state


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #
def judge(
    health_result: Dict[str, Any],
    window: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Turns the two signals into one honest verdict.

    ok     — the bot is alive and talking to Telegram.
    watch  — a window is alive but the trace lags; push latency, not an outage.
    alarm  — the silence is real; the operator must be told.
    """
    now = now or _now()
    window = window or {"state": "unknown"}
    health = health_result.get("health") or {}
    age = _age_min(health.get("last_poll_at"), now)
    window_state = window.get("state", "unknown")
    window_age = window.get("age_min")

    verdict: Dict[str, Any] = {
        "silent": False,
        "severity": "ok",
        "reason": "ok",
        "age_min": age,
        "health": health,
        "cursor": health_result.get("cursor", 0),
        "window": window,
        "checked_at": _iso(now),
    }

    def alarm(reason: str) -> Dict[str, Any]:
        verdict.update(silent=True, severity="alarm", reason=reason)
        return verdict

    # A window that was just restarted has not pushed its first trace yet.
    just_started = (
        window_state == "alive"
        and window_age is not None
        and window_age <= GRACE_MIN
    )

    if age is None:
        if just_started:
            verdict["reason"] = "window_restarted"
            return verdict
        if window_state == "dead":
            return alarm("no_window")
        return alarm("no_health_block")

    if window_state == "alive":
        if age <= SILENCE_THRESHOLD_MIN:
            verdict["reason"] = "ok"
            return verdict
        if age <= WEDGE_THRESHOLD_MIN:
            # The window is alive; only the committed trace lags behind it.
            verdict["severity"] = "watch"
            verdict["reason"] = "trace_lag"
            return verdict
        return alarm("window_wedged")

    if window_state == "dead":
        if age <= HANDOFF_MIN:
            # The chain is between two windows — the successor is about to run.
            verdict["reason"] = "handover"
            return verdict
        return alarm("no_window")

    # GitHub could not be asked: the trace alone has to decide.
    if age <= SILENCE_THRESHOLD_MIN:
        verdict["reason"] = "ok"
        return verdict
    return alarm("stale_trace")


# --------------------------------------------------------------------------- #
# The marker — one mail per outage
# --------------------------------------------------------------------------- #
def read_marker(path: Optional[Any] = None) -> Dict[str, Any]:
    target = Path(path) if path else MARKER_FILE
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_marker(payload: Dict[str, Any], path: Optional[Any] = None) -> None:
    target = Path(path) if path else MARKER_FILE
    # Atomic: the marker says "the alarm already fired" — a half-written
    # marker would make the alarm forget itself and flood the operator.
    if not atomic_io.write_json_atomic(target, payload):
        print("[!] Could not write the alarm marker")


def resolve_recipient() -> str:
    """
    The address the alarm writes to. Never hardcoded: the runtime settings
    win (the operator can change them from Telegram), the secret is the
    fallback, and an empty string means "nobody to tell".
    """
    try:
        from harness import runtime_settings
        recipient = runtime_settings.get_recipient_email()
    except Exception:
        recipient = ""
    return (recipient or os.getenv("DISPATCH_RECIPIENT_EMAIL", "")).strip()


# --------------------------------------------------------------------------- #
# The mail
# --------------------------------------------------------------------------- #
def _evidence_lines(verdict: Dict[str, Any]) -> List[str]:
    health = verdict.get("health") or {}
    window = verdict.get("window") or {}
    age = verdict.get("age_min")
    age_text = f"{age} мин назад" if age is not None else "нет записи об опросе"
    return [
        f"Время проверки (UTC) : {verdict.get('checked_at')}",
        f"Причина              : {REASONS.get(verdict.get('reason'), verdict.get('reason'))}",
        f"Окно бота            : {window.get('state')} ({window.get('detail') or '—'})",
        f"Начало окна (UTC)    : {window.get('started_at') or '—'}",
        f"Последний опрос      : {health.get('last_poll_at') or '—'} ({age_text})",
        f"Курсор Telegram      : {verdict.get('cursor', 0)}",
        f"Обновлений обработано: {health.get('updates_seen', 0)}",
        f"Отправлено ответов   : {health.get('messages_sent', 0)}",
        f"Ошибок               : {health.get('errors', 0)}",
        f"Последняя ошибка     : {health.get('last_error') or '—'}",
    ]


def render_mail(verdict: Dict[str, Any]) -> Tuple[str, str, str]:
    """Returns (subject, html, plain_text)."""
    reason = REASONS.get(verdict.get("reason"), verdict.get("reason") or "неизвестно")
    age = verdict.get("age_min")
    age_suffix = f" (молчит {age} мин)" if age is not None else ""
    subject = f"🚨 Telegram-бот не отвечает: {reason}{age_suffix}"

    lines = _evidence_lines(verdict)
    plain = (
        "Telegram-бот не отвечает.\n\n"
        + "\n".join(lines)
        + "\n\n"
        + "Что уже сделано: reviver (scripts/revive_bot.sh) уже попытался поднять "
        + "окно бота; это письмо — независимая проверка молчания.\n"
        + "Если бот только что был перезапущен, первые минуты он может молчать.\n"
    )
    html = (
        '<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'
        'max-width:640px">'
        '<h2 style="color:#b3261e;margin:0 0 12px">Telegram-бот не отвечает</h2>'
        f'<p style="margin:0 0 12px">{reason}{age_suffix}</p>'
        '<table cellpadding="4" cellspacing="0" style="border-collapse:collapse;'
        'font-size:13px">'
        + "".join(
            f'<tr><td style="color:#666;white-space:nowrap">{line.split(":", 1)[0]}</td>'
            f'<td style="font-family:ui-monospace,Menlo,Consolas,monospace">'
            f'{_escape_html(line.split(":", 1)[1].strip() if ":" in line else line)}</td></tr>'
            for line in lines
        )
        + "</table>"
        '<p style="color:#666;font-size:12px;margin-top:16px">Reviver уже попытался '
        'поднять окно бота; это письмо — независимая проверка молчания. '
        'Если бот только что был перезапущен, первые минуты он может молчать.</p>'
        "</div>"
    )
    return subject, html, plain


def _escape_html(text: str) -> str:
    return html.escape(str(text), quote=False)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_silence_alarm(
    now: Optional[datetime] = None,
    repo: Optional[str] = None,
    send: bool = True,
    sessions_path: Optional[Any] = None,
    marker_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Judges the silence and, when it is real, mails the evidence.

    Never raises: a broken alarm must not break the heartbeat that carries it.
    """
    now = now or _now()
    try:
        health = read_health(sessions_path)
        window = live_window(repo=repo)
        verdict = judge(health, window, now=now)
    except Exception as exc:
        return {
            "sent": False,
            "verdict": {"silent": False, "severity": "ok", "reason": "alarm_broken"},
            "error": f"{type(exc).__name__}: {exc}",
        }

    report: Dict[str, Any] = {"sent": False, "verdict": verdict}

    if not verdict["silent"]:
        marker = read_marker(marker_path)
        if marker.get("last_alert_at"):
            # The bot is back: clear the marker so the next outage is reported
            # at once, without waiting for the cooldown.
            write_marker(
                {
                    "state": "ok",
                    "last_alert_at": None,
                    "recovered_at": _iso(now),
                    "last_check_at": _iso(now),
                },
                marker_path,
            )
        report["reason"] = verdict["reason"]
        return report

    marker = read_marker(marker_path)
    last_alert = _parse_iso(marker.get("last_alert_at"))
    if last_alert is not None:
        elapsed = (now - last_alert).total_seconds() / 60.0
        if elapsed < ALERT_COOLDOWN_MIN:
            report["reason"] = "cooldown"
            report["minutes_since_last_alert"] = round(elapsed, 1)
            return report

    recipient = resolve_recipient()
    subject, html, plain = render_mail(verdict)
    outcome: Dict[str, Any] = {"status": "SKIPPED"}
    if not recipient:
        outcome = {"status": "SKIPPED_NO_RECIPIENT"}
    elif not send:
        outcome = {"status": "SKIPPED_NOT_SENDING"}
    else:
        try:
            from harness.email_sender import EmailSender
            outcome = EmailSender().send_email(recipient, subject, html, plain)
        except Exception as exc:
            outcome = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    delivered = outcome.get("status") == "DELIVERED"
    report["sent"] = delivered
    report["outcome"] = outcome
    report["recipient_configured"] = bool(recipient)

    # Only a delivered mail starts the cooldown — a failed attempt must be
    # retried on the next cycle, not silently swallowed.
    write_marker(
        {
            "state": "alarm" if delivered else "attempt_failed",
            "last_alert_at": _iso(now) if delivered else marker.get("last_alert_at"),
            "last_attempt_at": _iso(now),
            "last_reason": verdict.get("reason"),
            "last_age_min": verdict.get("age_min"),
            "last_check_at": _iso(now),
        },
        marker_path,
    )
    return report


def selftest(repo: Optional[str] = None) -> Dict[str, Any]:
    """
    Proves the outage path is armed — without mailing anyone.

    A synthetic trace that is 90 minutes old and a simulated dead window must
    produce exactly the alarm verdict a real outage would produce, and the
    recipient must be resolvable. The address itself is never printed: the
    alarm has to work without ever writing the address into a log.
    """
    import tempfile

    now = _now()
    stamp = (now - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    health = {
        "runs": 3, "started_at": stamp, "last_poll_at": stamp,
        "last_update_at": None, "last_send_at": None, "last_chat_id": None,
        "updates_seen": 0, "messages_sent": 0, "errors": 0, "last_error": None,
    }
    tmp = Path(tempfile.mkdtemp(prefix="silence-selftest-"))
    sessions = tmp / ".bot_sessions.json"
    sessions.write_text(
        json.dumps({"last_update_id": 0, "authorized_chats": [], "health": health},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The chain is simulated as dead: that is the only state in which the
    # alarm must fire, and it must not depend on the live GitHub state.
    window = {"state": "dead", "running": 0, "queued": 0,
              "started_at": None, "age_min": None, "detail": "simulated"}
    verdict = judge(read_health(sessions), window, now=now)
    subject, html, plain = render_mail(verdict)
    recipient = resolve_recipient()

    return {
        "verdict": verdict,
        "armed": bool(verdict.get("silent")) and bool(recipient),
        "subject": subject,
        "recipient_configured": bool(recipient),
        "evidence_lines": len(_evidence_lines(verdict)),
        "mail_bytes": len(html) + len(plain),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Alarm if the Telegram bot is silent")
    parser.add_argument("--repo", default=None, help="owner/name of the repository")
    parser.add_argument("--no-send", action="store_true", help="judge only, do not mail")
    parser.add_argument("--json", action="store_true", help="print the raw verdict")
    parser.add_argument("--selftest", action="store_true",
                        help="prove the outage path is armed, without mailing anyone")
    args = parser.parse_args(argv)

    if args.selftest:
        report = selftest(repo=args.repo)
        verdict = report.get("verdict") or {}
        print(f"[selftest] synthetic outage  : 90 min old trace + a dead window")
        print(f"[selftest] verdict           : {verdict.get('severity')} / {verdict.get('reason')}")
        print(f"[selftest] subject           : {report.get('subject')}")
        print(f"[selftest] evidence lines    : {report.get('evidence_lines')}")
        print(f"[selftest] recipient         : {'configured (not shown)' if report.get('recipient_configured') else 'NOT CONFIGURED'}")
        if report.get("armed"):
            print("[selftest] the outage path is armed — no mail was sent.")
            return 0
        print("[selftest] WARNING: the alarm is NOT armed — a real outage would go unnoticed.")
        return 1

    report = run_silence_alarm(repo=args.repo, send=not args.no_send)
    verdict = report.get("verdict") or {}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    print("[alarm] bot health trace : " + str((verdict.get("health") or {}).get("last_poll_at") or "—"))
    print("[alarm] trace age        : " + (f"{verdict['age_min']} min" if verdict.get("age_min") is not None else "—"))
    window = verdict.get("window") or {}
    print(f"[alarm] window           : {window.get('state')} ({window.get('detail') or '—'})")
    print(f"[alarm] verdict          : {verdict.get('severity')} / {verdict.get('reason')}")
    if report.get("sent"):
        print("[alarm] the operator was notified by email.")
    elif report.get("reason") == "cooldown":
        print(f"[alarm] already reported {report.get('minutes_since_last_alert')} min ago — not mailing again.")
    elif verdict.get("silent"):
        print(f"[alarm] silence is real but no mail was sent: {report.get('outcome', {}).get('status')}")
    else:
        print("[alarm] the bot is alive — no mail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
