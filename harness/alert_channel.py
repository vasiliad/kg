#!/usr/bin/env python3
"""
The backup alert channel: a GitHub Issue when a letter fails to deliver.

The primary channel is the letter itself — and that is exactly what breaks
when Gmail refuses, the key expires or the SMTP password rots. An alarm
that lives only in the log next to the failure is read AFTER the damage;
an Issue is a second pipe with a different failure mode (API, not SMTP),
and it is visible on the phone without opening anything.

Privacy contract: the Issue is PUBLIC — the repository is public. The
title carries only the label and the date; the body carries the error
AFTER `redaction.mask_emails_deep`. A Gmail 5xx quoting the recipient's
address must never reach the issue text raw.

Dedupe: one issue per label per UTC day. A failing day that fires the
dispatch twice must not open two issues.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from harness import redaction

ALERT_MARKER = "delivery-alert"


def _gh_env() -> tuple[str, str]:
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    repo = os.getenv("GITHUB_REPOSITORY") or os.getenv("GH_REPO") or ""
    return token.strip(), repo.strip()


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def issue_title(label: str, when: Optional[datetime] = None) -> str:
    """One issue per label per UTC day — the title IS the dedupe key."""
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return f"⚠️ [{ALERT_MARKER}] Письмо «{label}» не доставлено — {day}"


def _issue_exists(token: str, repo: str, title: str) -> bool:
    try:
        res = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            headers=_headers(token),
            params={"state": "open", "per_page": 50},
            timeout=10,
        )
        if res.status_code != 200:
            return False
        for issue in res.json() or []:
            if issue.get("title") == title:
                return True
    except Exception:
        return False
    return False


def raise_delivery_alert(
    label: str,
    error: str,
    recipient_tag: str = "",
    dry_run: bool = False,
) -> bool:
    """
    Opens the backup alert Issue. True = raised (or already open).

    A failure here is LOGGED, never raised: the backup channel must not
    take the dispatch cycle down with it.
    """
    token, repo = _gh_env()
    if not token or not repo:
        print("[alert] GH_TOKEN/GITHUB_REPOSITORY not set — the alert stays in the log only")
        return False

    title = issue_title(label)
    if _issue_exists(token, repo, title):
        print(f"[alert] an alert for «{label}» is already open today")
        return True

    masked = redaction.mask_emails_deep(str(error or "причина не зафиксирована"))
    body = (
        f"Автономный комплекс не смог доставить письмо.\n\n"
        f"- **Что:** {label}\n"
        f"- **Когда:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"- **Кому:** {recipient_tag or 'получатель не зафиксирован'}\n"
        f"- **Причина (маскировано):** {masked[:600]}\n\n"
        f"Письмо — основной канал. Этот Issue — резервный: он открыт потому, "
        f"что основной не сработал. Проверить секреты `GMAIL_SENDER` / "
        f"`GMAIL_APP_PASSWORD` и журнал `nim_history.log`."
    )

    if dry_run:
        print(f"[alert] dry-run: would open the issue «{title}»")
        return True

    try:
        res = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers=_headers(token),
            json={"title": title, "body": body, "labels": [ALERT_MARKER]},
            timeout=15,
        )
        if res.status_code in (201, 200):
            print(f"[alert] backup alert raised: {res.json().get('html_url', '')}")
            return True
        print(f"[alert] GitHub refused the issue: HTTP {res.status_code}")
        return False
    except Exception as e:
        print(f"[alert] could not raise the backup alert: {type(e).__name__}")
        return False
