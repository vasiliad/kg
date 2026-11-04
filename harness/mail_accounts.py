#!/usr/bin/env python3
"""
The complex's own mailboxes and its cloud disk — one place that knows how to
reach them.

Three independent providers, because one dead provider must never silence the
complex:

  proton  vanvasiliadi@protonmail.com  Free. Carries the Proton Drive cloud
          disk as well: the login is shared, the surfaces are separate.
  tuta    ivan.vasiliadi@tutamail.com  Free. The non-Proton verification
          inbox; its recovery code is the only way back in.
  mailtm  xray1793739979@fldubai.com   The only one with a plain REST API,
          so it is the only mailbox the complex can READ unattended — the
          other two are write/receive-only for a human.

WHERE THE CREDENTIALS LIVE (this repository is PUBLIC):
  - .env            — local runs; gitignored, never committed.
  - the vault       — cloud runs; data/.sealed_vault.json carries ciphertext
                      only, opened with VAULT_SALT / REDACTION_SALT.
  - GitHub secrets  — the cloud's environment.
This module never prints a password and never writes a secret to a tracked
file. `describe()` returns redacted fingerprints, not addresses.

USAGE
  python -m harness.mail_accounts --list          # redacted inventory
  python -m harness.mail_accounts --inbox         # newest mail.tm messages
  python -m harness.mail_accounts --inbox --grep github
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Loads .env into the environment (the project's own idiom, telegram_bot.py)."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        pass


_load_dotenv()

MAILTM_API = "https://api.mail.tm"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


def _fingerprint(value: str) -> str:
    """A short, one-way tag for an address: enough to recognise, never to read."""
    if not value:
        return ""
    import hashlib
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _mask(value: str) -> str:
    """Shows the shape of a secret without revealing it."""
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def mailbox(name: str) -> Dict[str, str]:
    """One mailbox by provider name ('proton' | 'tuta' | 'mailtm')."""
    name = (name or "").lower()
    if name == "proton":
        return {
            "provider": "proton",
            "address": _env("PROTON_MAIL_ADDRESS"),
            "username": _env("PROTON_MAIL_USERNAME"),
            "password": _env("PROTON_MAIL_PASSWORD"),
            "plan": _env("PROTON_MAIL_PLAN", "Free"),
            "readable": False,
        }
    if name == "tuta":
        return {
            "provider": "tuta",
            "address": _env("TUTA_ADDRESS"),
            "password": _env("TUTA_PASSWORD"),
            "recovery_code": _env("TUTA_RECOVERY_CODE"),
            "readable": False,
        }
    if name == "mailtm":
        return {
            "provider": "mailtm",
            "address": _env("MAILTM_ADDRESS"),
            "password": _env("MAILTM_PASSWORD"),
            "token": _env("MAILTM_TOKEN"),
            "readable": True,
        }
    raise KeyError(f"unknown mailbox provider: {name!r}")


def mailboxes() -> List[Dict[str, str]]:
    """Every mailbox the complex owns, in provider order."""
    return [mailbox(p) for p in ("proton", "tuta", "mailtm")]


def drive() -> Dict[str, str]:
    """The cloud disk: Proton Drive, sharing the Proton login."""
    return {
        "provider": "proton-drive",
        "address": _env("PROTON_DRIVE_ADDRESS") or _env("PROTON_MAIL_ADDRESS"),
        "password": _env("PROTON_DRIVE_PASSWORD") or _env("PROTON_MAIL_PASSWORD"),
        "tfa_secret": _env("PROTON_DRIVE_2FA_SECRET"),
        "surface": "Proton Drive (5 GB on the Free plan)",
    }


def describe() -> List[Dict[str, Any]]:
    """A redacted inventory — safe to print into a report or a log."""
    rows: List[Dict[str, Any]] = []
    for box in mailboxes():
        rows.append({
            "provider": box["provider"],
            "address_fp": _fingerprint(box.get("address", "")),
            "address_masked": _mask(box.get("address", "")),
            "password": _mask(box.get("password", "")),
            "readable": box.get("readable", False),
        })
    d = drive()
    rows.append({
        "provider": d["provider"],
        "address_fp": _fingerprint(d.get("address", "")),
        "address_masked": _mask(d.get("address", "")),
        "password": _mask(d.get("password", "")),
        "tfa": "on" if d.get("tfa_secret") else "off",
    })
    return rows


def _api(path: str, token: str, timeout: int = 30) -> Any:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    req = urllib.request.Request(MAILTM_API + path, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        members = payload.get("hydra:member")
        if isinstance(members, list):
            return members
    return []


def _login(address: str, password: str) -> str:
    """Exchanges the mailbox credentials for a session token."""
    body = json.dumps({"address": address, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        MAILTM_API + "/token", data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["token"]


def mailtm_token() -> str:
    """A live mail.tm token: the stored one, or a fresh login if it expired."""
    stored = _env("MAILTM_TOKEN")
    if stored:
        try:
            _api("/me", stored, timeout=15)
            return stored
        except Exception:
            pass  # expired or revoked — fall through to a fresh login
    box = mailbox("mailtm")
    if not box.get("address") or not box.get("password"):
        raise RuntimeError("mail.tm credentials are missing (.env or vault)")
    return _login(box["address"], box["password"])


def inbox(limit: int = 10, wait_seconds: int = 0) -> List[Dict[str, Any]]:
    """
    The newest messages of the readable mailbox, newest first.

    wait_seconds makes the call POLL instead of SNAPSHOT: a verification code
    that was just sent usually needs a few seconds to arrive, and a one-shot
    read that runs too early reports «no mail» when the truth is «not yet».
    """
    deadline = time.time() + max(0, wait_seconds)
    token = mailtm_token()
    messages: List[Dict[str, Any]] = []
    while True:
        messages = _items(_api("/messages", token))
        if messages or time.time() >= deadline:
            break
        time.sleep(5)
    out: List[Dict[str, Any]] = []
    for msg in messages[:limit]:
        entry = {
            "id": msg.get("id"),
            "from": (msg.get("from") or {}).get("address", ""),
            "subject": msg.get("subject") or "",
            "created_at": msg.get("createdAt") or msg.get("created_at") or "",
        }
        try:
            detail = _api(f"/messages/{msg.get('id')}", token)
            blob = (detail.get("text") or "") + " " + " ".join(str(v) for v in (detail.get("html") or []))
            codes = re.findall(r"\b(\d{6,8})\b", blob)
            # The same code appears in the text part and in the HTML part:
            # keep the first occurrence of each, in order.
            entry["codes"] = list(dict.fromkeys(codes))[:5]
            entry["text"] = (detail.get("text") or "")[:300]
        except Exception:
            entry["codes"] = []
        out.append(entry)
    return out


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the complex's own mailboxes and cloud disk")
    ap.add_argument("--list", action="store_true", help="redacted inventory")
    ap.add_argument("--inbox", action="store_true", help="newest mail.tm messages")
    ap.add_argument("--grep", default="", help="only messages matching this text")
    ap.add_argument("--wait", type=int, default=0, help="poll up to N seconds")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    if args.list or not args.inbox:
        for row in describe():
            print(f"{row['provider']:<14} {row['address_masked']:<28} "
                  f"pw={row['password']:<14} readable={row.get('readable', False)} "
                  f"tfa={row.get('tfa', '-')}")
    if args.inbox:
        try:
            messages = inbox(limit=args.limit, wait_seconds=args.wait)
        except Exception as exc:
            print(f"inbox failed: {exc}")
            return 1
        for msg in messages:
            if args.grep and args.grep.lower() not in (msg["subject"] + " " + msg["from"]).lower():
                continue
            print(f"[{msg['created_at']}] {msg['from']} :: {msg['subject']}")
            if msg.get("codes"):
                print(f"    codes: {', '.join(msg['codes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
