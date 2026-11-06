#!/usr/bin/env python3
"""
The complex's own mailboxes and its cloud disk — one place that knows how to
reach them.

Three independent providers, because one dead provider must never silence the
complex. The addresses are NOT written here: this repository is public, and
the operator's addresses live in `.env`, in the sealed vault and in the
operator's own letter — nowhere else (test 83 holds that line).

  proton  Free. Carries the Proton Drive cloud disk as well: the login is
          shared, the surfaces are separate.
  tuta    Free. The non-Proton verification inbox; its recovery code is the
          only way back in.
  mailtm  The only one with a plain REST API, so it is the only mailbox the
          complex can READ unattended — the other two are receive-only for
          a human.

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

from harness import env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Loads .env into the environment — the project's ONE loader (env_file.py)."""
    env_file.load_dotenv()


_load_dotenv()

MAILTM_API = "https://api.mail.tm"

# --------------------------------------------------------------------------
# The tiers of a mailbox — and the difference between them is the whole
# point of the inventory.
#
# Measured 2026-11-04 (ADL-031): the complex registered 10 mailboxes of its
# own in one drill — 5 with credentials (mail.tm) and 5 session-assigned
# (guerrillamail, no credentials at all). The record then said «10 created»
# and stopped there. That is the kind of sentence the complex is not allowed
# to write: a mailbox it can never log into again is not a mailbox it may
# count on, and a mailbox whose credentials sit in one untracked file dies
# with that file.
#
# So every mailbox carries a TIER, and the inventory says it out loud:
#   api-readable      the complex reads it unattended, over an API
#   human-only        a person must open it; the complex only sends to it
#   durable (sealed)  the complex registered it AND sealed the credentials
#                     in the vault — it survives this machine
#   durable, NOT sealed
#                     registered, credentials only in the local untracked
#                     file — it dies with that file, and the inventory says
#                     so instead of counting it as an asset
#   session-only      no credentials were ever issued; it dies with the
#                     session and can never be recovered
# --------------------------------------------------------------------------

TIER_API = "api-readable (unattended)"
TIER_HUMAN = "human-only (a person must read it)"
TIER_DURABLE_SEALED = "durable (sealed in the vault)"
TIER_DURABLE_LOCAL = "durable but NOT sealed (dies with this machine)"
TIER_SESSION = "session-only (no credentials, dies with the session)"

# The keeper's untracked working copy of the mailboxes the complex
# registered itself. It is in .gitignore: it holds passwords in the clear.
# Sealing it into the vault is what turns a registration into an asset.
SELF_PROVISIONED_FILE = PROJECT_ROOT / "self_provisioned_mailboxes.json"

# The vault's own naming for the complex's own mailboxes.
SELF_PROVISIONED_PREFIX = "selfprov:"

# The MEGA chain (harness/mega_provision.py) creates a mail.tm mailbox for
# the account it registers, and seals it under the «mega:» names: the
# account's email IS that mailbox's address, and its credentials are
# mega:mailtm_password / mega:mailtm_token.
#
# Measured 2026-11-06: an inventory that counted only the «selfprov:»
# names reported «свои, запечатаны в vault: 0» while the vault held TWO
# durable mailboxes of the complex's own (mega: and mega:_2). That is a
# FALSE ZERO — the worst kind of lie in this project, because the number
# was right and the thing it described was not.
MEGA_MAILBOX_EMAIL = "mega:email"
MEGA_MAILBOX_PASSWORD = "mega:mailtm_password"
MEGA_MAILBOX_TOKEN = "mega:mailtm_token"


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
            "tier": TIER_API if box.get("readable") else TIER_HUMAN,
        })
    d = drive()
    rows.append({
        "provider": d["provider"],
        "address_fp": _fingerprint(d.get("address", "")),
        "address_masked": _mask(d.get("address", "")),
        "password": _mask(d.get("password", "")),
        "tfa": "on" if d.get("tfa_secret") else "off",
        "tier": "cloud disk (the complex keeps it by itself)",
    })
    for box in self_provisioned():
        rows.append({
            "provider": f"self:{box.get('provider') or 'mailtm'}",
            "address_fp": _fingerprint(box.get("address", "")),
            "address_masked": _mask(box.get("address", "")),
            "password": _mask(box.get("password", "")),
            "readable": bool(box.get("token")),
            "tier": box.get("tier") or TIER_DURABLE_LOCAL,
        })
    return rows


# --------------------------------------------------------------------------
# The complex's own mailboxes (ADL-031)
# --------------------------------------------------------------------------

def vault_salt(explicit: Optional[str] = None) -> str:
    """The salt that opens the vault: VAULT_SALT first, REDACTION_SALT after."""
    env_file.load_dotenv()
    salt = (explicit if explicit is not None else
            os.getenv("VAULT_SALT") or os.getenv("REDACTION_SALT") or "").strip()
    if not salt:
        raise RuntimeError("neither VAULT_SALT nor REDACTION_SALT is set — "
                           "the vault cannot be opened")
    return salt


def read_self_provisioned_file(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    The keeper's untracked working copy of the mailboxes it registered.

    The format is a list of rows (a dict with a «mailboxes» key is also
    accepted):
      {"provider": "mailtm", "address": "...", "password": "...",
       "token": "...", "tier": "durable" | "session"}
    A row without a password is a session-only mailbox — the provider never
    issued one, and no amount of effort will make it recoverable.
    """
    target = Path(path) if path else SELF_PROVISIONED_FILE
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("mailboxes") or data.get("accounts") or []
    rows: List[Dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        address = str(item.get("address") or "").strip()
        if not address:
            continue
        rows.append({
            "provider": str(item.get("provider") or "mailtm").strip().lower(),
            "address": address,
            "password": str(item.get("password") or "").strip(),
            "token": str(item.get("token") or "").strip(),
            "tier": str(item.get("tier") or "").strip().lower(),
        })
    return rows


def self_provisioned_key(address: str) -> str:
    """
    The vault's name for one of the complex's own mailboxes.

    Derived from the ADDRESS, not from a row number: a row number changes
    when the file is reordered or a session-only row is added, and the
    sealed credentials must not silently swap places.
    """
    return f"{SELF_PROVISIONED_PREFIX}{_fingerprint(address).replace('sha256:', '')}:"


def _sealed_self_provisioned_names() -> List[str]:
    """The vault names of the complex's own mailboxes (no salt needed)."""
    from harness import sealed_vault
    try:
        names = sealed_vault.entry_names()
    except Exception:
        return []
    return [n for n in names
            if n.startswith(SELF_PROVISIONED_PREFIX) and n.endswith(":address")]


def sealed_self_provisioned_count() -> int:
    """
    How many of the complex's own mailboxes are sealed in the vault.

    The COUNT comes from the entry NAMES — which are plaintext in the
    ciphertext file — so the number is known even when the salt is not at
    hand. «I cannot open the vault» must never become «there are none».
    """
    return len(_sealed_self_provisioned_names())


def sealed_own_mailboxes() -> List[Dict[str, str]]:
    """
    Every mailbox of the complex's own that is SEALED — one record per
    mailbox, with the vault names that hold it.

    Two layouts live in the vault, and BOTH are knowable from the entry
    names alone (no salt needed, because the names are plaintext):
      - «selfprov:<fingerprint>:address» — the drill's registrations, sealed
        by scripts/seal_mail_accounts.py;
      - «mega:email» + «mega:mailtm_password» — the MEGA chain's own
        mail.tm mailbox. The password's presence is the SIGNATURE that the
        complex made this mailbox itself; an account registered against
        somebody else's address carries no such entry and is not counted.

    The count is the fact that matters: measured 2026-11-06, the vault
    holds two of the complex's own mailboxes, and an inventory that only
    knew the «selfprov:» names said zero.
    """
    from harness import sealed_vault
    try:
        names = set(sealed_vault.entry_names())
    except Exception:
        return []
    found: List[Dict[str, str]] = []
    for name in sorted(n for n in names
                       if n.startswith(SELF_PROVISIONED_PREFIX)
                       and n.endswith(":address")):
        base = name[:-len(":address")]
        found.append({"kind": "selfprov", "address": name,
                      "password": base + ":password",
                      "token": base + ":token"})
    for name in sorted(n for n in names
                       if n.startswith(MEGA_MAILBOX_EMAIL)
                       and (n == MEGA_MAILBOX_EMAIL
                            or n[len(MEGA_MAILBOX_EMAIL):].startswith("_"))):
        suffix = name[len(MEGA_MAILBOX_EMAIL):]
        password = MEGA_MAILBOX_PASSWORD + suffix
        if password not in names:
            # The account was registered against somebody else's mailbox —
            # not a mailbox of the complex's own.
            continue
        found.append({"kind": "mega", "address": name,
                      "password": password,
                      "token": MEGA_MAILBOX_TOKEN + suffix})
    return found


def sealed_own_mailbox_count() -> int:
    """How many mailboxes of the complex's own are sealed — knowable without the salt."""
    return len(sealed_own_mailboxes())


def self_provisioned(salt: Optional[str] = None,
                     path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    The complex's own mailboxes, each with the tier it really has.

    Two sources, and the vault wins: a sealed mailbox survives this
    machine; a mailbox that only lives in the local file dies with it — and
    the inventory says exactly that instead of counting it as an asset.
    """
    from harness import sealed_vault
    rows: List[Dict[str, str]] = []
    sealed = sealed_own_mailboxes()
    names = [item["address"] for item in sealed]
    sealed_names = set(names)
    try:
        the_salt = vault_salt(salt)
    except Exception:
        the_salt = ""
    if the_salt and sealed:
        wanted: List[str] = []
        for item in sealed:
            wanted.extend([item["address"], item["password"], item["token"]])
        opened = sealed_vault.open_many(wanted, salt=the_salt)
        for item in sealed:
            address = (opened.get(item["address"]) or "").strip()
            if not address:
                continue
            rows.append({
                "provider": "mailtm",
                "address": address,
                "password": (opened.get(item["password"]) or "").strip(),
                "token": (opened.get(item["token"]) or "").strip(),
                "tier": TIER_DURABLE_SEALED,
            })

    seen = {row["address"] for row in rows}
    for item in read_self_provisioned_file(path):
        if item["address"] in seen:
            continue
        if item.get("tier") == "session" or not item.get("password"):
            item["tier"] = TIER_SESSION
        elif self_provisioned_key(item["address"]) + "address" in sealed_names:
            # The vault's entry NAME is derived from the address, so the
            # FACT that this mailbox is sealed is knowable even without the
            # salt. «I cannot open the vault» must not become «this mailbox
            # is not sealed» — that is exactly the lie the tiers exist to
            # prevent.
            item["tier"] = TIER_DURABLE_SEALED
        else:
            item["tier"] = TIER_DURABLE_LOCAL
        rows.append(item)
        seen.add(item["address"])
    return rows


def mailbox_tiers(salt: Optional[str] = None) -> Dict[str, Any]:
    """
    The inventory in numbers — what the complex can actually use.

    This is what the health letter shows: how many mailboxes are readable
    unattended, how many only a person can open, how many the complex
    registered itself and kept, and how many were session-only all along.

    The addresses come from the environment first and from the sealed vault
    after — the cloud's inventory IS the vault. And when neither can be
    read, the answer is «not measured», never «zero»: a cycle that could
    not look must not report an empty shelf.
    """
    from harness import sealed_vault
    env_names = {"proton": "PROTON_MAIL_ADDRESS",
                 "tuta": "TUTA_ADDRESS",
                 "mailtm": "MAILTM_ADDRESS"}
    vault_names = {"proton": "proton:address",
                   "tuta": "tuta:address",
                   "mailtm": "mailtm:address"}
    addresses = {key: (os.getenv(name) or "").strip()
                 for key, name in env_names.items()}
    if not all(addresses.values()):
        try:
            the_salt = vault_salt(salt)
        except Exception:
            the_salt = ""
        if the_salt:
            try:
                opened = sealed_vault.open_many(list(vault_names.values()),
                                                salt=the_salt)
            except Exception:
                opened = {}
            for key, vault_name in vault_names.items():
                if not addresses.get(key):
                    addresses[key] = (opened.get(vault_name) or "").strip()

    measured = any(addresses.values())
    readable = 1 if addresses.get("mailtm") else 0
    human = sum(1 for key in ("proton", "tuta") if addresses.get(key))
    own = self_provisioned(salt)
    local = sum(1 for box in own if box.get("tier") == TIER_DURABLE_LOCAL)
    session = sum(1 for box in own if box.get("tier") == TIER_SESSION)
    # The SEALED count comes from the vault's entry NAMES, not from the rows:
    # the rows can only show the vault when the salt opens it, and a closed
    # vault must never turn a sealed mailbox into «not sealed». Measured
    # 2026-11-06: the rows said 0 while the names said 2 — a false zero,
    # because only the «selfprov:» layout was being looked at.
    sealed = sealed_own_mailbox_count()
    # Is the keeper's working file of registrations reachable from HERE?
    # It is not tracked (it holds plaintext passwords), so on the cloud
    # runner it is absent by design — and «I see none of my own mailboxes»
    # is a different fact from «I have none». The letter must be able to
    # tell the two apart.
    try:
        file_present = Path(SELF_PROVISIONED_FILE).exists()
    except Exception:
        file_present = False
    return {
        "measured": measured,
        "api_readable": readable,
        "human_only": human,
        "own_durable_sealed": sealed,
        "own_durable_local": local,
        "own_session": session,
        "own_total": sealed + local + session,
        "own_file_present": file_present,
        # The count of sealed entries is knowable without the salt — the
        # names are plaintext — so this number never hides behind «the
        # vault is closed». Both layouts count here: the drill's
        # «selfprov:» names and the MEGA chain's «mega:» mailboxes.
        "sealed_in_vault": sealed_own_mailbox_count(),
        "sealed_self_provisioned": sealed_self_provisioned_count(),
    }


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
            print(f"{'':<14} tier: {row.get('tier', '-')}")
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
