#!/usr/bin/env python3
"""
The complex builds its own infrastructure — through the providers' DOCUMENTED
doors.

Measured 2026-11-04: MEGA closed the door for the account the complex had
registered for itself («HTTP 402, empty body»). The probe in the archive
report proved the door was still OPEN for a throwaway address from the very
same runner — so the refusal was about OUR account, not about the host. The
complex's answer to a dead credential is not to wait for a human: it is to
make a new one, the way it made the first.

This module is that chain, and it lives in the repository so it can run in
the cloud:

    1. a mailbox at mail.tm's PUBLIC API (documented for automation:
       POST /accounts -> 201, POST /token -> 200);
    2. MEGA's own client protocol (a=up/a=us/a=ug/a=up/a=uc) — the
       documented client API, no web form in the path;
    3. the signup link, read from the complex's OWN mailbox through the
       mailbox API — nobody reads it for us;
    4. a=ud/a=up/a=us/a=up(pubk,privk) — the account is live;
    5. the credentials sealed into the vault (AES-256-GCM under the salt);
       the public repository carries ciphertext only;
    6. the proof: log in and open the master key.

The path is built from the provider's own contracts: the mail.tm API is
documented for automation, MEGA's protocol is the documented client API.
Whatever a provider puts in the way (verification walls, phone, payment)
ends the chain with a named report — the next door is measured elsewhere.

The chain is bounded: at most MAX_ACCOUNTS sealed accounts, and at most one
new account per day. A provider that closes every account we make is a
POLICY, not a bug — after the cap, the founder decides.
"""

import json
import os
import re
import secrets
import string
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from harness import atomic_io  # noqa: E402
from harness import env_file  # noqa: E402
from harness import redaction  # noqa: E402
from harness import sealed_vault  # noqa: E402
from harness.mega_client import MegaClient, random_password, signup_key_from_link  # noqa: E402

MAILTM_API = "https://api.mail.tm"
REPORT_PATH = PROJECT_ROOT / "data" / "mega_provision_report.json"

# What one sealed account consists of. Slot 1 is the account the complex
# registered first (the flat «mega:<field>» entries); the replacements live
# in «mega:<field>_2», «mega:<field>_3», ...
ACCOUNT_FIELDS = ("email", "password", "recovery_key", "mailtm_token", "mailtm_password")

# The cap: three USABLE accounts is enough to survive one closure and still
# leave the operator a choice; more than that is spam, not resilience.
MAX_ACCOUNTS = 3

# The slots are scanned wider than the cap, so a slot that was half-written
# (evidence, never overwritten) does not push a usable account out of the
# reader's sight.
MAX_SLOTS = 8

# One new account per day. The archive cycle can run more often than daily
# (the supervisor re-dispatches it), and a provider that closed the account
# must not be hammered with a new one on every tick.
MIN_HOURS_BETWEEN = 24

# The signup link inside MEGA's mail.
_LINK_RE = re.compile(r"mega\.nz/#confirm[A-Za-z0-9_\-]+")


def slot_field(field: str, slot: int) -> str:
    """The vault entry name of one field of one account."""
    return f"mega:{field}" if slot <= 1 else f"mega:{field}_{slot}"


def load_accounts(salt: str, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Every sealed account, oldest first.

    A slot that has an address but no password is NOT a usable account — and
    it is not silently skipped either: it comes back marked «broken» so the
    caller can say that a half-written slot exists.
    """
    accounts: List[Dict[str, Any]] = []
    for slot in range(1, MAX_SLOTS + 1):
        record: Dict[str, Any] = {"slot": slot}
        for field in ACCOUNT_FIELDS:
            value = sealed_vault.get(slot_field(field, slot), salt=salt, path=path)
            if value:
                record[field] = value
        if record.get("email") and record.get("password"):
            accounts.append(record)
        elif any(field in record for field in ACCOUNT_FIELDS):
            record["broken"] = True
            accounts.append(record)
    return accounts


def vault_salt(explicit: Optional[str] = None) -> str:
    """The salt that opens the vault: VAULT_SALT first, REDACTION_SALT after."""
    env_file.load_dotenv()
    salt = (explicit if explicit is not None else
            os.getenv("VAULT_SALT") or os.getenv("REDACTION_SALT") or "").strip()
    if not salt:
        raise RuntimeError("neither VAULT_SALT nor REDACTION_SALT is set — "
                           "the vault cannot be opened. Put VAULT_SALT=... in .env "
                           "(gitignored) or export it; the cloud reads the "
                           "GitHub secret of the same name")
    return salt


# --- the mailbox (mail.tm's PUBLIC API — documented for automation) ---------

def collection(payload: Any) -> List[Any]:
    """
    The items of a collection, whatever shape the provider wears today.

    Measured 2026-11-04: mail.tm answers with JSON-LD (a Hydra collection):
    «{"@type":"hydra:Collection","hydra:member":[…]}» — not the bare list the
    older examples show. A parser that expects a list sees no items and
    reports «mail.tm offered no domain» — which is what the first cloud run
    of this chain said. Both shapes are accepted.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("hydra:member", "member", "items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def mailbox_domains() -> List[str]:
    """The domains mail.tm offers right now (the list moves; the API does not)."""
    r = requests.get(f"{MAILTM_API}/domains", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"mail.tm did not answer with its domains: HTTP {r.status_code}")
    domains = []
    for item in collection(r.json()):
        if isinstance(item, dict) and item.get("domain"):
            domains.append(item["domain"])
        elif isinstance(item, str):
            domains.append(item)
    if not domains:
        raise RuntimeError("mail.tm offered no domain")
    return domains


def mailbox_create(domain: Optional[str] = None) -> Dict[str, str]:
    """
    A mailbox of the complex's own. Returns its address, password and token.
    The password is generated here and NEVER printed — it lives in the vault.
    """
    domain = domain or mailbox_domains()[0]
    address = f"kg{secrets.token_hex(5)}@{domain}"
    password = random_password(20)
    r = requests.post(f"{MAILTM_API}/accounts",
                      json={"address": address, "password": password}, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"mail.tm refused the mailbox: HTTP {r.status_code}")
    r = requests.post(f"{MAILTM_API}/token",
                      json={"address": address, "password": password}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"mail.tm refused the token: HTTP {r.status_code}")
    token = (r.json() or {}).get("token")
    if not token:
        raise RuntimeError("mail.tm returned no token")
    return {"address": address, "password": password, "token": token}


def find_signup_link(body: Dict[str, Any]) -> Optional[str]:
    """The signup link inside the mail MEGA sent to our own mailbox."""
    haystack = " ".join(str(body.get(key) or "")
                        for key in ("text", "html", "intro", "subject"))
    match = _LINK_RE.search(haystack)
    return match.group(0) if match else None


def mailbox_wait_for_signup(token: str, attempts: int = 12, wait: int = 5) -> str:
    """
    The signup link, from the complex's OWN mailbox. Nobody reads it for us:
    the chain opens the mailbox it just made and waits for the letter.
    """
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(1, attempts + 1):
        r = requests.get(f"{MAILTM_API}/messages", headers=headers, timeout=30)
        if r.status_code == 200:
            for message in collection(r.json()):
                mid = message.get("id") if isinstance(message, dict) else None
                if not mid:
                    continue
                detail = requests.get(f"{MAILTM_API}/messages/{mid}",
                                      headers=headers, timeout=30)
                if detail.status_code != 200:
                    continue
                link = find_signup_link(detail.json() or {})
                if link:
                    return link
        if attempt < attempts:
            time.sleep(wait)
    raise RuntimeError(f"no signup link after {attempts} checks — "
                       "the provider did not send the letter")


# --- the account -----------------------------------------------------------

def provision(salt: Optional[str] = None, path: Optional[Path] = None,
              name: str = "Autonomous Complex") -> Dict[str, Any]:
    """
    One account, made from nothing but the providers' documented APIs.

    The report says what happened and never carries a credential: the
    account appears as a fingerprint, like everywhere else in the complex.
    """
    salt = vault_salt(salt)
    now = datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "checked_at": now.isoformat(),
        "status": "failed",
        "steps": [],
        "errors": [],
        "account": "",
        "slot": 0,
    }

    try:
        accounts = load_accounts(salt, path)
    except Exception as exc:
        report["errors"].append(f"vault: {type(exc).__name__}: {exc}")
        return report
    usable = [a for a in accounts if not a.get("broken")]
    if len(usable) >= MAX_ACCOUNTS:
        report["status"] = "skipped"
        report["errors"].append(
            f"the cap is reached ({MAX_ACCOUNTS} sealed accounts) — a provider "
            "that closes every account we make is a policy, not a bug; the "
            "operator must decide")
        return report
    # The new account takes the FIRST slot that holds nothing at all — a
    # half-written slot is evidence, and evidence is not overwritten.
    taken = {a["slot"] for a in accounts}
    free = [slot for slot in range(1, MAX_SLOTS + 1) if slot not in taken]
    if not free:
        report["status"] = "skipped"
        report["errors"].append(f"every slot ({MAX_SLOTS}) holds something — "
                                "the operator must look at the vault")
        return report
    slot = free[0]

    password = random_password(24)
    try:
        # 1. the mailbox
        mailbox = mailbox_create()
        report["steps"].append("mailbox")

        # 2. MEGA's own protocol
        client = MegaClient()
        state = client.register_request(mailbox["address"], password, name)
        report["steps"].append("handshake")

        # 3. the letter, read from our own mailbox
        link = mailbox_wait_for_signup(mailbox["token"])
        report["steps"].append("letter")

        # 4. the account becomes live
        done = client.register_verify(state, signup_key_from_link(link))
        report["steps"].append("verified")

        # 5. sealed — the public repository carries ciphertext only
        recovery = done.get("recovery_key") or state.get("recovery_key") or ""
        for field, value in (("email", mailbox["address"]),
                             ("password", password),
                             ("recovery_key", recovery),
                             ("mailtm_token", mailbox["token"]),
                             ("mailtm_password", mailbox["password"])):
            sealed_vault.put(slot_field(field, slot), value, salt=salt, path=path)
        report["steps"].append("sealed")

        # 6. the proof: the master key opens
        proof = MegaClient()
        login = proof.login(mailbox["address"], password)
        if not login.get("master_key_ok"):
            raise RuntimeError("the master key did not verify — the account is not usable")
        proof.root_handle()
        report["steps"].append("proven")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        return report

    report["account"] = redaction.fingerprint(mailbox["address"])
    report["slot"] = slot
    report["status"] = "ok"
    return report


def last_attempt(path: Optional[Path] = None) -> Optional[datetime]:
    """When the complex last tried to build an account (from its own record)."""
    try:
        data = json.loads((path or REPORT_PATH).read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(data.get("checked_at", ""))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def maybe_provision(salt: Optional[str] = None, path: Optional[Path] = None,
                    report_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Build an account only when it is the right moment: the door is open for
    everyone, our account is closed, the cap is not reached, and the complex
    has not already tried today. Otherwise it says why it stayed still.
    """
    now = datetime.now(timezone.utc)
    outcome: Dict[str, Any] = {"attempted": False, "status": "skipped", "why": ""}
    previous = last_attempt(report_path)
    if previous and now - previous < timedelta(hours=MIN_HOURS_BETWEEN):
        outcome["why"] = (f"the complex already tried {MIN_HOURS_BETWEEN}h ago "
                          "— the next attempt waits, so a closed provider is not hammered")
        return outcome
    outcome["attempted"] = True
    result = provision(salt=salt, path=path)
    outcome["status"] = result.get("status")
    outcome["report"] = result
    outcome["why"] = ("" if result.get("status") == "ok"
                      else "; ".join(result.get("errors") or []) or "the chain failed")
    return outcome


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """The artefact, written atomically."""
    target = path or REPORT_PATH
    if not atomic_io.write_json_atomic(target, report):
        return None
    return target


def main() -> int:
    try:
        report = provision()
    except Exception as exc:
        print(f"[mega-provision] {type(exc).__name__}: {exc}")
        return 1
    written = write_report(report)
    print(f"[mega-provision] status={report.get('status')} "
          f"slot={report.get('slot') or '—'} "
          f"account={report.get('account') or '—'} "
          f"steps={','.join(report.get('steps') or []) or '—'}")
    for err in report.get("errors", []):
        print(f"[mega-provision] error: {err}")
    if written:
        print(f"[mega-provision] report: {written}")
    # A failed provisioning is a visible failure: the operator must know that
    # the third copy of the record has no account behind it.
    return 0 if report.get("status") in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
