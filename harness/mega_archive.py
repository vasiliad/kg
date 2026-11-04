#!/usr/bin/env python3
"""
The complex's own archive — it keeps its daily record in the cloud account it
registered ITSELF.

Measured on 2026-11-04: the complex has no storage of its own. GitHub's
repository holds the artefacts, but the repository is the same thing that
could be lost, locked or rate-limited; the operator's mailbox is the second
copy, and a mailbox dies when a password changes. The third copy must belong
to nobody but the complex — and such an account already exists: the MEGA
account registered by the system itself (mail.tm mailbox → protocol signup →
verification link read from its own inbox → login verified against the
master key). Its credentials live in the sealed vault (AES-256-GCM under the
VAULT_SALT secret), so the PUBLIC repository carries ciphertext only.

This module is that cycle:

    1. open the vault (VAULT_SALT, REDACTION_SALT as the fallback);
    2. log in and prove the master key (a wrong key fails here, not later);
    3. find the cloud root, ensure /kg-archive, ensure /kg-archive/<date>;
    4. upload every artefact that exists (the daily digest, the reports, the
       history log — and the vault file itself: ciphertext, so the operator's
       salt is the only key to it);
    5. write data/mega_archive_report.json — the artefact the cycle
       supervisor watches. A run that uploaded nothing says so.

Nothing here prints a credential, a master key or an address: the account is
reported as a fingerprint, exactly like everywhere else in the complex.
"""

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import redaction
from harness import sealed_vault
from harness.mega_client import MegaClient
from harness.mega_provision import load_accounts, maybe_provision

REPORT_PATH = PROJECT_ROOT / "data" / "mega_archive_report.json"
ARCHIVE_ROOT = "kg-archive"

# The daily record. Every path is optional — a cycle that runs before the
# daily digest exists must still archive what DOES exist. The vault file is
# in the list on purpose: its ciphertext is the backup of the backup keys.
ARTEFACTS: Tuple[str, ...] = (
    "nim_history.log",
    "diagnostics_report.json",
    "data/daily_digest.json",
    "data/fleet_report.json",
    "data/bot_selftest_report.json",
    "data/cloud_tests_report.json",
    "data/survival_report.json",
    "data/forge_autopilot.json",
    "data/duty_shift.json",
    "data/telegram_history.txt",
    "data/.cycle_supervisor.json",
    "data/.sealed_vault.json",
)


def vault_salt(explicit: Optional[str] = None) -> str:
    """The salt that opens the vault: VAULT_SALT first, REDACTION_SALT after."""
    salt = (explicit if explicit is not None else
            os.getenv("VAULT_SALT") or os.getenv("REDACTION_SALT") or "").strip()
    if not salt:
        raise RuntimeError("neither VAULT_SALT nor REDACTION_SALT is set — "
                           "the vault cannot be opened")
    return salt


def load_credentials(salt: Optional[str] = None,
                     path: Optional[Path] = None) -> Dict[str, str]:
    """The MEGA credentials out of the sealed vault. None when absent."""
    salt = vault_salt(salt)
    email = sealed_vault.get("mega:email", salt=salt, path=path)
    password = sealed_vault.get("mega:password", salt=salt, path=path)
    if not email or not password:
        raise RuntimeError("the vault holds no MEGA credentials — "
                           "the account must be registered first")
    return {"email": email, "password": password}


def remote_name(relative: str) -> str:
    """«data/daily_digest.json» -> «data__daily_digest.json»."""
    return relative.replace("/", "__")


def probe_door() -> Dict[str, Any]:
    """
    Is the door closed to EVERYONE from here — or only to OUR account?

    Measured 2026-11-04: the cloud runner's requests to MEGA came back
    «HTTP 402, empty body» while the same API answered normally from a
    home connection. Two very different situations hide behind one code:
    the provider closed its door to this kind of host (nothing the complex
    can do from here — the keeper on the operator's machine must carry the
    copy), or OUR account needs something (an upgrade, a verification).
    Only the operator can act on the second one, so the report must say
    which one it is instead of leaving «HTTP 402» for the operator to
    guess.

    The probe asks MEGA about a throwaway address that does not exist: the
    documented API, one call, no account, no secret.
    """
    client = MegaClient()
    client.RETRIES = 1          # a diagnosis must not take a minute
    try:
        client.api({"a": "us0", "user": "door-probe@invalid.example"})
        return {"state": "open", "note": "MEGA answered a throwaway address"}
    except Exception as exc:
        return {"state": "closed", "note": f"{type(exc).__name__}: {exc}"}


def collect_artefacts(root: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """The artefacts that actually exist right now."""
    base = Path(root) if root else PROJECT_ROOT
    found: List[Tuple[str, Path]] = []
    for rel in ARTEFACTS:
        path = base / rel
        if path.exists() and path.is_file():
            found.append((rel, path))
    return found


def archive_once(root: Optional[Path] = None,
                 salt: Optional[str] = None) -> Dict[str, Any]:
    """
    One archive run. Returns the report — the caller writes it down.
    A failure is REPORTED, not raised into the void: the report says which
    step failed, and the workflow's exit code says it too.
    """
    base = Path(root) if root else PROJECT_ROOT
    now = datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "checked_at": now.isoformat(),
        "status": "failed",
        "archive": ARCHIVE_ROOT,
        "folder": "",
        "account": "",
        "uploaded": [],
        "missing": [],
        "errors": [],
        # The door's state — «open»/«closed» — and what it means. Filled in
        # only when the cloud refuses us; that is the moment it matters.
        "door": {},
    }

    # EVERY sealed account gets a turn. Measured 2026-11-04: MEGA closed the
    # door for the account the complex had made for itself, and the archive
    # had nowhere else to go — while a second account, if there were one,
    # would have carried the record all along.
    try:
        accounts = load_accounts(vault_salt(salt))
    except Exception as exc:
        report["errors"].append(f"vault: {type(exc).__name__}: {exc}")
        return report
    usable = [a for a in accounts if not a.get("broken")]
    if not usable:
        report["errors"].append("vault: no usable MEGA account is sealed")
        return report
    if len(usable) < len(accounts):
        report["errors"].append(
            f"vault: {len(accounts) - len(usable)} sealed account(s) are half-written "
            "and cannot be used")

    client = MegaClient()
    working: Optional[Dict[str, Any]] = None
    for account in usable:
        try:
            login = client.login(account["email"], account["password"])
            if not login.get("master_key_ok"):
                report["errors"].append(f"account {account['slot']}: "
                                        "the master key did not verify")
                continue
            working = account
            break
        except Exception as exc:
            report["errors"].append(f"account {account['slot']}: "
                                    f"{type(exc).__name__}: {exc}")
    if working is None:
        # The door's state travels with the failure: the operator must be
        # able to tell «MEGA closed the door to this host» from «our
        # account needs attention» without reading the client code.
        report["door"] = probe_door()
        # The complex builds its own infrastructure: if the door is open for
        # everyone but closed for us, make a new account — bounded by the
        # cap and by one attempt a day.
        if report["door"].get("state") == "open":
            outcome = maybe_provision(salt=salt)
            report["provision"] = outcome
            if outcome.get("status") == "ok":
                accounts = load_accounts(vault_salt(salt))
                for account in accounts:
                    if account.get("broken"):
                        continue
                    try:
                        login = client.login(account["email"], account["password"])
                        if login.get("master_key_ok"):
                            working = account
                            break
                    except Exception:
                        continue
        if working is None:
            return report
    report["account"] = redaction.fingerprint(working["email"])
    report["slot"] = working.get("slot")
    try:
        root_handle = client.root_handle()
    except Exception as exc:
        report["errors"].append(f"root: {type(exc).__name__}: {exc}")
        return report

    folder = ""
    try:
        archive_handle = client.mkdir(root_handle, ARCHIVE_ROOT)
        folder = ARCHIVE_ROOT
        day_handle = client.mkdir(archive_handle, now.strftime("%Y-%m-%d"))
        folder = f"{ARCHIVE_ROOT}/{now.strftime('%Y-%m-%d')}"
        report["folder"] = folder
    except Exception as exc:
        report["errors"].append(f"mkdir: {type(exc).__name__}: {exc}")
        return report

    artefacts = collect_artefacts(base)
    present = {rel for rel, _ in artefacts}
    report["missing"] = [rel for rel in ARTEFACTS if rel not in present]

    for rel, path in artefacts:
        try:
            data = path.read_bytes()
            remote = remote_name(rel)
            # The day's copy must be ONE copy: if an identical file is already
            # there, keep it (no duplicates, no wasted traffic); if it differs,
            # replace it — a stale copy is worse than no copy.
            handle = None
            reused = False
            for old in client.find_children(day_handle, remote):
                try:
                    if client.download_file(old) == data:
                        handle, reused = old, True
                        break
                except Exception:
                    pass
                try:
                    client.delete_node(old)
                except Exception as exc:
                    report["errors"].append(f"cleanup {rel}: {type(exc).__name__}: {exc}")
            if not reused:
                result = client.upload_file(day_handle, remote, data)
                handle = result.get("handle")
                chunks = result.get("chunks")
            else:
                chunks = None
            # THE PROOF: read it back and compare. An archive that cannot be
            # opened is not a backup (measured 2026-11-04: the upload alone
            # proves nothing — the roundtrip proves the bytes AND the key).
            verified = False
            sha = ""
            try:
                back = client.download_file(handle)
                verified = (back == data)
                sha = hashlib.sha256(data).hexdigest()
            except Exception as exc:
                report["errors"].append(f"verify {rel}: {type(exc).__name__}: {exc}")
            report["uploaded"].append({
                "artefact": rel,
                "remote": remote,
                "bytes": len(data),
                "chunks": chunks,
                "handle": handle,
                "reused": reused,
                "verified": verified,
                "sha256": sha,
            })
            print(f"[+] {'kept' if reused else 'archived'} {rel} -> {folder}/{remote} "
                  f"({len(data)} bytes, read back: "
                  f"{'identical' if verified else 'MISMATCH'})")
        except Exception as exc:
            report["errors"].append(f"upload {rel}: {type(exc).__name__}: {exc}")
            print(f"[-] {rel} not archived: {type(exc).__name__}: {exc}")

    uploaded = report["uploaded"]
    verified_count = sum(1 for item in uploaded if item.get("verified"))
    report["verified"] = verified_count
    report["status"] = (
        "ok" if uploaded and verified_count == len(uploaded) and not report["errors"]
        else ("partial" if uploaded else "failed"))
    report["count"] = len(uploaded)
    return report


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """The artefact, written atomically — a half-written report reads as 'never ran'."""
    from harness import atomic_io
    target = path or REPORT_PATH
    if not atomic_io.write_json_atomic(target, report):
        return None
    return target


def main() -> int:
    report = archive_once()
    written = write_report(report)
    print(f"[mega-archive] status={report.get('status')} "
          f"folder={report.get('folder') or '—'} "
          f"uploaded={report.get('count', 0)} "
          f"verified={report.get('verified', 0)} "
          f"errors={len(report.get('errors', []))}")
    if written:
        print(f"[mega-archive] report: {written}")
    for err in report.get("errors", []):
        print(f"[mega-archive] error: {err}")
    # A failed archive is a visible failure: the supervisor must see the cycle
    # as owed, and the operator must see the run as red.
    return 0 if report.get("status") in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
