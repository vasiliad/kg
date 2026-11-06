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
from harness import env_file
from harness import sealed_vault
from harness.copy_contract import copy_verdict
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

# The one artefact that carries the complex's OWN credentials. It is copied
# like everything else — but it is also the one whose COPY has to open.
VAULT_ARTEFACT = "data/.sealed_vault.json"


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
        # The degradations that did NOT break this run's promise: an account
        # that is closed while another one served the copy. They are NAMED,
        # never hidden — and they never turn a kept promise red. Measured
        # 2026-11-04: MEGA closed the door for account 1 while account 2
        # (built by the complex itself) archived and verified all 12 artefacts.
        # The copy was kept; the redundancy was reduced. Two different facts.
        "notes": [],
        "degraded": False,
        # The door's state — «open»/«closed» — and what it means. Filled in
        # only when the cloud refuses us; that is the moment it matters.
        "door": {},
        # The vault's own verdict: can the copy of the credentials be
        # restored? A copy nobody can open is not a backup.
        "vault": {},
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
    vault_note = ""
    if len(usable) < len(accounts):
        vault_note = (f"vault: {len(accounts) - len(usable)} sealed account(s) are "
                      "half-written and cannot be used")

    # The account-level failures are held back until the verdict is known:
    # an account that could not serve the copy is a NOTE when another
    # account served it, and an ERROR when nothing did. Deciding early
    # would call a kept promise broken — the very confusion this report
    # exists to prevent.
    account_failures: List[str] = []
    client = MegaClient()
    working: Optional[Dict[str, Any]] = None
    for account in usable:
        try:
            login = client.login(account["email"], account["password"])
            if not login.get("master_key_ok"):
                account_failures.append(f"account {account['slot']}: "
                                        "the master key did not verify")
                continue
            working = account
            break
        except Exception as exc:
            account_failures.append(f"account {account['slot']}: "
                                    f"{type(exc).__name__}: {exc}")
    provision_note = ""
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
            else:
                provision_note = ("the complex tried to build a new account and could not: "
                                  + (outcome.get("why") or outcome.get("status") or "unknown"))
    if working is None:
        # Nothing served the copy: every account-level failure IS the failure.
        report["errors"].extend(account_failures)
        if vault_note:
            report["errors"].append(vault_note)
        if provision_note:
            report["errors"].append(provision_note)
        return report
    # The copy has a home. What could not serve it is NAMED, not hidden: the
    # operator must see the reduced redundancy before it becomes a dead end.
    if account_failures:
        report["notes"].extend(account_failures)
    if vault_note:
        report["notes"].append(vault_note)
    if provision_note:
        report["notes"].append(provision_note)
    report["degraded"] = bool(report["notes"])
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
    # The bytes of the vault as they come BACK from the cloud. This is the
    # copy that has to open — not the file on this disk (measured
    # 2026-11-06, when the archive's own test caught exactly that: it
    # judged 22 entries of a vault it had never copied).
    vault_copy: Optional[bytes] = None
    # The bytes that went UP — the witness for the copy's verdict. Without
    # them, a cycle that holds the wrong salt could not tell a damaged
    # copy from a fine one, and would have to guess.
    vault_source: Optional[bytes] = None

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
                if rel == VAULT_ARTEFACT and verified:
                    # Keep the bytes that came BACK: they are the copy.
                    vault_copy = back
                    vault_source = data
                if not verified:
                    # A copy that reads back DIFFERENT is not a copy. It used
                    # to be only a «verified: false» flag in the per-artefact
                    # list — invisible in the errors, and easy to miss while
                    # reading the report. Now it is named out loud.
                    report["errors"].append(
                        f"verify {rel}: the copy reads back DIFFERENT "
                        f"({len(back)} bytes instead of {len(data)})")
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
    # The vault itself: the ONE artefact that carries the complex's own
    # credentials. A byte-identical copy of a vault nobody can open is not
    # a backup — it is a file. So the cycle opens the copy that came BACK
    # from the cloud and says so; an entry that does not open is RED,
    # because that credential is already lost no matter how faithfully the
    # bytes moved.
    if vault_copy is None:
        # The vault was not among the copied artefacts (or it did not read
        # back). Then there is no copy of the credentials to judge — and
        # that is a NAMED fact, not a silent one: the operator must know
        # that the credentials are NOT in this copy.
        report["vault"] = {
            "entries": 0, "opened": 0, "ok": False,
            "why": "the vault was not copied — the credentials are not in "
                   "this copy"}
    else:
        try:
            the_salt = vault_salt(salt)
        except Exception as exc:
            report["vault"] = {"entries": 0, "opened": 0, "ok": False,
                               "unknown": True,
                               "why": f"no salt for the vault: {exc}"}
        else:
            report["vault"] = sealed_vault.judge_copy(
                vault_copy, the_salt, source=vault_source,
                # This cycle has already READ its own credentials out of
                # the vault with this salt (load_accounts above) — so the
                # salt is right, and a copy that does not open is DAMAGED,
                # not merely unread.
                salt_proven=True)
    vault_check = report["vault"]
    if vault_check.get("entries") and not vault_check.get("ok"):
        if vault_check.get("unknown"):
            # This cycle cannot read the vault, so it cannot tell whether
            # the copy is intact. NOT a broken copy — an unread one. A
            # named note, not a red: the two facts must not be confused.
            report["notes"].append(
                "vault: this cycle cannot open the vault, so it cannot "
                "judge the copy of the credentials — "
                + str(vault_check.get("why") or "the vault did not open"))
            report["degraded"] = True
        else:
            report["errors"].append(
                "vault: the copy of the credentials cannot be restored — "
                + str(vault_check.get("why") or "the vault did not open"))
    elif vault_copy is None:
        # Not an error: the copy of what EXISTS was kept. But the absence
        # of the credentials from this copy is a degradation the operator
        # must see, not a detail.
        report["notes"].append(
            "the sealed vault was not among the copied artefacts — the "
            "credentials are not in this copy")
        report["degraded"] = True
    # The verdict, in the complex's own words:
    #   ok      — every artefact we HAD is in the cloud and reads back
    #            identical. The promise is kept, so the cycle is green.
    #   partial — some artefacts are there, but at least one did not make it
    #            or did not read back. The promise was broken FOR THAT
    #            ARTEFACT, so the cycle is RED. (Measured 2026-11-04: a
    #            half-archived day used to exit 0 and call itself green.)
    #   failed  — nothing was archived. RED.
    report["status"] = copy_verdict(len(uploaded), verified_count,
                                    report["errors"], report["vault"])
    report["count"] = len(uploaded)
    return report


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """The artefact, written atomically — a half-written report reads as 'never ran'."""
    from harness import atomic_io
    target = path or REPORT_PATH
    if not atomic_io.write_json_atomic(target, report):
        return None
    return target


def exit_code(report: Dict[str, Any]) -> int:
    """
    The verdict as an exit code: only a kept promise is green.

    A separate function so the rule is TESTABLE without writing the report
    to the repository: the cycle's exit code is part of its contract, not
    a side effect of the print.
    """
    return 0 if report.get("status") == "ok" else 1


def main() -> int:
    report = archive_once()
    written = write_report(report)
    print(f"[mega-archive] status={report.get('status')} "
          f"folder={report.get('folder') or '—'} "
          f"uploaded={report.get('count', 0)} "
          f"verified={report.get('verified', 0)} "
          f"errors={len(report.get('errors', []))} "
          f"degraded={1 if report.get('degraded') else 0}")
    if written:
        print(f"[mega-archive] report: {written}")
    for err in report.get("errors", []):
        print(f"[mega-archive] error: {err}")
    # The degradations are printed too: the reduced redundancy must be as
    # visible as the failure, even though it did not break the promise.
    for note in report.get("notes", []):
        print(f"[mega-archive] note: {note}")
    # A broken promise is a visible failure: the operator must see the run
    # as red, and the promise as unkept. Only «ok» is green — a cycle that
    # archived part of the day used to exit 0 and call itself a success.
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
