#!/usr/bin/env python3
"""
The complex's own cloud space — Proton Drive, kept BY THE COMPLEX.

Measured 2026-11-04: the operator handed over three mailboxes and a Proton
Drive disk. The letter's first version pointed the disk at someone ELSE'S
VPS (lama2) — but the complex has no business depending on a borrowed host:
every process must revive itself after a restart on its OWN infrastructure.
So the remote is built here, from the credentials that live in the vault and
in the cloud's environment:

  - NO config file. The remote is assembled in the environment
    (RCLONE_CONFIG_PROTONDRIVE_*) and handed to rclone that way, so the
    password is never a file, never an argument, and never in the log;
  - NO mount. A mount dies with the runner — it is ephemeral by nature. The
    working scheme is `rclone copy`: one step, one proof;
  - NO app_version. rclone must send its own correct value (measured by the
    operator: a hand-written one is refused by Proton — Code=2064);
  - the round trip is PROVED. Every file is read back and compared byte for
    byte: an upload alone proves nothing (the MEGA lesson).

WHAT IS COPIED
  The complex's own state — the daily record, the reports, the keeper's
  settings, the vault's ciphertext. This is the copy that must survive this
  computer, the operator's laptop and (if it ever happens) the repository.

WHAT IS NEVER DONE HERE
  A password is never printed, never written to a file and never put in an
  argument. The report names the account as a fingerprint, and any text
  rclone returns is masked before it enters the report.
"""

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import env_file
from harness import redaction
from harness import sealed_vault

REPORT_PATH = PROJECT_ROOT / "data" / "proton_drive_report.json"

# The remote's name and where the complex's files live on it. The marker
# file (immortal/disk_readme.md) is the operator's proof that the root share
# exists — it was created once, by a human, in the Proton web client.
REMOTE = "protondrive"
REMOTE_ROOT = f"{REMOTE}:/immortal/"

RCLONE = "rclone"
COPY_TIMEOUT = 600
READ_TIMEOUT = 120

# The complex's state — what must survive this computer. Every path is
# optional: a cycle that runs before the daily digest exists must still copy
# what DOES exist, and say what was missing.
#
# TWO TIERS, because the cloud and the keeper do not hold the same files:
#   STATE_FILES   travel with the repository. If one is missing, that is a
#                 NAMED degradation — «it is not there» must not look like
#                 «it is copied».
#   KEEPER_FILES  live only on a keeper machine (they are in .gitignore by
#                 design — local settings, possibly local paths). They are
#                 copied when this machine has them; the cloud runner never
#                 will, and that is NOT a degradation: the copy is of what
#                 this machine holds.
STATE_FILES: Tuple[str, ...] = (
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
    "data/.bot_sessions.json",
    "data/.credential_heartbeat.json",
    "data/.sealed_vault.json",
)

KEEPER_FILES: Tuple[str, ...] = (
    "data/.bot_settings.json",
)


class DriveError(RuntimeError):
    """The disk could not be reached — and the reason is worth reading."""


def _text(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def vault_salt(explicit: Optional[str] = None) -> str:
    """The salt that opens the vault: VAULT_SALT first, REDACTION_SALT after."""
    env_file.load_dotenv()
    salt = (explicit if explicit is not None else
            os.getenv("VAULT_SALT") or os.getenv("REDACTION_SALT") or "").strip()
    if not salt:
        raise RuntimeError("neither VAULT_SALT nor REDACTION_SALT is set — "
                           "the vault cannot be opened")
    return salt


def drive_credentials(salt: Optional[str] = None) -> Dict[str, str]:
    """
    Where the Drive credentials come from, in order:

      1. the environment — the cloud's own secrets, or `.env` on a local run;
      2. the sealed vault — drive:address / drive:password / drive:tfa.

    Both paths are real: the cloud has the secrets, the local keeper may have
    only the salt. The report says WHICH one was used, so a missing
    credential is never a mystery.
    """
    env_file.load_dotenv()
    address = (os.getenv("PROTON_DRIVE_ADDRESS") or "").strip()
    password = (os.getenv("PROTON_DRIVE_PASSWORD") or "").strip()
    tfa = (os.getenv("PROTON_DRIVE_2FA_SECRET") or "").strip()
    source = "environment"
    if not address or not password:
        source = "vault"
        try:
            the_salt = vault_salt(salt)
        except Exception:
            the_salt = ""
        if the_salt:
            try:
                address = (sealed_vault.get("drive:address", salt=the_salt) or "").strip()
                password = (sealed_vault.get("drive:password", salt=the_salt) or "").strip()
                tfa = (sealed_vault.get("drive:tfa", salt=the_salt) or tfa).strip()
            except Exception:
                # A wrong salt raises here. The caller reports the failure;
                # the environment path may still work.
                pass
    return {"address": address, "password": password, "tfa": tfa, "source": source}


def remote_env(address: str, password: str, tfa: str = "") -> Dict[str, str]:
    """
    The remote, as environment variables.

    NO config file is written (a file on the runner dies with the runner, and
    a config file in the repository would be a secret in the repository), and
    the password is never an argument (arguments show up in the process list
    and in the log). `app_version` is deliberately NOT set: rclone must send
    its own correct value.
    """
    env = dict(os.environ)
    env[f"RCLONE_CONFIG_{REMOTE.upper()}_TYPE"] = "protondrive"
    env[f"RCLONE_CONFIG_{REMOTE.upper()}_USERNAME"] = address
    env[f"RCLONE_CONFIG_{REMOTE.upper()}_PASSWORD"] = obscure_password(password)
    # The TOTP secret is NOT passed to rclone: Proton Drive's rclone backend
    # has no 2FA field. It is kept because it tells the operator whether the
    # account is protected — and because a human needs it if the account ever
    # asks for a second factor.
    # The proxy, when the network needs one. Measured 2026-11-05: rclone
    # 1.74.3 has NO `--proxy-server` flag any more (the operator's note
    # predates that) — but rclone speaks HTTP, so the standard
    # HTTPS_PROXY/HTTP_PROXY environment variables are honoured and are the
    # supported way through.
    proxy = rclone_proxy()
    if proxy:
        env["HTTPS_PROXY"] = proxy
        env["HTTP_PROXY"] = proxy
    return env


# rclone reads a config password in its OBFUSCATED form.
#
# Measured 2026-11-05: handing rclone the plaintext through the environment
# produced «couldn't decrypt password: base64 decode failed ... is it
# obscured?» — the very first live run of this module. And `rclone obscure`
# takes the password as an ARGUMENT, which would put it in the process list
# and in the log — the thing the operator's letter forbids.
#
# So the complex obscures it itself. The algorithm is rclone's own
# (fs/config/obscure, v1.74.3): AES-CTR under a FIXED key, a random 16-byte
# IV in front of the ciphertext, and base64url without padding. The result is
# checked against rclone itself: `rclone reveal` turns this value back into
# the password.
_RCLONE_CRYPT_KEY = bytes((
    0x9c, 0x93, 0x5b, 0x48, 0x73, 0x0a, 0x55, 0x4d,
    0x6b, 0xfd, 0x7c, 0x63, 0xc8, 0x86, 0xa9, 0x2b,
    0xd3, 0x90, 0x19, 0x8e, 0xb8, 0x12, 0x8a, 0xfb,
    0xf4, 0xde, 0x16, 0x2b, 0x8b, 0x95, 0xf6, 0x38,
))


def obscure_password(password: str) -> str:
    """The password in the form rclone expects from an environment variable."""
    from Crypto.Cipher import AES
    iv = os.urandom(16)
    cipher = AES.new(_RCLONE_CRYPT_KEY, AES.MODE_CTR, nonce=b"",
                     initial_value=int.from_bytes(iv, "big"))
    blob = iv + cipher.encrypt((password or "").encode("utf-8"))
    return base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")


def rclone_proxy() -> str:
    """
    The proxy, when the network needs one.

    Measured 2026-11-05: from a Russian network Proton's API is not
    reachable directly — the connection is RESET after ~35 s, while another
    provider answers in half a second. The operator's own xray opens it
    (http://127.0.0.1:10809). The cloud runner is outside that network and
    needs no proxy at all, so this is opt-in: set PROTON_DRIVE_PROXY and the
    standard HTTPS_PROXY/HTTP_PROXY variables carry it to rclone.
    """
    env_file.load_dotenv()
    return (os.getenv("PROTON_DRIVE_PROXY") or "").strip()


def run_rclone(args: List[str], env: Dict[str, str],
               timeout: int = COPY_TIMEOUT) -> Tuple[int, bytes, bytes]:
    """
    Runs rclone and returns (returncode, stdout, stderr) as BYTES.

    Bytes, not text: the round-trip proof compares the file that comes back
    with the file that went out, and a text decoder would quietly rewrite
    what it cannot read.
    """
    cmd = [RCLONE] + list(args)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        raise DriveError(
            "rclone is not installed on this host — the complex cannot reach "
            "its own disk without it (the workflow installs it; a local run "
            "needs it on PATH)")
    except subprocess.TimeoutExpired:
        raise DriveError(f"rclone did not answer in {timeout}s: {' '.join(args[:2])}")
    return proc.returncode, proc.stdout, proc.stderr


def collect_state(root: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """The state files that actually exist right now (both tiers)."""
    base = Path(root) if root else PROJECT_ROOT
    found: List[Tuple[str, Path]] = []
    for rel in tuple(STATE_FILES) + tuple(KEEPER_FILES):
        path = base / rel
        if path.exists() and path.is_file():
            found.append((rel, path))
    return found


def stage(files: List[Tuple[str, Path]], staging: Path) -> None:
    """Lays the state out under its own relative paths, for one clean copy."""
    for rel, path in files:
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def sync_once(root: Optional[Path] = None,
              salt: Optional[str] = None) -> Dict[str, Any]:
    """
    One copy to the complex's own disk, and the proof that it is there.

    The report is the artefact: it says what was copied, what read back
    identical, and — if anything went wrong — what went wrong.
    """
    base = Path(root) if root else PROJECT_ROOT
    now = datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "checked_at": now.isoformat(),
        "status": "failed",
        "remote": REMOTE_ROOT,
        "account": "",
        "source": "",
        "uploaded": [],
        "missing": [],
        "errors": [],
        "notes": [],
        "degraded": False,
        "two_factor": "",
    }

    creds = drive_credentials(salt)
    report["source"] = creds["source"]
    report["two_factor"] = "on (a secret is sealed)" if creds["tfa"] else "off"
    if not creds["address"] or not creds["password"]:
        report["errors"].append(
            "the drive credentials are not available — neither the environment "
            "nor the vault holds them (PROTON_DRIVE_ADDRESS / PROTON_DRIVE_PASSWORD, "
            "or drive:address / drive:password in the vault)")
        return report
    report["account"] = redaction.fingerprint(creds["address"])

    env = remote_env(creds["address"], creds["password"], creds["tfa"])

    files = collect_state(base)
    present = {rel for rel, _ in files}
    report["missing"] = [rel for rel in STATE_FILES if rel not in present]
    if report["missing"]:
        # NAMED, not hidden. A state file that does not exist on this machine
        # is not a broken copy — the copy of what DOES exist is still a copy —
        # but «it is not there» and «I cannot read it» are different facts,
        # and the operator is told which one it is.
        report["notes"].append(
            "these state files are not on this machine, so they were not "
            "copied: " + ", ".join(report["missing"]))
    if not files:
        report["errors"].append("there is no state to copy — nothing was written")
        return report

    with tempfile.TemporaryDirectory(prefix="drive-stage-") as tmp:
        staging = Path(tmp)
        stage(files, staging)
        code, out, err = run_rclone(["copy", str(staging), REMOTE_ROOT], env)
        if code != 0:
            message = redaction.mask_emails(_text(err) or _text(out)).strip()
            report["errors"].append(
                f"copy: rclone answered {code}: {message[:300] or 'no explanation'}")
            return report

        # THE PROOF: read every file back and compare the bytes. An upload
        # that cannot be opened is not a copy.
        for rel, path in files:
            data = path.read_bytes()
            remote = f"{REMOTE_ROOT}{rel}"
            verified = False
            sha = ""
            try:
                code, out, err = run_rclone(["cat", remote], env, timeout=READ_TIMEOUT)
                if code != 0:
                    message = redaction.mask_emails(_text(err) or _text(out)).strip()
                    report["errors"].append(f"read back {rel}: rclone answered {code}: "
                                            f"{message[:200] or 'no explanation'}")
                else:
                    verified = (out == data)
                    sha = hashlib.sha256(data).hexdigest()
                    if not verified:
                        report["errors"].append(
                            f"read back {rel}: the copy reads back DIFFERENT "
                            f"({len(out)} bytes instead of {len(data)})")
            except DriveError as exc:
                report["errors"].append(f"read back {rel}: {exc}")
            report["uploaded"].append({
                "artefact": rel,
                "remote": remote,
                "bytes": len(data),
                "verified": verified,
                "sha256": sha,
            })
            print(f"[+] {'copied' if verified else 'COPIED BUT NOT PROVEN'} {rel} "
                  f"-> {remote} ({len(data)} bytes, read back: "
                  f"{'identical' if verified else 'MISMATCH'})")

    uploaded = report["uploaded"]
    verified_count = sum(1 for item in uploaded if item.get("verified"))
    report["verified"] = verified_count
    report["count"] = len(uploaded)
    report["status"] = (
        "ok" if uploaded and verified_count == len(uploaded) and not report["errors"]
        else ("partial" if uploaded else "failed"))
    # The copy was kept — but something about it is worth naming. Same
    # contract as the MEGA archive: a degradation is visible, not red.
    report["degraded"] = bool(report["notes"])
    return report


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """The artefact, written atomically — a half-written report reads as 'never ran'."""
    from harness import atomic_io
    target = path or REPORT_PATH
    if not atomic_io.write_json_atomic(target, report):
        return None
    return target


def exit_code(report: Dict[str, Any]) -> int:
    """Only a proved copy is green — the same contract as the MEGA archive."""
    return 0 if report.get("status") == "ok" else 1


def main() -> int:
    report = sync_once()
    written = write_report(report)
    print(f"[proton-drive] status={report.get('status')} "
          f"remote={report.get('remote')} "
          f"copied={report.get('count', 0)} "
          f"verified={report.get('verified', 0)} "
          f"errors={len(report.get('errors', []))}")
    if written:
        print(f"[proton-drive] report: {written}")
    for err in report.get("errors", []):
        print(f"[proton-drive] error: {err}")
    for note in report.get("notes", []):
        print(f"[proton-drive] note: {note}")
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
