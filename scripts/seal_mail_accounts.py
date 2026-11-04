#!/usr/bin/env python3
"""
Seals the complex's mailboxes and its cloud disk into the sealed vault.

The repository is PUBLIC, so a plaintext password may never enter it. The
vault is the project's answer: data/.sealed_vault.json carries ciphertext
(AES-256-GCM under a salt that lives only in .env / GitHub Actions secrets),
and the cloud opens it with the same salt the local run uses.

The salt is the whole security of this file. The script therefore PROVES it
has the right salt before writing anything: it opens one existing entry
first. A wrong salt raises a GCM tag failure — not garbage — and the script
refuses to seal rather than create entries the cloud cannot read.

WHAT IS SEALED
  proton:address / :username / :password / :plan   the Proton mailbox
  tuta:address / :password / :recovery_code         the Tuta mailbox
  mailtm:address / :password / :token               the readable mailbox
  drive:address / :password / :tfa                 the Proton Drive disk

USAGE
  python scripts/seal_mail_accounts.py               # seal from .env
  python scripts/seal_mail_accounts.py --check       # only verify the salt
  python scripts/seal_mail_accounts.py --force       # seal even if the vault
                                                     # cannot be opened
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import sealed_vault  # noqa: E402
from harness import env_file  # noqa: E402
from harness import mail_accounts  # noqa: E402

# The entries that must already exist for the salt to be proven right.
PROOF_ENTRIES = ("mega:email", "mega:password", "mega:mailtm_token")


def vault_salt() -> str:
    """The salt that opens the vault: VAULT_SALT first, REDACTION_SALT after."""
    env_file.load_dotenv()
    salt = (os.getenv("VAULT_SALT") or os.getenv("REDACTION_SALT") or "").strip()
    if not salt:
        raise RuntimeError("neither VAULT_SALT nor REDACTION_SALT is set — "
                           "the vault cannot be opened or sealed. Put "
                           "VAULT_SALT=... in .env (gitignored) or export it; "
                           "the cloud reads the GitHub secret of the same name")
    return salt


def prove_salt(salt: str) -> bool:
    """True when the salt opens an entry that is already in the vault."""
    for name in PROOF_ENTRIES:
        try:
            value = sealed_vault.get(name, salt=salt)
        except Exception:
            return False
        if value:
            return True
    return False


def collect() -> dict:
    """The plaintexts to seal, taken from .env / the environment."""
    boxes = {b["provider"]: b for b in mail_accounts.mailboxes()}
    drive = mail_accounts.drive()
    return {
        "proton:address": boxes["proton"].get("address", ""),
        "proton:username": boxes["proton"].get("username", ""),
        "proton:password": boxes["proton"].get("password", ""),
        "proton:plan": boxes["proton"].get("plan", ""),
        "tuta:address": boxes["tuta"].get("address", ""),
        "tuta:password": boxes["tuta"].get("password", ""),
        "tuta:recovery_code": boxes["tuta"].get("recovery_code", ""),
        "mailtm:address": boxes["mailtm"].get("address", ""),
        "mailtm:password": boxes["mailtm"].get("password", ""),
        "mailtm:token": boxes["mailtm"].get("token", ""),
        "drive:address": drive.get("address", ""),
        "drive:password": drive.get("password", ""),
        "drive:tfa": drive.get("tfa_secret", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="seal the mailboxes and the cloud disk")
    ap.add_argument("--check", action="store_true", help="only verify the salt")
    ap.add_argument("--force", action="store_true",
                    help="seal even when the existing vault cannot be opened")
    args = ap.parse_args()

    try:
        salt = vault_salt()
    except RuntimeError as exc:
        print(str(exc))
        return 2
    proven = prove_salt(salt)
    print(f"salt present: yes | opens the existing vault: {'yes' if proven else 'NO'}")
    if not proven and not args.force:
        print("refusing to seal: the salt does not open the existing vault entries, "
              "so the cloud would not be able to read what is written. "
              "Fix VAULT_SALT (it must equal the GitHub secret of the same name) "
              "or pass --force deliberately.")
        return 2
    if args.check:
        return 0 if proven else 1

    values = collect()
    missing = [k for k, v in values.items() if not v and k != "drive:tfa"]
    if missing:
        print("missing values, refusing to seal a half-configured vault: "
              + ", ".join(missing))
        return 3

    changed = 0
    for name, value in values.items():
        if not value:
            continue
        # Idempotence: re-sealing an unchanged value would rewrite the file
        # with a fresh nonce and churn the repository for nothing.
        try:
            current = sealed_vault.get(name, salt=salt)
        except Exception:
            current = None
        if current == value:
            print(f"  unchanged {name}")
            continue
        sealed_vault.put(name, value, salt=salt)
        changed += 1
        print(f"  sealed {name} ({len(value)} chars)")

    # Prove the round trip: read one entry back.
    check = sealed_vault.get("proton:address", salt=salt)
    print("round trip:", "ok" if check == values["proton:address"] else "FAILED")
    return 0 if check == values["proton:address"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
