#!/usr/bin/env python3
"""
The sealed vault — credentials that live in the PUBLIC repository but
cannot be read from it.

The complex registers its own accounts (MEGA today) and must keep the
recovery keys somewhere durable and shared between cloud runs. A public
repo cannot hold plaintext secrets — so the vault seals each entry with
AES-256-GCM under a key derived from the REDACTION_SALT secret (which
exists in the Actions environment and never in the repo). The committed
file data/.sealed_vault.json carries only ciphertext + nonce + tag.

Tampering is detected by the GCM tag; a wrong salt yields a tag failure,
not garbage. seal/open are pure functions of (plaintext, salt) — no state.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from Crypto.Cipher import AES

VAULT_PATH = Path(__file__).resolve().parent.parent / "data" / ".sealed_vault.json"
SALT_ENV = "REDACTION_SALT"


def _vault_key(salt: Optional[str] = None) -> bytes:
    salt = (salt if salt is not None else os.getenv(SALT_ENV, "")).strip()
    if not salt:
        raise RuntimeError(f"{SALT_ENV} is not set — the vault cannot be opened")
    # HKDF-SHA256, domain-separated from the redaction fingerprints.
    return hashlib.pbkdf2_hmac("sha256", salt.encode(), b"kg-sealed-vault-v1", 200_000, dklen=32)


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _b64d(s: str) -> bytes:
    return base64.b64decode(s)


def seal(name: str, plaintext: str, salt: Optional[str] = None) -> Dict[str, str]:
    """One sealed entry: name -> {nonce, ct, tag} (all base64)."""
    key = _vault_key(salt)
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    return {"name": name, "nonce": _b64e(nonce), "ct": _b64e(ct), "tag": _b64e(tag)}


def open_entry(entry: Dict[str, str], salt: Optional[str] = None) -> str:
    """Decrypt one entry; raises on tampering or a wrong salt (GCM tag)."""
    key = _vault_key(salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=_b64d(entry["nonce"]))
    return cipher.decrypt_and_verify(_b64d(entry["ct"]), _b64d(entry["tag"])).decode("utf-8")


def load_vault(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or VAULT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_vault(vault: Dict[str, Any], path: Optional[Path] = None) -> bool:
    path = path or VAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(vault, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return True


def put(name: str, plaintext: str, salt: Optional[str] = None,
        path: Optional[Path] = None) -> Dict[str, Any]:
    """Seal one entry into the vault file (atomic)."""
    vault = load_vault(path)
    entries = vault.get("entries")
    if not isinstance(entries, list):
        entries = []
    entries = [e for e in entries if isinstance(e, dict) and e.get("name") != name]
    entries.append(seal(name, plaintext, salt))
    vault = {"version": 1, "cipher": "AES-256-GCM", "kdf": "PBKDF2-HMAC-SHA256",
             "entries": entries}
    save_vault(vault, path)
    return vault


def get(name: str, salt: Optional[str] = None, path: Optional[Path] = None) -> Optional[str]:
    """Open one entry from the vault file; None when absent."""
    for entry in load_vault(path).get("entries", []):
        if isinstance(entry, dict) and entry.get("name") == name:
            return open_entry(entry, salt)
    return None


def entry_names(path: Optional[Path] = None) -> List[str]:
    """
    The names of the sealed entries — without opening any of them.

    The names are plaintext in the ciphertext file, so the INVENTORY of the
    vault is knowable even when the salt is not at hand. «I cannot open the
    vault» must never become «there is nothing in it».
    """
    return [str(entry.get("name") or "")
            for entry in load_vault(path).get("entries", [])
            if isinstance(entry, dict)]


def open_many(names: List[str], salt: Optional[str] = None,
              path: Optional[Path] = None) -> Dict[str, str]:
    """
    Open several entries with ONE key derivation.

    The KDF is the cost of this vault (200 000 PBKDF2 rounds), and reading
    one mailbox means three entries. Deriving the key once per entry made
    the inventory slower than the thing it measures — so the key is derived
    once here, and the entries are opened with it.
    """
    key = _vault_key(salt)
    wanted = set(names)
    out: Dict[str, str] = {}
    for entry in load_vault(path).get("entries", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if name not in wanted:
            continue
        cipher = AES.new(key, AES.MODE_GCM, nonce=_b64d(entry["nonce"]))
        out[name] = cipher.decrypt_and_verify(
            _b64d(entry["ct"]), _b64d(entry["tag"])).decode("utf-8")
    return out


def self_check_bytes(data: bytes, salt: Optional[str] = None) -> Dict[str, Any]:
    """
    Can the vault in THESE BYTES be opened — entry by entry?

    This is the check that belongs to a COPY: it takes the bytes that came
    back from the cloud, not the file that went there. The difference
    matters — a cycle that checked the file on disk would prove that the
    ORIGINAL opens, and say nothing about the copy (measured 2026-11-06,
    when the archive's own test caught exactly that: it judged 22 entries
    of a vault it had never copied).

    The GCM tag is the proof: a wrong salt or a tampered ciphertext fails
    the tag instead of yielding garbage. The result is a FACT, not a
    feeling — and an empty vault is reported as empty, never as «broken»
    and never as «fine».
    """
    try:
        parsed = json.loads(bytes(data).decode("utf-8"))
    except Exception as exc:
        return {"entries": 0, "opened": 0, "ok": False,
                "why": f"the copied vault is not readable as a vault: {exc}"}
    if not isinstance(parsed, dict):
        return {"entries": 0, "opened": 0, "ok": False,
                "why": "the copied vault is not a vault"}
    entries = [e for e in (parsed.get("entries") or []) if isinstance(e, dict)]
    names = [str(entry.get("name") or "") for entry in entries]
    if not names:
        return {"entries": 0, "opened": 0, "ok": False,
                "why": "the copied vault is empty — there is nothing to open yet"}
    opened = 0
    broken: List[str] = []
    try:
        key = _vault_key(salt)
    except Exception as exc:
        return {"entries": len(names), "opened": 0, "ok": False,
                "why": f"the copied vault cannot be opened at all: {exc}"}
    for entry in entries:
        name = str(entry.get("name") or "")
        try:
            cipher = AES.new(key, AES.MODE_GCM, nonce=_b64d(entry["nonce"]))
            cipher.decrypt_and_verify(_b64d(entry["ct"]), _b64d(entry["tag"]))
            opened += 1
        except Exception:
            broken.append(name)
    if broken:
        return {"entries": len(names), "opened": opened, "ok": False,
                "broken": broken[:5],
                "why": f"{len(broken)} of {len(names)} entries do not open — "
                       "the copy of the credentials cannot restore them"}
    return {"entries": len(names), "opened": opened, "ok": True, "why": ""}


def self_check(salt: Optional[str] = None,
               path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Can this vault be opened AT ALL — entry by entry?

    The same check as self_check_bytes, for a vault that is still on this
    machine. A cycle that COPIES the vault must use the bytes that came
    BACK, not this one — the copy is what has to open.
    """
    try:
        raw = (path or VAULT_PATH).read_bytes()
    except Exception as exc:
        return {"entries": 0, "opened": 0, "ok": False,
                "why": f"the vault cannot be read: {exc}"}
    return self_check_bytes(raw, salt)


def judge_copy(copy_bytes: bytes, salt: Optional[str] = None,
               source: Optional[bytes] = None,
               path: Optional[Path] = None,
               salt_proven: Optional[bool] = None) -> Dict[str, Any]:
    """
    The verdict of a COPY of the vault — measured against the source.

    A copy that does not open has two possible causes, and they are
    DIFFERENT facts:
      - the copy is damaged (the salt is known right, and still the copy
        does not open) — the credential copy is BROKEN, and the promise was
        not kept;
      - this cycle cannot read the vault at all — the copy may be perfectly
        fine, and «I cannot read it» must never become «it is broken».

    Measured 2026-11-05: the two salts do NOT open the same vault. So a
    cycle holding only REDACTION_SALT would see a vault it cannot open and
    could report a broken backup — a false red about a backup that is
    probably fine.

    Two witnesses, in this order:
      1. salt_proven=True — this cycle USED the salt successfully (it read
         its own credentials out of the vault). Then the salt is right, and
         a copy that does not open is DAMAGED.
      2. the SOURCE bytes that went up. If they open with this salt, the
         salt is right ⇒ a copy that does not open is DAMAGED. If neither
         opens, this cycle simply cannot read the vault ⇒ UNKNOWN, a named
         note and never a red.

    The result says which of the two happened, so the report never has to
    guess.
    """
    copy_check = self_check_bytes(copy_bytes, salt)
    if copy_check.get("ok"):
        return copy_check
    if salt_proven:
        return dict(copy_check, why=(
            "the copy does not open, although this cycle opened the vault "
            "with the same salt — the copy is damaged"))
    if source is not None:
        source_check = self_check_bytes(source, salt)
    else:
        source_check = self_check(salt, path=path)
    # The source opens with this salt ⇒ the salt is right ⇒ a copy that
    # does not open is DAMAGED. That breaks the promise.
    if source_check.get("ok"):
        return dict(copy_check, why=(
            "the copy does not open, although the original does with the "
            "same salt — the copy is damaged"))
    # Neither opens: this cycle simply cannot read the vault. UNKNOWN —
    # not broken. Named so the operator can see the difference.
    return dict(copy_check, unknown=True, why=(
        "this cycle cannot open the vault, so it cannot tell whether the "
        "copy is intact — " + str(copy_check.get("why") or "the vault did "
        "not open")))
