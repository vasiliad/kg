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
