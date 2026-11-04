#!/usr/bin/env python3
"""
Minimal MEGA protocol client — account registration, verification and login.

The protocol is MEGA's own documented client protocol (the same one the
community CLI megatools implements; the reference sequence is in its
lib/mega.c: mega_session_register / mega_session_register_verify):

  register_request:  a=up (k,ts) -> anon handle; a=us -> tsid; a=ug;
                     a=up name; a=uc (c,n,m) -> MEGA emails the signup link
  register_verify:   a=us (anon) -> a=ud c=<signup_key> -> decrypt master key
                     and challenge, check them -> a=up c uh -> a=us user uh
                     -> a=up pubk privk (RSA-2048, MPI-packed) => account live
  login:             a=us0 (salt) -> pbkdf2-sha512 100k -> a=us user uh

Crypto notes (they matter): the AES here is ECB per 16-byte block (the C
code calls AES_encrypt block by block — identical to MODE_ECB), the
password key is the 65536-round AES chain, the username hash is 16384
rounds, and RSA components travel as MPI (2-byte bit length + big-endian).

The recovery key the operator must never lose IS the master key we
generate here (base64url of the 16 bytes) — register_request returns it.
"""

import base64
import hashlib
import json
import os
import secrets
import string
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA

MEGA_API = "https://g.api.mega.co.nz/cs"
UA = {"User-Agent": "kg-autonomous-complex/1.0 (documented MEGA client protocol)"}


# --- MEGA crypto primitives (ported 1:1 from the public protocol) ----------

def str_to_a32(b: bytes) -> Tuple[int, ...]:
    if isinstance(b, str):
        b = b.encode("latin-1")
    if len(b) % 4:
        b += b"\0" * (4 - len(b) % 4)
    return struct.unpack(">%dI" % (len(b) // 4), b)


def a32_to_str(a) -> bytes:
    return struct.pack(">%dI" % len(a), *a)


def aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    return AES.new(key, AES.MODE_ECB).encrypt(data)


def aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    return AES.new(key, AES.MODE_ECB).decrypt(data)


def b64url_encode(data: bytes) -> str:
    return base64.b64encode(data).decode().rstrip("=").replace("+", "-").replace("/", "_")


def b64url_decode(s: str) -> bytes:
    s += "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(s.replace("-", "+").replace("_", "/"))


def prepare_key(password: str) -> bytes:
    """The 65536-round AES password chain -> 16-byte password key."""
    pkey = bytearray([0x93, 0xC4, 0x67, 0xE3, 0x7D, 0xB0, 0xC7, 0xA4,
                      0xD1, 0xBE, 0x3F, 0x81, 0x01, 0x52, 0xCB, 0x56])
    pw = password.encode("latin-1")
    for _ in range(0x10000):
        for i in range(0, len(pw), 16):
            key = (pw[i:i + 16] + b"\0" * 16)[:16]
            pkey = bytearray(aes_ecb_encrypt(bytes(pkey), key))
    return bytes(pkey)


def stringhash(text: str, key: bytes) -> str:
    """The 16384-round username hash; result is 8 bytes, base64url."""
    s32 = str_to_a32(text)
    h = bytearray(16)
    for i, v in enumerate(s32):
        h[(i * 4) % 16] ^= (v >> 24) & 0xFF
        h[(i * 4 + 1) % 16] ^= (v >> 16) & 0xFF
        h[(i * 4 + 2) % 16] ^= (v >> 8) & 0xFF
        h[(i * 4 + 3) % 16] ^= v & 0xFF
    # The C code XORs byte-wise over the raw string; the a32 route above is
    # equivalent for latin-1-safe strings. Keep the reference form simple:
    hb = bytearray(16)
    raw = text.encode("latin-1")
    for i, ch in enumerate(raw):
        hb[i % 16] ^= ch
    for _ in range(16384):
        hb = bytearray(aes_ecb_encrypt(bytes(hb), key))
    return b64url_encode(bytes(hb[0:4] + hb[8:12]))


def mpi(n: int) -> bytes:
    size = (n.bit_length() + 7) // 8
    return struct.pack(">H", n.bit_length()) + n.to_bytes(size, "big")


def encrypt_key_a32(a, key: bytes) -> bytes:
    """Encrypt a 4-int (16-byte) key block-wise with the master key (ECB)."""
    data = a32_to_str(a)
    return b"".join(aes_ecb_encrypt(data[i:i + 16], key) for i in range(0, len(data), 16))


# --- the API ---------------------------------------------------------------

class MegaClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.seq = 0
        self.sid: Optional[str] = None

    def api(self, payload: Dict[str, Any]) -> Any:
        self.seq += 1
        params = {"id": self.seq}
        if self.sid:
            params["sid"] = self.sid
        r = requests.post(MEGA_API, params=params, data=json.dumps([payload]),
                          headers=UA, timeout=self.timeout)
        resp = r.json()
        if isinstance(resp, list) and resp and isinstance(resp[0], int):
            code = resp[0]
            if code == 0:
                return 0
            raise RuntimeError(f"MEGA API error {code} for {payload.get('a')}")
        return resp[0] if isinstance(resp, list) else resp

    # --- registration ------------------------------------------------------

    def register_request(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """Part 1: ask MEGA to email the signup link. Returns the reg state."""
        email = email.lower()
        master_key = secrets.token_bytes(16)          # this IS the recovery key
        pw_key = prepare_key(password)
        ssc = secrets.token_bytes(16)
        ts_data = ssc + aes_ecb_encrypt(ssc, master_key)

        handle = self.api({"a": "up",
                           "k": b64url_encode(aes_ecb_encrypt(master_key, pw_key)),
                           "ts": b64url_encode(ts_data)})
        if not isinstance(handle, str):
            raise RuntimeError(f"a=up returned {handle!r}")

        login = self.api({"a": "us", "user": handle})
        self.sid = login.get("tsid")
        self.api({"a": "ug"})
        self.api({"a": "up", "name": name})

        c_data = master_key + secrets.token_bytes(16)  # master key + challenge
        self.api({"a": "uc",
                  "c": b64url_encode(aes_ecb_encrypt(c_data, pw_key)),
                  "n": b64url_encode(name.encode()),
                  "m": b64url_encode(email.encode())})
        return {
            "email": email,
            "user_handle": handle,
            "password_key": pw_key,
            "challenge": c_data[16:],
            "master_key": master_key,
            "recovery_key": b64url_encode(master_key),
        }

    def register_verify(self, state: Dict[str, Any], signup_key: str) -> Dict[str, Any]:
        """Part 2: consume the emailed link's key and finish the account."""
        pw_key = state["password_key"]
        self.sid = None
        login = self.api({"a": "us", "user": state["user_handle"]})
        self.sid = login.get("tsid")

        arr = self.api({"a": "ud", "c": signup_key})
        if not (isinstance(arr, list) and len(arr) == 5):
            raise RuntimeError(f"a=ud returned {arr!r}")
        email = b64url_decode(arr[0]).decode()
        master_key = aes_ecb_decrypt(b64url_decode(arr[3]), pw_key)
        challenge = aes_ecb_decrypt(b64url_decode(arr[4]), pw_key)
        if challenge != state["challenge"]:
            raise RuntimeError("challenge mismatch — wrong link or wrong state")
        if master_key != state["master_key"]:
            raise RuntimeError("master key mismatch — wrong link or wrong state")

        uh = stringhash(email.lower(), pw_key)
        self.api({"a": "up", "c": signup_key, "uh": uh})
        login = self.api({"a": "us", "user": email.lower(), "uh": uh})
        self.sid = login.get("tsid")

        key = RSA.generate(2048)
        pubk = b64url_encode(mpi(key.n) + mpi(key.e))
        privk_plain = mpi(key.p) + mpi(key.q) + mpi(key.d) + mpi(key.u)
        if len(privk_plain) % 16:
            privk_plain += b"\0" * (16 - len(privk_plain) % 16)
        privk = b64url_encode(aes_ecb_encrypt(privk_plain, master_key))
        self.api({"a": "up", "pubk": pubk, "privk": privk})
        return {"email": email, "sid": self.sid, "recovery_key": state["recovery_key"]}

    # --- login of an existing account ---------------------------------------

    def login(self, email: str, password: str) -> Dict[str, Any]:
        email = email.lower()
        salt_resp = self.api({"a": "us0", "user": email})
        salt = salt_resp.get("s")
        if salt:
            derived = hashlib.pbkdf2_hmac("sha512", password.encode(),
                                           b64url_decode(salt), 100000, dklen=32)
            pw_key = derived[:16]
            uh = b64url_encode(derived[-16:])
        else:
            pw_key = prepare_key(password)
            uh = stringhash(email, pw_key)
        resp = self.api({"a": "us", "user": email, "uh": uh})
        if isinstance(resp, int):
            raise RuntimeError(f"login failed: MEGA error {resp}")
        self.sid = resp.get("tsid") or resp.get("csid")
        k = resp.get("k")
        master = aes_ecb_decrypt(b64url_decode(k), pw_key) if k else None
        return {"sid": self.sid, "master_key_ok": master is not None}


def signup_key_from_link(link: str) -> str:
    """«https://mega.nz/#confirm<KEY>» -> <KEY>."""
    if "#confirm" in link:
        return link.split("#confirm", 1)[1].strip()
    if "#" in link:
        return link.split("#", 1)[1].strip()
    return link.strip()


def random_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))
