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
import binascii
import hashlib
import json
import os
import re
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


class MegaCryptoError(RuntimeError):
    """
    A node could not be READ.

    «I cannot read it» and «it is not there» are different facts. The
    client used to speak them with the same word (None), and the archive
    paid for it: a node whose name could not be decrypted looked exactly
    like a node with a different name, so the complex created a SECOND
    copy of the day's folder instead of reusing the first one
    (measured 2026-11-04). Only one of those two answers may be silent.
    """


# --- MEGA crypto primitives (ported 1:1 from the public protocol) ----------

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
    """
    The 16384-round username hash; result is 8 bytes, base64url.

    ONE implementation, and it is the live one: the bytes of the string are
    folded into a 16-byte block (byte i -> block[i % 16]) and the block is
    AES-encrypted 16384 times with the password key. The protocol's a32
    word-wise spelling folded into the SAME positions and then threw the
    result away — the code carried two routes, one of them dead, and a
    comment claiming they were «equivalent» that pointed at the dead one.
    A reader could not tell which route the server had actually accepted,
    and the known-answer vectors (tests) are what pinned it: the hash below
    is unchanged, byte for byte, since the account was registered with it.
    """
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


def mpi_to_int(data: bytes) -> int:
    """The inverse of mpi(): two header bytes (bit length) + big-endian body."""
    return int.from_bytes(data[2:], "big")


# --- the API ---------------------------------------------------------------

class MegaClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.seq = 0
        self.sid: Optional[str] = None
        # Set by login(): the account master key — needed to pack node keys
        # for uploads. Never printed, never returned in a result dict.
        self.master_key: Optional[bytes] = None
        self._root_handle: Optional[str] = None

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
        k = resp.get("k")
        master = aes_ecb_decrypt(b64url_decode(k), pw_key) if k else None
        self.master_key = master
        # API v2 (measured 2026-11-04): the login answer carries `csid` — an
        # RSA-ENCRYPTED session id, NOT the session id itself. Passing it
        # straight back earns MEGA error -15 (ESID, invalid session). The real
        # id is opened with the account's own private key, which the server
        # hands over encrypted under the master key (`privk`) — the very key
        # WE generated at registration and sealed into the vault.
        self.sid = resp.get("tsid")
        if not self.sid and resp.get("csid") and resp.get("privk"):
            self.sid = self._open_session_id(resp)
        if not self.sid:
            raise RuntimeError(f"login returned no usable session: {sorted(resp)}")
        return {"sid": self.sid, "master_key_ok": master is not None}

    def _open_session_id(self, resp: Dict[str, Any]) -> str:
        """Decrypt the `csid` blob with the account's RSA key -> the session id."""
        if not self.master_key:
            raise RuntimeError("the master key is needed to open the session")
        raw = aes_ecb_decrypt(b64url_decode(resp["privk"]), self.master_key)
        parts: List[int] = []
        for _ in range(4):
            if len(raw) < 4:
                raise RuntimeError("the account's private key block is truncated")
            bitlen = raw[0] * 256 + raw[1]
            bytelen = (bitlen + 7) // 8 + 2
            parts.append(mpi_to_int(raw[:bytelen]))
            raw = raw[bytelen:]
        p, q, d, _u = parts
        n = p * q
        e = pow(d, -1, (p - 1) * (q - 1))
        key = RSA.construct((n, e, d, p, q))
        encrypted = mpi_to_int(b64url_decode(resp["csid"]))
        plain_int = key._decrypt(encrypted)
        hexed = "%x" % plain_int
        if len(hexed) % 2:
            hexed = "0" + hexed
        sid_bytes = binascii.unhexlify(hexed)
        return b64url_encode(sid_bytes[:43])

    # --- filesystem: list, root, mkdir, upload ------------------------------
    # Ported from the same public protocol's filesystem path (megatools'
    # mega_session_refresh / mega_session_mkdir / mega_session_put in lib/mega.c).

    def list_files(self) -> List[Dict[str, Any]]:
        """a=f c=1 — the whole node list. The root node carries t=2."""
        resp = self.api({"a": "f", "c": 1})
        nodes = resp.get("f") if isinstance(resp, dict) else None
        return nodes if isinstance(nodes, list) else []

    def root_handle(self) -> str:
        """The cloud root's handle (the node with t=2)."""
        if self._root_handle:
            return self._root_handle
        for node in self.list_files():
            if isinstance(node, dict) and node.get("t") == 2 and node.get("h"):
                self._root_handle = node["h"]
                return self._root_handle
        raise RuntimeError("no root node (t=2) in the file list — not logged in?")

    def decrypt_attrs(self, node: Dict[str, Any]) -> Optional[str]:
        """
        Decrypt a node's «MEGA{...}» attributes with its own node key.

        None means «this node carries no attributes at all» — there is
        nothing to read. Every FAILURE to read raises MegaCryptoError: it
        used to answer None as well, and that was the root cause of the
        duplicate folder (see MegaCryptoError). The caller must be able to
        tell «there is no such folder» from «I could not look».
        """
        k = node.get("k")
        a = node.get("a")
        if not (k and a):
            return None
        if not self.master_key:
            raise MegaCryptoError("no master key — the node names cannot be read")
        try:
            # API v2 (measured 2026-11-04): the key arrives scoped to its
            # owner — «<user-handle>:<key>». The handle is not part of the
            # key; without stripping it the base64 decode fails with
            # «33 data characters» and every name stays unreadable.
            if ":" in k:
                k = k.rsplit(":", 1)[-1]
            key_block = aes_ecb_decrypt(b64url_decode(k), self.master_key)
            if node.get("t") == 0:  # file: 32-byte packed key
                aes_key, _, _ = unpack_node_key(key_block)
            else:                   # folder: the key is the 16 bytes as-is
                aes_key = key_block[:16]
            cipher = AES.new(aes_key, AES.MODE_CBC, iv=b"\0" * 16)
            plain = cipher.decrypt(b64url_decode(a))
            text = plain.decode("utf-8", "replace").rstrip("\x00")
            if not text.startswith("MEGA"):
                raise MegaCryptoError("the decrypted attributes are not a MEGA block")
            payload = json.loads(text[4:])
        except MegaCryptoError:
            raise
        except Exception as exc:
            raise MegaCryptoError(f"could not decrypt a node's attributes: {exc}") from exc
        name = payload.get("n")
        if not name:
            raise MegaCryptoError("the node carries attributes without a name")
        return name

    def find_child(self, parent_handle: str, name: str) -> Optional[str]:
        """
        The handle of the child called `name`, or None. When several children
        bear the name (measured 2026-11-04: a run that could not decrypt the
        names once created a second copy), the NEWEST wins — the older one is
        a leftover, and the archive must keep using one branch.

        A child whose name cannot be read is NOT skipped. Skipping it means
        answering «there is no child called <name>» while an unreadable
        child may be hiding under exactly that name — and the caller
        (mkdir) would then create a duplicate. So the answer becomes an
        error: the complex refuses to write into a tree it cannot read.
        """
        matches: List[Dict[str, Any]] = []
        unreadable: List[str] = []
        for node in self.list_files():
            if not (isinstance(node, dict) and node.get("p") == parent_handle):
                continue
            try:
                child_name = self.decrypt_attrs(node)
            except MegaCryptoError as exc:
                unreadable.append(f"{node.get('h')}: {exc}")
                continue
            if child_name is None:
                unreadable.append(f"{node.get('h')}: the node carries no attributes")
                continue
            if child_name == name and node.get("h"):
                matches.append(node)
        if unreadable:
            raise MegaCryptoError(
                f"{len(unreadable)} child(ren) of {parent_handle} could not be read — "
                f"refusing to guess whether «{name}» already exists: "
                + "; ".join(unreadable[:3]))
        if not matches:
            return None
        matches.sort(key=lambda n: int(n.get("ts") or 0))
        return matches[-1].get("h")

    def mkdir(self, parent_handle: str, name: str) -> str:
        """Create a folder; returns its new handle. Idempotent via find_child."""
        existing = self.find_child(parent_handle, name)
        if existing:
            return existing
        folder_key = secrets.token_bytes(16)
        attrs_enc = b64url_encode(aes128_cbc_encrypt_str(encode_node_attrs(name), folder_key))
        key_enc = b64url_encode(aes_ecb_encrypt(folder_key, self.master_key))
        resp = self.api({"a": "p", "t": parent_handle,
                         "n": [{"h": "xxxxxxxx", "t": 1, "k": key_enc, "a": attrs_enc}]})
        arr = resp.get("f") if isinstance(resp, dict) else None
        if not (isinstance(arr, list) and arr and isinstance(arr[0], dict) and arr[0].get("h")):
            raise RuntimeError(f"mkdir returned {resp!r}")
        return arr[0]["h"]

    def upload_file(self, parent_handle: str, name: str, data: bytes) -> Dict[str, Any]:
        """
        Upload one file into a folder. The wire format (measured against the
        public protocol): AES-128-CTR per chunk with iv=nonce‖BE64(offset/16),
        a per-chunk CBC-MAC over the PLAINTEXT folded into meta_mac, chunked
        POSTs to the a=u URL («<p>/<offset>?c=<xor-checksum>»), then one a=p
        with the 32-byte packed node key ECB-encrypted under the master key.
        """
        if not self.master_key:
            raise RuntimeError("upload needs login() first (master key)")
        aes_key = secrets.token_bytes(16)
        nonce = secrets.token_bytes(8)

        # ask for an upload URL
        resp = self.api({"a": "u", "e": 0, "ms": 0, "r": 0,
                         "s": len(data), "ssl": 0, "v": 2})
        p_url = resp.get("p") if isinstance(resp, dict) else None
        if not p_url:
            raise RuntimeError(f"a=u returned {resp!r}")

        macs: List[bytes] = []
        upload_handle: Optional[str] = None
        for offset, length in chunk_plan(len(data)):
            plain = data[offset:offset + length]
            macs.append(chunk_mac(nonce, aes_key, plain))
            iv = nonce + struct.pack(">Q", offset // 16)
            cipher = aes128_ctr(plain, aes_key, iv)
            url = f"{p_url}/{offset}?c={upload_checksum(cipher)}"
            r = requests.post(url, data=cipher, timeout=self.timeout,
                              headers={"Content-Type": "application/octet-stream", **UA})
            # The answer is BINARY: the 36-byte upload handle. Decoding it as
            # text (r.text) destroys it — measured 2026-11-04: the handle came
            # back as mojibake, its length stopped being 36, and every upload
            # failed with «no chunk returned the 36-byte upload handle».
            raw = r.content or b""
            text = raw.decode("utf-8", "replace").strip()
            if r.status_code >= 400 or re.match(r"^-\d+$", text):
                raise RuntimeError(f"chunk at {offset} refused: HTTP {r.status_code} {text[:40]}")
            if len(raw) == 36:
                upload_handle = b64url_encode(raw)
        if not upload_handle:
            raise RuntimeError("no chunk returned the 36-byte upload handle")

        meta_mac = meta_mac_fold(macs, aes_key)
        attrs_enc = b64url_encode(aes128_cbc_encrypt_str(encode_node_attrs(name), aes_key))
        node_key_enc = b64url_encode(
            aes_ecb_encrypt(pack_node_key(aes_key, nonce, meta_mac), self.master_key))
        resp = self.api({"a": "p", "t": parent_handle,
                         "n": [{"h": upload_handle, "t": 0,
                                "k": node_key_enc, "a": attrs_enc}]})
        arr = resp.get("f") if isinstance(resp, dict) else None
        node = arr[0] if isinstance(arr, list) and arr and isinstance(arr[0], dict) else {}
        return {"name": name, "size": len(data), "handle": node.get("h"),
                "chunks": len(macs), "parent": parent_handle}

    def find_children(self, parent_handle: str, name: str) -> List[str]:
        """
        Every child called `name` (a day may hold an older and a newer copy).

        The same rule as find_child: a child that cannot be read makes the
        whole answer unreliable, so it is an error — the caller cleans up
        duplicates by name, and a cleanup that cannot see a name must not
        delete the wrong node (nor claim it did).
        """
        found: List[str] = []
        unreadable: List[str] = []
        for node in self.list_files():
            if not (isinstance(node, dict) and node.get("p") == parent_handle):
                continue
            try:
                child_name = self.decrypt_attrs(node)
            except MegaCryptoError as exc:
                unreadable.append(f"{node.get('h')}: {exc}")
                continue
            if child_name is None:
                unreadable.append(f"{node.get('h')}: the node carries no attributes")
                continue
            if node.get("h") and child_name == name:
                found.append(node["h"])
        if unreadable:
            raise MegaCryptoError(
                f"{len(unreadable)} child(ren) of {parent_handle} could not be read — "
                f"refusing to guess which of them is called «{name}»: "
                + "; ".join(unreadable[:3]))
        return found

    def delete_node(self, handle: str) -> bool:
        """Move a node to the trash (the delete request is a=d with n=handle)."""
        self.api({"a": "d", "n": handle})
        return True

    def download_file(self, handle: str) -> bytes:
        """
        Fetch a node's bytes back and decrypt them with its own node key.

        A backup that cannot be read is not a backup (measured 2026-11-04:
        the archive proved itself only after the roundtrip — 12 files in,
        12 files out, byte-identical). The download request in the current
        API is a=g (a=d is the DELETE request — using it here earned the
        error -2 and cost one debugging round).
        """
        resp = self.api({"a": "g", "g": 1, "ssl": 0, "n": handle})
        url = resp.get("g") if isinstance(resp, dict) else None
        if not url:
            raise RuntimeError(f"a=g returned no download URL for {handle}")
        raw = requests.get(url, timeout=self.timeout).content
        node = None
        for candidate in self.list_files():
            if isinstance(candidate, dict) and candidate.get("h") == handle:
                node = candidate
                break
        if not node or not node.get("k"):
            raise RuntimeError(f"the node {handle} is not in the file list")
        k = node["k"]
        if ":" in k:
            k = k.rsplit(":", 1)[-1]
        key_block = aes_ecb_decrypt(b64url_decode(k), self.master_key)
        aes_key, nonce, _meta_xor = unpack_node_key(key_block)
        return aes128_ctr(raw, aes_key, nonce + b"\0" * 8)


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


# --- the upload wire format (ported 1:1 from the public protocol) ----------
# Reference: megatools' lib/mega.c — prepare_transfer / tman_worker_upload_chunk /
# chunk_mac_calculate / meta_mac_calculate / pack_node_key / mega_session_put.

def get_chunk_size(idx: int) -> int:
    """Chunk sizes grow: 128K, 256K, … 1M, then 1M until EOF."""
    return (idx + 1 if idx < 8 else 8) * 128 * 1024


def chunk_plan(size: int) -> List[Tuple[int, int]]:
    """[(offset, length), …] covering the file exactly."""
    plan: List[Tuple[int, int]] = []
    off, idx = 0, 0
    while off < size:
        n = min(get_chunk_size(idx), size - off)
        plan.append((off, n))
        off += n
        idx += 1
    return plan


def aes128_ctr(data: bytes, key: bytes, iv: bytes) -> bytes:
    """AES-128-CTR over the full 16-byte counter (what EVP_aes_128_ctr does)."""
    return AES.new(key, AES.MODE_CTR, nonce=b"",
                   initial_value=int.from_bytes(iv, "big")).encrypt(data)


def chunk_mac(nonce8: bytes, key: bytes, plain: bytes) -> bytes:
    """One chunk's CBC-MAC over the PLAINTEXT; IV = nonce‖nonce (16 bytes)."""
    mac = bytearray(nonce8 + nonce8)
    padded = plain + b"\0" * ((-len(plain)) % 16)
    for i in range(0, len(padded), 16):
        block = bytes(a ^ b for a, b in zip(mac, padded[i:i + 16]))
        mac = bytearray(aes_ecb_encrypt(block, key))
    return bytes(mac)


def meta_mac_fold(macs: List[bytes], key: bytes) -> bytes:
    """meta_mac = fold: XOR each chunk mac in, then one AES block after each."""
    mm = bytearray(16)
    for m in macs:
        for i in range(16):
            mm[i] ^= m[i]
        mm = bytearray(aes_ecb_encrypt(bytes(mm), key))
    return bytes(mm)


def _dw(b: bytes, i: int) -> int:
    return struct.unpack(">I", b[i * 4:i * 4 + 4])[0]


def _pack_dw(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def pack_node_key(aes_key: bytes, nonce8: bytes, meta_mac: bytes) -> bytes:
    """The 32-byte file node key: key^nonce, key^meta_mac parts, then raw parts."""
    return b"".join([
        _pack_dw(_dw(aes_key, 0) ^ _dw(nonce8, 0)),
        _pack_dw(_dw(aes_key, 1) ^ _dw(nonce8, 1)),
        _pack_dw(_dw(aes_key, 2) ^ _dw(meta_mac, 0) ^ _dw(meta_mac, 1)),
        _pack_dw(_dw(aes_key, 3) ^ _dw(meta_mac, 2) ^ _dw(meta_mac, 3)),
        nonce8[0:8],
        _pack_dw(_dw(meta_mac, 0) ^ _dw(meta_mac, 1)),
        _pack_dw(_dw(meta_mac, 2) ^ _dw(meta_mac, 3)),
    ])


def unpack_node_key(node_key: bytes) -> Tuple[bytes, bytes, bytes]:
    """Inverse of pack_node_key: (aes_key, nonce8, meta_mac_xor8)."""
    aes_key = b"".join([
        _pack_dw(_dw(node_key, 0) ^ _dw(node_key, 4)),
        _pack_dw(_dw(node_key, 1) ^ _dw(node_key, 5)),
        _pack_dw(_dw(node_key, 2) ^ _dw(node_key, 6)),
        _pack_dw(_dw(node_key, 3) ^ _dw(node_key, 7)),
    ])
    nonce = node_key[16:24]
    meta_mac_xor = b"".join([_pack_dw(_dw(node_key, 6)), _pack_dw(_dw(node_key, 7))])
    return aes_key, nonce, meta_mac_xor


def encode_node_attrs(name: str) -> str:
    """Attributes are «MEGA» + compact JSON {"n": name}."""
    return "MEGA" + json.dumps({"n": name}, ensure_ascii=False, separators=(",", ":"))


def aes128_cbc_encrypt_str(text: str, key: bytes) -> bytes:
    """CBC (zero IV) over the string zero-padded past a NUL to a 16-byte block."""
    raw = text.encode("utf-8")
    aligned = len(raw) + 1
    if aligned % 16:
        aligned += 16 - aligned % 16
    data = raw + b"\0" * (aligned - len(raw))
    return AES.new(key, AES.MODE_CBC, iv=b"\0" * 16).encrypt(data)


def upload_checksum(cipher: bytes) -> str:
    """The ?c= checksum: 12 bytes, each XOR-rotated into place, base64url."""
    crc = bytearray(12)
    for i, b in enumerate(cipher):
        crc[i % 12] ^= b
    return b64url_encode(bytes(crc))
