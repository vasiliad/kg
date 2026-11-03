#!/usr/bin/env python3
"""
Keeps the operator's address out of the public repository.

The repository is PUBLIC. The recipient of the reports is the operator's
personal address, and it lives in exactly one place — `data/.bot_settings.json`,
which is untracked, plus the `DISPATCH_RECIPIENT_EMAIL` secret. It must never
be hardcoded in the source, and it must never be committed.

Measured on 2026-11-01, it WAS committed in four places:
`.env.example`, `nim_history.log`, `data/bot_selftest_report.json` and a test
fixture. Two of them were permanent (the example file and the fixture) and two
were state files that the workflows commit on every run — the delivery record
writes the address, and the bot's own honest answer about "where shall I send
it" quotes it, so the committed transcript carried it too.

The fix is not to stop writing the record — the record is the evidence that the
report really went out. It is to write the address in a form that cannot be
read back into an address: a short, non-reversible fingerprint. The operator
can still confirm that the report went to THEIR address (hash it and compare),
and the public repository learns nothing.

THE TELEGRAM CHAT IS THE SAME KIND OF ADDRESS.

Measured 2026-11-03: the operator's Telegram chat ids were committed in six
places — `data/.bot_sessions.json` (the authorized chats and
`health.last_chat_id`), `data/bot_transport_report.json`,
`data/telegram_history.txt`, `diagnostics_report.json`, `nim_history.log`
(«чат=<the operator's id>») and a test fixture — and they were also PRINTED into the
Actions log, which is public for a public repository.

So the chat id gets the same treatment: the real id lives in exactly one
place, the `TELEGRAM_OPERATOR_CHAT_ID` secret, and everything the repository
keeps is a fingerprint (`чат#1a2b3c4d`).

WHY THE FINGERPRINT IS SALTED.

A plain SHA-256 of a 9-10 digit chat id is not a secret: the id space is
small enough to enumerate in seconds, so anyone could confirm WHICH chat id
produced a published tag — the fingerprint would be as good as the raw id.
The `REDACTION_SALT` secret (a random value that lives only in the Actions
environment) keys the digest, so the tag cannot be confirmed from the
repository alone. Without the salt the digest still works — it is only a
confirmation oracle, never a way back to the id.
"""

import hashlib
import hmac
import os
import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# A chat id in free text: the word that introduces it («чат», «chat»,
# «chat_id», «last_chat_id») and the digits after it. The id must be at
# least six digits long — a Telegram chat id is much longer, and a short
# number after the word «чат» is far more likely to be a count («чат 5
# сообщений») than an address.
CHAT_RE = re.compile(
    r"(?i)"
    r"(?P<key>\b(?:last[_\s]?)?(?:chat|чат)[_\s]?id\b|\bchat\b|\bчат\b)"
    r"(?P<sep>[\"'`]*\s*[:=]?\s*[\"'`]*)"
    r"(?P<id>-?\d{6,15})"
)

# The JSON keys whose VALUE is a chat id (or a list of them).
CHAT_KEYS = {
    "chat_id",
    "last_chat_id",
    "authorized_chats",
    "target_chat",
    "operator_chat_id",
}

CHAT_TAG_PREFIX = "чат#"


def _digest(text: str) -> str:
    """
    The keyed digest behind every fingerprint.

    With `REDACTION_SALT` in the environment the digest is an HMAC: the
    published tag cannot be confirmed against a guessed id by anyone who
    does not hold the secret. Without the salt it is a plain SHA-256 — still
    a one-way tag, just a confirmable one.
    """
    salt = (os.getenv("REDACTION_SALT") or "").strip()
    raw = text.encode("utf-8")
    if salt:
        return hmac.new(salt.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hashlib.sha256(raw).hexdigest()


def recipient_tag(address: str) -> str:
    """
    A short, non-reversible fingerprint of an address.

    «настроен (sha256:1a2b3c4d)» says "an address was resolved and it is the
    same one as last time" without saying which address it is.
    """
    text = (address or "").strip()
    if not text:
        return "не задан"
    digest = _digest(text.lower())[:8]
    return f"настроен (sha256:{digest})"


def chat_tag(chat_id: Any) -> str:
    """
    A short, non-reversible fingerprint of a Telegram chat id.

    «чат#1a2b3c4d» says "a chat was resolved and it is the same one as last
    time" without saying which chat it is. The operator can still confirm it
    (the secret is theirs) and the public repository learns nothing.
    """
    text = str(chat_id if chat_id is not None else "").strip()
    if not text:
        return "не задан"
    if text.startswith(CHAT_TAG_PREFIX):
        return text  # already a tag — a tag must never be tagged again
    return f"{CHAT_TAG_PREFIX}{_digest(text)[:8]}"


def is_chat_tag(value: Any) -> bool:
    """True when the value is already a fingerprint, not an id."""
    return isinstance(value, str) and value.strip().startswith(CHAT_TAG_PREFIX)


def mask_emails(text: str) -> str:
    """
    Replaces every email address in free text with its fingerprint.

    Applied to everything that is about to be written into a file the
    repository commits — the history log and the self-test transcript.
    """
    if not text:
        return text
    return EMAIL_RE.sub(lambda match: recipient_tag(match.group(0)), text)


def mask_chat_ids(text: str) -> str:
    """
    Replaces every chat id in free text with its fingerprint.

    «чат=<the operator's id>» becomes «чат=чат#1a2b3c4d» — the sentence survives, the
    address does not. A number that is not introduced by the word «chat» is
    left alone: the log must still be able to say «проверок=7/7».
    """
    if not text:
        return text

    def _replace(match: "re.Match[str]") -> str:
        return f"{match.group('key')}{match.group('sep')}{chat_tag(match.group('id'))}"

    return CHAT_RE.sub(_replace, text)


def mask_sensitive(text: str) -> str:
    """Both masks, in one call — for everything that goes into a tracked file."""
    if not text:
        return text
    return mask_chat_ids(mask_emails(text))


def mask_emails_deep(value: Any) -> Any:
    """The same masking, through a JSON-shaped structure."""
    if isinstance(value, str):
        return mask_emails(value)
    if isinstance(value, list):
        return [mask_emails_deep(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_emails_deep(item) for key, item in value.items()}
    return value


def _mask_chat_value(value: Any) -> Any:
    """One chat id (or a list of them) becomes a fingerprint."""
    if value is None:
        return None
    if isinstance(value, list):
        return [chat_tag(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return chat_tag(value)
    text = str(value).strip()
    if not text:
        return value
    if is_chat_tag(text):
        return text
    if text.lstrip("-").isdigit():
        return chat_tag(text)
    return mask_chat_ids(text)


def mask_chat_ids_deep(value: Any) -> Any:
    """
    The same masking, through a JSON-shaped structure.

    The keys that HOLD a chat id are masked by their key («last_chat_id»),
    and every string is additionally masked by its text — so a chat id
    quoted in a message is caught even when it sits under an innocent key.
    """
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            if key in CHAT_KEYS:
                out[key] = _mask_chat_value(item)
            else:
                out[key] = mask_chat_ids_deep(item)
        return out
    if isinstance(value, list):
        return [mask_chat_ids_deep(item) for item in value]
    if isinstance(value, str):
        return mask_chat_ids(value)
    return value


def mask_sensitive_deep(value: Any) -> Any:
    """Both masks, through a JSON-shaped structure."""
    return mask_chat_ids_deep(mask_emails_deep(value))

