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
"""

import hashlib
import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def recipient_tag(address: str) -> str:
    """
    A short, non-reversible fingerprint of an address.

    «настроен (sha256:1a2b3c4d)» says "an address was resolved and it is the
    same one as last time" without saying which address it is.
    """
    text = (address or "").strip()
    if not text:
        return "не задан"
    digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:8]
    return f"настроен (sha256:{digest})"


def mask_emails(text: str) -> str:
    """
    Replaces every email address in free text with its fingerprint.

    Applied to everything that is about to be written into a file the
    repository commits — the history log and the self-test transcript.
    """
    if not text:
        return text
    return EMAIL_RE.sub(lambda match: recipient_tag(match.group(0)), text)


def mask_emails_deep(value: Any) -> Any:
    """The same masking, through a JSON-shaped structure."""
    if isinstance(value, str):
        return mask_emails(value)
    if isinstance(value, list):
        return [mask_emails_deep(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_emails_deep(item) for key, item in value.items()}
    return value
