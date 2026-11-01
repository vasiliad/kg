#!/usr/bin/env python3
"""
One-off: replace every email address in the tracked state files with its
fingerprint. Used on 2026-11-01 to undo the leak of the operator's address
into the public repository (nim_history.log, data/bot_selftest_report.json).

The code that writes those files now masks addresses on the way in
(harness/redaction.py); this script only cleans up what was already there.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import redaction

TARGETS = [
    "nim_history.log",
    "data/bot_selftest_report.json",
    "data/bot_transport_report.json",
    "data/.bot_settings.json",
]


def main() -> int:
    changed = []
    for rel in TARGETS:
        path = Path(rel)
        if not path.exists():
            print(f"[skip] {rel} does not exist")
            continue
        original = path.read_text(encoding="utf-8")
        cleaned = redaction.mask_emails(original)
        if cleaned == original:
            print(f"[clean] {rel} holds no address")
            continue
        path.write_text(cleaned, encoding="utf-8")
        changed.append(rel)
        print(f"[fixed] {rel}")
    print(f"[done] {len(changed)} file(s) redacted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
