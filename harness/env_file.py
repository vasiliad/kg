#!/usr/bin/env python3
"""
The project's own `.env` loader — ONE implementation, used everywhere.

Measured 2026-11-04: `harness/mail_accounts.py` loaded `.env` itself (its own
private copy of the idiom), while every module that reads the VAULT SALT or the
Drive credentials only looked at the process environment. So the operator's own
instruction — «python scripts/seal_mail_accounts.py --check» — failed on the
local machine with «neither VAULT_SALT nor REDACTION_SALT is set», even though
the letter says the salt lives in `.env` for local runs.

Two spellings of one loader is how the next reader «fixes» the wrong one. So
the loader lives here, and the modules that need the environment call it.

The rules:
  - the PROCESS environment wins (a GitHub Actions secret is not overridden by
    a file that happens to be on the runner);
  - `.env` never leaves the machine — it is gitignored, and this module never
    prints a value;
  - a broken `.env` must not crash a cycle: it is a convenience, not a
    contract. A missing salt is still reported loudly by the caller.
"""

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Optional[Path] = None) -> int:
    """
    Loads `KEY=VALUE` lines from `.env` into `os.environ`.

    Returns the number of keys that were actually added. Never raises: a
    `.env` that cannot be read is the same as no `.env` at all.
    """
    env_file = Path(path) if path else (PROJECT_ROOT / ".env")
    if not env_file.exists():
        return 0
    loaded = 0
    try:
        with open(env_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                # setdefault: the process environment (a cloud secret) wins.
                if key not in os.environ:
                    os.environ[key] = value.strip()
                    loaded += 1
    except Exception:
        return loaded
    return loaded
