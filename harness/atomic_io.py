#!/usr/bin/env python3
"""
Atomic state writes.

Every JSON state file in this repository is COMMITTED and read by the next
cycle — `data/.cycle_supervisor.json` decides what the supervisor revives,
`data/duty_shift.json` holds the duty model, `active_models.json` holds the
fleet. A plain ``open(path, "w")`` + ``json.dump`` is a truncated file the
moment the runner dies mid-write, and then the next cycle inherits a
corrupted state it did not make.

So every writer goes through here: write to a temporary file in the SAME
directory, flush, then ``os.replace`` — one atomic rename, the reader sees
either the old file or the new one, never a half of either.
"""

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_json_atomic(path: Any, payload: Any, indent: int = 2) -> bool:
    """
    Writes `payload` as JSON to `path` atomically. True on success.

    A failure leaves the OLD file untouched — the caller's own ``except``
    decides what to say, the state never turns into a half-written file.
    """
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(target))
        return True
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        print(f"[atomic] could not write {target.name}: {type(e).__name__}: {e}")
        return False


def write_text_atomic(path: Any, text: str) -> bool:
    """Writes plain text atomically. Same contract as `write_json_atomic`."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(target))
        return True
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        print(f"[atomic] could not write {target.name}: {type(e).__name__}: {e}")
        return False
