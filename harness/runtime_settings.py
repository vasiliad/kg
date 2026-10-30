#!/usr/bin/env python3
"""
Runtime-settable configuration that must survive runner restarts.

GitHub Actions secrets are frozen when a job starts, so a value the operator
changes from Telegram cannot be stored in an environment variable — it would
be gone the next time the runner is recycled. Such values are persisted in
data/.bot_settings.json, which the cloud bot daemon commits back into the
repository, and are read back on the next runner start.

Resolution order for a setting:
    1. data/.bot_settings.json  (runtime override — wins, it is the newest
       operator intent and survives restarts)
    2. environment variable      (GitHub secret / default configuration)
    3. caller-supplied fallback
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "data" / ".bot_settings.json"

# The display name that appears in the "From:" field of every outgoing report.
# It is a runtime setting, not a constant: the operator can change the
# signature from Telegram, and the change has to survive runner recycling.
DEFAULT_SENDER_NAME = "Автономный комплекс KG"


def _read() -> Dict[str, Any]:
    """Reads the persisted runtime settings (never raises)."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Returns the effective value of a setting: the persisted runtime override
    if present, otherwise the environment value, otherwise `default`.
    """
    value = _read().get(name)
    if value:
        return str(value)
    env_value = os.getenv(name)
    if env_value:
        return env_value
    return default


def set_setting(name: str, value: str) -> None:
    """Persists a runtime setting so it survives the runner being recycled."""
    settings = _read()
    settings[name] = value
    settings["updated_at"] = datetime.now(timezone.utc).isoformat()

    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SETTINGS_FILE.with_suffix(SETTINGS_FILE.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_path), str(SETTINGS_FILE))

    # Keep the current process in sync so the change takes effect immediately.
    os.environ[name] = value


def all_settings() -> Dict[str, Any]:
    """Returns every persisted runtime setting."""
    return _read()


def get_recipient_email(default: str = "") -> str:
    """The address that daily reports are delivered to."""
    return get_setting("DISPATCH_RECIPIENT_EMAIL", default) or default


def get_sender_name(default: str = DEFAULT_SENDER_NAME) -> str:
    """The display name shown in the "From:" field of outgoing reports."""
    return get_setting("GMAIL_SENDER_NAME", default) or default


def set_sender_name(name: str) -> str:
    """Persists a new sender signature (survives runner recycling)."""
    clean = " ".join((name or "").split()).strip()
    if not clean:
        raise ValueError("Пустая подпись отправителя")
    # Strip characters that would break the RFC 5322 header.
    clean = clean.replace("<", "").replace(">", "").replace("\n", " ").replace("\r", " ")
    if len(clean) > 60:
        clean = clean[:60].strip()
    set_setting("GMAIL_SENDER_NAME", clean)
    return clean
