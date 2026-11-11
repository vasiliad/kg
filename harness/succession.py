#!/usr/bin/env python3
"""
The succession protocol: «разработчик умер — продукт живёт».

The founder's axiom (2026-11-11): a day may come when nobody comes — no one
to click, to answer, to decide. At any spot where the complex WAITS for a
human, that spot is not a safeguard but a countdown. So the complex measures
the one authority it cannot afford to lose track of: the founder's voice.
Every founder message in the durable Telegram log is a heartbeat of that
authority; while the heartbeats are fresh, the complex acts as it always
did. After `silence_days` of measured silence it switches to STEWARD mode
and acts under STANDING_ORDERS.md — the will the founder leaves in the
repository — and SAYS so in every report: a governance that changed
silently would be a coup, not a succession.

Honesty contract:
  * unknown is not dead: a missing or unreadable log keeps founder mode and
    names the reason — the same rule as every measured artefact here;
  * the will is the ONLY source of the steward's powers; this module reads
    it and never amends it;
  * nothing here speculates WHY the founder is silent. It measures THAT the
    voice has not been heard, and nothing more.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import atomic_io

HISTORY_FILE = PROJECT_ROOT / "data" / "telegram_history.txt"
STATE_FILE = PROJECT_ROOT / "data" / "succession.json"
ORDERS_FILE = PROJECT_ROOT / "STANDING_ORDERS.md"

DEFAULT_SILENCE_DAYS = 14.0

# A founder line looks like «[2026-11-01 15:30:07 UTC] USER: ...» in the
# durable log. Only USER is the founder's own voice; DUTY/SYSTEM/ARBITER
# lines are the complex talking to itself.
_FOUNDER_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]\s+USER\b", re.MULTILINE)

# A signature is a DATE after the marker: the template's «SIGNED: (нет — …)»
# placeholder must never count as a signed will.
_SIGNED_LINE = re.compile(r"^SIGNED:\s*\d{4}-\d{2}-\d{2}", re.MULTILINE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _configured_silence_days() -> Optional[float]:
    """The threshold from config.yaml (succession.silence_days), or None."""
    try:
        import yaml
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        value = (cfg.get("succession") or {}).get("silence_days")
        return float(value) if value is not None else None
    except Exception:
        return None


def _parse_iso(text: Any) -> Optional[datetime]:
    """A «2026-11-11T19:05:00Z» stamp, or None."""
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _configured_drill() -> Optional[Dict[str, str]]:
    """The declared drill window from config.yaml, or None."""
    try:
        import yaml
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        drill = (cfg.get("succession") or {}).get("drill") or None
        if isinstance(drill, dict) and drill.get("declared") and drill.get("ends"):
            return {"declared": str(drill["declared"]), "ends": str(drill["ends"])}
    except Exception:
        pass
    return None


def founder_last_seen(path: Path = HISTORY_FILE,
                      now: Optional[datetime] = None) -> Dict[str, Any]:
    """The founder's last word in the durable log, measured.

    Returns {"measured": True, "at": iso, "age_days": float} — or
    {"measured": False, "reason": ...}: an unreadable or empty witness is
    NOT a dead founder, and nothing here may guess one into existence."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"measured": False,
                "reason": f"журнал недоступен: {e.__class__.__name__}"}
    last = None
    for m in _FOUNDER_LINE.finditer(text):
        last = m.group(1)
    if last is None:
        return {"measured": False,
                "reason": "в журнале нет ни одного слова основателя"}
    try:
        moment = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return {"measured": False, "reason": "метка времени не разобрана"}
    age_days = ((now or _now()) - moment).total_seconds() / 86400.0
    return {"measured": True, "at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "age_days": round(max(0.0, age_days), 2)}


def _orders_armed(path: Path = ORDERS_FILE) -> bool:
    """Whether the founder signed the will: a DATED SIGNED line. The
    template's own «SIGNED: (нет — …)» placeholder is not a signature."""
    try:
        return bool(_SIGNED_LINE.search(Path(path).read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def measure(silence_days: Optional[float] = None,
            path: Path = HISTORY_FILE,
            state_path: Path = STATE_FILE,
            orders_path: Path = ORDERS_FILE,
            drill: Optional[Dict[str, str]] = None,
            now: Optional[datetime] = None) -> Dict[str, Any]:
    """Measure the governance: who may decide today, and on what evidence.

    Steward mode = a DECLARED drill window covers now (the founder's own
    exercise: «час икс настал») — OR the founder's MEASURED silence is
    older than the threshold (explicit argument, else config.yaml, else
    the default). The switch moment survives in the artefact: the first
    steward measure stamps `steward_since`, later measures keep it; when
    the stewardship ends (the founder is back, the drill over, the log
    fresh), the stamp clears."""
    if silence_days is None:
        silence_days = _configured_silence_days()
    threshold = float(silence_days) if silence_days is not None else DEFAULT_SILENCE_DAYS
    seen = founder_last_seen(path, now)

    window = drill if drill is not None else _configured_drill()
    drill_active = False
    drill_report: Optional[Dict[str, Any]] = None
    if window:
        declared_at = _parse_iso(window.get("declared"))
        ends_at = _parse_iso(window.get("ends"))
        moment = now or _now()
        drill_active = bool(declared_at and ends_at and declared_at <= moment < ends_at)
        drill_report = {"declared": window.get("declared"), "ends": window.get("ends"),
                        "active": drill_active}

    measured_silent = bool(seen.get("measured") and seen["age_days"] > threshold)
    steward = drill_active or measured_silent

    state: Dict[str, Any] = {}
    try:
        if Path(state_path).exists():
            state = json.loads(Path(state_path).read_text(encoding="utf-8")) or {}
    except Exception:
        state = {}
    steward_since = state.get("steward_since")
    if steward and not steward_since:
        # The drill stamps the DECLARED moment, not the discovery moment:
        # hour X is when the founder said it, not when the cycle noticed.
        steward_since = window.get("declared") if drill_active else (now or _now()).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not steward:
        steward_since = None

    report = {
        "measured_at": (now or _now()).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "founder_last_seen": seen,
        "silence_days": threshold,
        "steward_mode": steward,
        "steward_since": steward_since,
        "drill": drill_report,
        "orders_armed": _orders_armed(orders_path),
        "orders_file": Path(orders_path).name,
    }
    # Timestamps and verdicts only — the history file carries chat text and
    # must never leak into an artefact.
    atomic_io.write_json_atomic(state_path, report)
    return report


def describe(report: Dict[str, Any]) -> str:
    """One human line: who governs the complex today, and why."""
    seen = report.get("founder_last_seen") or {}
    threshold = report.get("silence_days", DEFAULT_SILENCE_DAYS)
    will = "завещание на месте" if report.get("orders_armed") else "завещание не подписано"
    drill = report.get("drill") or {}
    if drill.get("active"):
        return (f"УЧЕНИЕ (объявлено основателем {str(drill.get('declared', ''))[:10]}): "
                f"режим наместника до {str(drill.get('ends', ''))[:10]}; "
                f"измеренное молчание в журнале: "
                f"{seen.get('age_days', '?')} сут — учение, не диагноз ({will})")
    if not seen.get("measured"):
        return (f"Правление: основатель (молчание не измерено: "
                f"{seen.get('reason')}; {will})")
    if report.get("steward_mode"):
        since = report.get("steward_since") or "?"
        return (f"РЕЖИМ НАМЕСТНИКА: основатель молчит {seen['age_days']} сут "
                f"(порог {threshold}; с {since}); действуем по "
                f"{report.get('orders_file', 'STANDING_ORDERS.md')} ({will})")
    return (f"Правление: основатель на связи — последнее слово "
            f"{seen['age_days']} сут назад (порог {threshold}; {will})")
