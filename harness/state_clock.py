#!/usr/bin/env python3
"""
THE CLOCK OF THE MEASURED STATE — one rule for everyone who asks
«when was this measured?».

Measured 2026-11-06: the emergency ladder read `active_models.json`
["timestamp"] — a field the emergency runner writes ITSELF («now»). So the
ladder measured «when did I last run», not «when was the fleet measured».
The consequences were measured too:

  * at 13:37 the ladder fired with «State stale by 14.4h» while the fleet
    had been measured 6.4h earlier — the number in the reason was false;
  * two hours later the very same stamp would have read «up to date» with
    a ten-hour-old measurement — the ladder would have gone silent exactly
    when it was needed;
  * `emergency_mode: true` then stayed in the state for the rest of the
    day, so the operator's letter kept announcing «Экстренный CPU
    fallback» long after the fleet had been measured again.

A clock that measures the wrong thing cannot see an outage, and it cries
wolf when there is none. So the clock lives HERE, and everybody who reads
the state — the ladder, the letter, the reports — asks the same question
the same way.

The order of the witnesses is deliberate:
  1. `fleet_decision.tuned_at` — the tuner's own stamp; only a cycle that
     MEASURED writes it;
  2. `active_fleet[].measured.score_as_of` — the newest measurement of the
     roster itself;
  3. `last_updated` — the generic stamp, which ANY writer may touch, so it
     is the last resort and always says so.

The emergency runner's own `timestamp` is NEVER a witness: it is the
moment the ladder ran, not the state the ladder judged.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# The fleet is measured every ~6h. A measurement older than this is stale.
STATE_STALE_AFTER_H = 6.0


def parse_ts(text: Any) -> Optional[datetime]:
    """Any stamp the state file carries, as an aware UTC datetime."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.endswith("UTC"):
        raw = raw[:-3].strip().replace(" ", "T", 1) + "+00:00"
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def fleet_state_stamp(state: Dict[str, Any]) -> Tuple[Optional[datetime], str]:
    """
    When the fleet was last MEASURED — and which field says so.

    Returns (None, reason) when the state carries no measurement at all:
    «I cannot read it» is never «it is fresh».
    """
    state = state or {}
    decision = state.get("fleet_decision") or {}
    stamp = parse_ts(decision.get("tuned_at"))
    if stamp:
        return stamp, "fleet_decision.tuned_at"
    newest = None
    for member in state.get("active_fleet") or []:
        if not isinstance(member, dict):
            continue
        measured = member.get("measured") or {}
        candidate = parse_ts(measured.get("score_as_of") or member.get("score_as_of"))
        if candidate and (newest is None or candidate > newest):
            newest = candidate
    if newest:
        return newest, "active_fleet[].score_as_of"
    stamp = parse_ts(state.get("last_updated"))
    if stamp:
        return stamp, "last_updated"
    return None, "нет отметки замера"


def fleet_state_age_hours(
    state: Dict[str, Any], now: Optional[datetime] = None
) -> Tuple[Optional[float], str]:
    """How many hours ago the fleet was measured (None when unknown)."""
    stamp, source = fleet_state_stamp(state)
    if stamp is None:
        return None, source
    moment = now or datetime.now(timezone.utc)
    return round((moment - stamp).total_seconds() / 3600.0, 2), source


def fleet_state_line_ru(
    state: Dict[str, Any], now: Optional[datetime] = None
) -> str:
    """
    The measurement age in one Russian line — for the letter and the
    reports. It never rounds a fresh measurement into «долго»: the number
    is the number, and the source is named.
    """
    age_hours, source = fleet_state_age_hours(state, now=now)
    if age_hours is None:
        return "флот: замера нет (нечем подтвердить состояние)"
    if age_hours < 0:
        age_hours = 0.0
    if age_hours < 1:
        return f"флот измерен {round(age_hours * 60)} мин назад ({source})"
    return f"флот измерен {round(age_hours, 1)} ч назад ({source})"
