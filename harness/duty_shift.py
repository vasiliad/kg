#!/usr/bin/env python3
"""
The duty shift: ONE model serves the operator for six hours.

The operator's rule (2026-11-02): «выбирать только одну дежурную модель на
6 часов, а не устраивать распределение задач по моделям — дежурная модель
меняется только если она перестает отвечать быстро».

Before this, every free-form task was dispatched to two or three duty
models at once and got a second opinion for code and analysis. The operator
saw three voices, three latencies and three prices for one question — and
the model that answered «Найди на озон» with raw JSON was among them.

The shift:
  * the KEEPER (the local CPU model) chooses ONE model from the measured
    roster when a shift starts — not a rule, the keeper's own verdict;
  * every task goes to that model until the shift ends (six hours);
  * the shift is broken ONLY when the holder stops answering quickly: it
    failed, or it answered slower than the cap;
  * if the local keeper cannot run (the cloud runner has no torch), the
    choice falls back to the MEASURED best — and says so.

The shift lives in `data/duty_shift.json`, which the bot daemon commits
between windows: a 6-hour shift outlives a 50-minute runner job, so the
state must survive the restart.

Honesty contract: nothing here invents a quality score. The keeper sees
the MEASURED roster (score, latency, verified flag) and picks from it; the
fallback uses the same measured numbers.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import atomic_io

SHIFT_FILE = PROJECT_ROOT / "data" / "duty_shift.json"

# How long one model stays on duty.
SHIFT_HOURS = 6

# "Отвечает быстро": a duty answer slower than this breaks the shift.
# The measured duty latencies are 0.7-5 s, so 30 s is a generous cap that
# still means "the model stopped answering quickly".
SLOW_RESPONSE_SEC = 30.0

# A model that takes 5x its own measured latency is slow even if the
# absolute number looks small.
LATENCY_FACTOR = 5.0
LATENCY_FLOOR_SEC = 15.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


# --------------------------------------------------------------------------
# The shift file
# --------------------------------------------------------------------------

def load_shift(path: Path = SHIFT_FILE) -> Dict[str, Any]:
    """The current shift, or an empty dict. Never raises."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_shift(shift: Dict[str, Any], path: Path = SHIFT_FILE) -> None:
    """Persists the shift. The bot daemon commits this file between windows."""
    # Atomic: a truncated shift file would look like "no shift" and the
    # keeper would silently start a new 6-hour term mid-shift.
    if not atomic_io.write_json_atomic(path, shift):
        print("[!] Duty shift could not be saved")


def shift_expired(shift: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Whether the six hours are over."""
    if not shift or not shift.get("model_id"):
        return True
    until = _parse_iso(shift.get("until", ""))
    if until is None:
        return True
    return (now or _now()) >= until


def shift_broken(shift: Dict[str, Any]) -> bool:
    """Whether the holder stopped answering quickly and the shift was cut."""
    return bool(shift.get("broken_at"))


# --------------------------------------------------------------------------
# Fast or not
# --------------------------------------------------------------------------

def is_fast_response(elapsed_sec: Any, measured_latency: Any = None) -> bool:
    """
    Whether the duty model answered quickly enough to keep its shift.

    The cap is absolute (30 s) and relative (5x the model's own measured
    latency, never below 15 s): a model that normally answers in 2 s and
    took 20 s is slow even though 20 s is under the absolute cap.
    """
    try:
        elapsed = float(elapsed_sec)
    except Exception:
        return False
    if elapsed < 0:
        return False
    cap = SLOW_RESPONSE_SEC
    try:
        measured = float(measured_latency)
        if measured > 0:
            cap = max(cap, max(measured * LATENCY_FACTOR, LATENCY_FLOOR_SEC))
    except Exception:
        pass
    return elapsed <= cap


# --------------------------------------------------------------------------
# The keeper chooses
# --------------------------------------------------------------------------

def _roster_candidates(roster: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The measured network models the keeper may choose from."""
    duty = (roster or {}).get("duty") or []
    return [m for m in duty if isinstance(m, dict) and m.get("id")]


def _match_roster_id(text: str, candidates: List[Dict[str, Any]]) -> str:
    """
    The roster id the keeper named — or an empty string.

    Longest id first: «nvidia/nemotron-3-ultra-550b-a55b» must not be
    matched by a shorter id that happens to be a prefix of it.
    """
    if not text:
        return ""
    flat = str(text).strip()
    for model in sorted(candidates, key=lambda m: len(str(m.get("id"))), reverse=True):
        model_id = str(model.get("id"))
        if re.search(rf"(?<![\w./-]){re.escape(model_id)}(?![\w./-])", flat):
            return model_id
    return ""


def keeper_task(candidates: List[Dict[str, Any]]) -> str:
    """The question the keeper is asked, with the measured roster."""
    rows = []
    for m in candidates:
        score = m.get("score")
        score_txt = f"{score}" if score is not None else "UNRANKED"
        verified = "живой замер" if m.get("verified") else "сид, не проверено"
        latency = m.get("latency_sec")
        lat_txt = f"{latency}с" if isinstance(latency, (int, float)) else "не измерен"
        roles = ", ".join(m.get("roles") or []) or "без роли"
        rows.append(f"- {m.get('id')} | балл {score_txt} ({verified}) | задержка {lat_txt} | роли: {roles}")
    roster_text = "\n".join(rows)
    return (
        "Ты Хранитель автономного комплекса. Твоя задача — выбрать ОДНУ дежурную "
        "модель на ближайшие 6 часов.\n"
        "Выбирай по ЖИВЫМ ЗАМЕРАМ: балл, задержку и проверенность. Модель должна "
        "отвечать быстро и отвечать по-русски.\n"
        "Ответь ТОЛЬКО идентификатором выбранной модели — одной строкой, без "
        "пояснений и без кавычек.\n\n"
        f"Измеренный состав:\n{roster_text}"
    )


def pick_duty_model(
    roster: Dict[str, Any],
    keeper: Any = None,
    exclude: Optional[str] = None,
) -> Tuple[str, str]:
    """
    The keeper's choice of the duty model, and the reason for it.

    The keeper is the LOCAL CPU model (Qwen2.5-1.5B). When it cannot run —
    the cloud runner has no torch — the choice falls back to the measured
    best and SAYS so: a silent substitution would be a lie in the log.

    `exclude` keeps a model that just broke its shift out of the choice:
    the keeper may choose anyone else, but not the one that stopped
    answering.
    """
    candidates = _roster_candidates(roster)
    if exclude:
        excluded = str(exclude)
        remaining = [m for m in candidates if str(m.get("id")) != excluded]
        # If the broken model was the only one measured, it stays — a shift
        # with nobody is worse than a shift with a model that failed.
        if remaining:
            candidates = remaining
    if not candidates:
        return "", "дежурные модели не измерены"

    picked = ""
    reason = ""
    try:
        if keeper is None:
            from harness.sovereign_cpu_engine import SovereignCPUEngine
            keeper = SovereignCPUEngine()
        result = keeper.execute_sovereign_mission(
            task_description=keeper_task(candidates),
            context={"candidate_models": candidates},
        )
        # The neural keeper answers with prose; the deterministic kernel
        # answers with a canned report that names no model.
        text = str(result.get("output") or "")
        picked = _match_roster_id(text, candidates)
        if picked:
            reason = "выбрал Хранитель (локальная CPU-модель)"
        elif str(result.get("engine") or "") == "deterministic_kernel":
            reason = "Хранитель без нейросети (детерминированное ядро)"
    except Exception as e:
        reason = f"Хранитель недоступен ({type(e).__name__})"

    if picked:
        return picked, reason

    if not reason:
        reason = "Хранитель не назвал модель из состава"
    # Fallback: the MEASURED best, fastest on a tie. The same evidence the
    # keeper would have seen.
    ranked = sorted(
        candidates,
        key=lambda m: (-(float(m.get("score") or 0.0)), float(m.get("latency_sec") or 999.0)),
    )
    return str(ranked[0].get("id")), f"{reason}; выбрано по живому замеру"


# --------------------------------------------------------------------------
# The shift itself
# --------------------------------------------------------------------------

def start_shift(
    roster: Dict[str, Any],
    keeper: Any = None,
    now: Optional[datetime] = None,
    path: Path = SHIFT_FILE,
    exclude: Optional[str] = None,
) -> Dict[str, Any]:
    """Starts a new six-hour shift with the keeper's choice."""
    moment = now or _now()
    model_id, reason = pick_duty_model(roster, keeper, exclude=exclude)
    shift = {
        "model_id": model_id,
        "started_at": _iso(moment),
        "until": _iso(moment + timedelta(hours=SHIFT_HOURS)),
        "chosen_by": reason,
        "calls": 0,
        "failures": 0,
        "last_latency_sec": None,
        "broken_at": "",
        "broken_reason": "",
    }
    if model_id:
        save_shift(shift, path)
    return shift


def current_duty(
    roster: Dict[str, Any],
    keeper: Any = None,
    now: Optional[datetime] = None,
    path: Path = SHIFT_FILE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The model on duty right now, and the shift it belongs to.

    The shift is kept while it is alive, its holder is still in the roster
    and the holder has not broken it. Otherwise a new shift starts — and
    the keeper chooses again, without the model that just failed.
    """
    moment = now or _now()
    shift = load_shift(path)
    candidates = {str(m.get("id")) for m in _roster_candidates(roster)}

    alive = (
        shift
        and shift.get("model_id")
        and not shift_expired(shift, moment)
        and not shift_broken(shift)
        and (not candidates or str(shift.get("model_id")) in candidates)
    )
    if alive:
        return shift, shift

    reason = ""
    exclude = None
    if shift.get("model_id"):
        if shift_broken(shift):
            reason = "дежурная модель перестала отвечать быстро"
            exclude = str(shift.get("model_id"))
        elif shift_expired(shift, moment):
            reason = "смена на 6 часов истекла"
        else:
            reason = "дежурной модели больше нет в измеренном составе"
    new_shift = start_shift(roster, keeper, moment, path, exclude=exclude)
    if reason:
        new_shift["started_because"] = reason
        save_shift(new_shift, path)
    return new_shift, new_shift


def note_outcome(
    model_id: str,
    elapsed_sec: Any,
    ok: bool,
    measured_latency: Any = None,
    path: Path = SHIFT_FILE,
) -> Dict[str, Any]:
    """
    Records one answered task against the shift.

    A model that failed, or that answered slower than the cap, LOSES the
    shift: the next task gets a new model. That is the only reason the duty
    model changes inside the six hours.
    """
    shift = load_shift(path)
    if not shift or str(shift.get("model_id")) != str(model_id):
        return shift
    shift["calls"] = int(shift.get("calls", 0)) + 1
    try:
        elapsed = float(elapsed_sec)
        shift["last_latency_sec"] = round(elapsed, 2)
    except Exception:
        elapsed = None
    if not ok:
        shift["failures"] = int(shift.get("failures", 0)) + 1
        shift["broken_at"] = _iso(_now())
        shift["broken_reason"] = "модель не ответила"
    elif elapsed is not None and not is_fast_response(elapsed, measured_latency):
        shift["failures"] = int(shift.get("failures", 0)) + 1
        shift["broken_at"] = _iso(_now())
        shift["broken_reason"] = (
            f"ответ занял {round(elapsed, 1)}с — медленнее допустимого"
        )
    save_shift(shift, path)
    return shift


def describe_shift(shift: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """
    One human-readable line about the duty shift — and the TRUTH about it.

    Measured 2026-11-06: the shift file had not changed since 2026-11-02.
    The shift was rotated only when a task arrived, so a complex with no
    traffic kept a dead shift for days, and everything that read it called
    it «Текущая смена» — a fact that had expired four days earlier,
    repeated as if it were alive. An expired shift says so; a broken one
    says why.
    """
    if not shift or not shift.get("model_id"):
        return "Дежурная модель не назначена: измеренного состава нет."
    moment = now or _now()
    until = _parse_iso(shift.get("until", ""))
    until_txt = until.strftime("%d.%m %H:%M UTC") if until else "—"
    who = shift.get("chosen_by", "")
    if shift_broken(shift):
        reason = shift.get("broken_reason") or "без причины"
        return (
            f"Дежурная модель: {shift['model_id']} — смена ПРЕРВАНА "
            f"({reason}; была до {until_txt}; {who})."
        )
    if shift_expired(shift, moment):
        hours = _hours_since(until, moment)
        age = f"ей {hours} ч" if hours is not None else "неизвестно сколько"
        return (
            f"Дежурная модель: {shift['model_id']} — смена ИСТЕКЛА "
            f"(была до {until_txt}, {age}; {who})."
        )
    return (
        f"Дежурная модель: {shift['model_id']} (смена до {until_txt}; {who})."
    )


def shift_status_line(shift: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """
    The shift as a FACT for the complex's own reports: who, how fast, and
    whether the shift is still alive. Used by the forge's facts digest, so
    the self-improvement reads a live fact instead of an expired one.
    """
    if not shift or not shift.get("model_id"):
        return "Дежурная смена не назначена."
    moment = now or _now()
    latency = shift.get("last_latency_sec")
    latency_txt = f"{latency}с" if isinstance(latency, (int, float)) else "замеров нет"
    who = shift.get("chosen_by", "")
    until = _parse_iso(shift.get("until", ""))
    if shift_broken(shift):
        state = f"смена прервана: {shift.get('broken_reason') or 'без причины'}"
    elif shift_expired(shift, moment):
        hours = _hours_since(until, moment)
        state = (f"смена ИСТЕКЛА ({hours} ч назад)" if hours is not None
                 else "смена ИСТЕКЛА")
    else:
        state = "смена живая"
    return (
        f"Дежурная смена: {shift['model_id']}, задержка {latency_txt}, "
        f"{state}; выбрана: {who}"
    )


def _hours_since(moment: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    """How many hours ago that moment was (None when it is unknown)."""
    if moment is None:
        return None
    delta = (now or _now()) - moment
    return round(max(0.0, delta.total_seconds()) / 3600.0, 1)


def rotate_if_expired(
    roster: Dict[str, Any],
    keeper: Any = None,
    now: Optional[datetime] = None,
    path: Path = SHIFT_FILE,
) -> Tuple[Dict[str, Any], bool]:
    """
    Rotates the shift when it is no longer alive, and says whether it did.

    Measured 2026-11-06: the shift was rotated ONLY when a task arrived
    (the bot's message handler called current_duty). A complex with no
    traffic therefore kept an expired shift for days — and the letter, the
    bot's status and the forge's own facts repeated it as «текущая смена».
    The cycle that MEASURES the roster now rotates the shift: the roster it
    just measured is the evidence for the next shift, and a shift that has
    run out is not a shift any more.
    """
    moment = now or _now()
    before = load_shift(path)
    shift, _info = current_duty(roster, keeper=keeper, now=moment, path=path)
    rotated = bool(shift.get("model_id")) and (
        not before
        or str(before.get("started_at")) != str(shift.get("started_at"))
        or str(before.get("model_id")) != str(shift.get("model_id"))
    )
    return shift, rotated


if __name__ == "__main__":
    from harness.duty_team import DutyTeam

    roster = DutyTeam().load_roster()
    shift, _ = current_duty(roster)
    print(describe_shift(shift))
