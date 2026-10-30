#!/usr/bin/env python3
"""
Duty team: the roster the keeper (local CPU model) commands.

Architecture the operator asked for:
  - the project chief ("хранитель системы") is a LOCAL model on CPU —
    sovereign, no network needed;
  - the duty models are NETWORK models, and there can be two or three of
    them on one task, chosen for what the task needs.

This module turns the tuner's measured fleet into that duty team:

  roster()    keeper + duty models with measured scores, latencies, roles
  dispatch()  a deterministic dispatch plan for a free-form task: which duty
              models to call, in what order, and why (in plain Russian)

The planner is deterministic on purpose: the keeper is a small CPU model and
must never be the single point of failure for routing. Its verdict travels
WITH the plan; if the keeper is unavailable the plan still executes.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "active_models.json"

# The project chief: a local sovereign model. It is never a NIM catalogue
# entry and must never be reported as "removed by the supplier".
KEEPER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
KEEPER_TITLE = "Хранитель системы (локальный CPU)"

ROLE_TITLES = {
    "duty_primary": "дежурный основной",
    "duty_fast": "дежурный быстрый",
    "duty_code": "дежурный кодер",
    "duty_flagship": "дежурный флагман",
    "duty_fallback": "дежурный резерв",
    "reserve": "резерв",
}

# Free-form task classification: which measured strengths does this task need?
TASK_HINTS = {
    "code": [
        r"\bкод\b", r"python", r"скрипт", r"функци", r"ошибк", r"\bbug\b",
        r"рефактор", r"regex", r"\bsql\b", r"\bapi\b", r"баг", r"отлад", r"напиши.*(модуль|класс|функцию)",
    ],
    "analysis": [
        r"почему", r"проанализ", r"сравни", r"объясни", r"стратег", r"причин",
        r"докаж", r"посчитай", r"вероятн", r"матем", r"исследов", r"оцени",
    ],
    "quick": [
        r"\?\s*$", r"статус", r"сколько", r"когда", r"какой", r"проверь",
        r"покажи", r"напомни",
    ],
}

# Which duty roles to try first, per task type. The keeper dispatches along
# this order until max_calls distinct models are collected.
ROLE_ORDER = {
    "code": ["duty_code", "duty_primary", "duty_fast"],
    "analysis": ["duty_flagship", "duty_primary", "duty_code"],
    "quick": ["duty_fast", "duty_primary"],
    "general": ["duty_primary", "duty_fast", "duty_code"],
}


class DutyTeam:
    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = Path(state_path) if state_path else STATE_FILE

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------
    def load_roster(self) -> Dict[str, Any]:
        """
        Returns {"keeper": {...}, "duty": [entries sorted by measured score]}.
        Never raises: a missing or broken state yields an empty duty list and
        the caller falls back to the default executor.
        """
        duty: List[Dict[str, Any]] = []
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            fleet = state.get("active_fleet") or []
            for m in fleet:
                if not isinstance(m, dict) or not m.get("id"):
                    continue
                if str(m.get("id")) in (KEEPER_ID, "Qwen/Qwen2.5-0.5B-Instruct"):
                    continue  # local models are not network duty
                duty.append({
                    "id": m["id"],
                    "roles": m.get("roles") or [],
                    "tier": m.get("tier", ""),
                    "score": m.get("quality_score"),
                    "verified": bool(m.get("score_verified")),
                    "latency_sec": (m.get("measured") or {}).get("latency_sec"),
                    "params": m.get("recommended_params") or {},
                    "vendor": (m["id"].split("/", 1)[0].lower()),
                })
        except Exception:
            pass
        duty.sort(key=lambda m: (m["score"] is not None, m["score"] or 0), reverse=True)
        return {"keeper": {"id": KEEPER_ID, "title": KEEPER_TITLE}, "duty": duty}

    def format_roster(self) -> str:
        """Human-readable team roster for Telegram (HTML-safe text)."""
        roster = self.load_roster()
        lines = [f"🧑‍💼 <b>СОСТАВ КОМАНДЫ КОМПЛЕКСА</b>\n"]
        keeper = roster["keeper"]
        lines.append(
            f"👑 <b>{keeper['title']}</b>\n   • {keeper['id']} — начальник проекта: "
            "распределяет задачи, работает без сети\n"
        )
        duty = roster["duty"]
        if not duty:
            lines.append("⚠️ Дежурные сетевые модели не измерены — тюнер ещё не запускался. Нажмите «🔄 Обновить флот».")
            return "\n".join(lines)
        lines.append(f"⚡ <b>Дежурные сетевые модели ({len(duty)}):</b>")
        for idx, m in enumerate(duty, 1):
            roles = ", ".join(ROLE_TITLES.get(r, r) for r in m["roles"]) or "без роли"
            score = f"{m['score']}" if m["score"] is not None else "UNRANKED"
            check = "✅ замер" if m["verified"] else "⚠️ сид"
            lat = f", {m['latency_sec']}с" if m["latency_sec"] else ""
            lines.append(
                f"<b>{idx}. {m['id']}</b>\n   • роль: {roles}\n"
                f"   • балл: <b>{score}</b> ({check}{lat})\n   • вендор: {m['vendor']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dispatch planning
    # ------------------------------------------------------------------
    def classify(self, task_text: str) -> str:
        low = (task_text or "").lower()
        scores = {}
        for kind, patterns in TASK_HINTS.items():
            scores[kind] = sum(1 for p in patterns if re.search(p, low))
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "general"

    def dispatch(self, task_text: str, max_calls: int = 3) -> Dict[str, Any]:
        """
        Deterministic dispatch plan: ordered duty models + plain-Russian
        rationale the keeper's verdict can quote. Never raises.
        """
        roster = self.load_roster()
        duty = roster["duty"]
        if not duty:
            return {"task_type": "general", "calls": [], "rationale": "Дежурные не измерены."}

        task_type = self.classify(task_text)
        by_role: Dict[str, List[Dict[str, Any]]] = {}
        for m in duty:
            for r in m["roles"] or ["reserve"]:
                by_role.setdefault(r, []).append(m)

        calls: List[Dict[str, Any]] = []
        seen: set = set()
        for role in ROLE_ORDER.get(task_type, ROLE_ORDER["general"]):
            for m in by_role.get(role, []):
                if m["id"] not in seen:
                    calls.append({**m, "primary_role": role})
                    seen.add(m["id"])
                    break
            if len(calls) >= max_calls:
                break
        # fill with the best remaining measured models
        for m in duty:
            if len(calls) >= max_calls:
                break
            if m["id"] not in seen:
                calls.append({**m, "primary_role": "duty_fallback"})
                seen.add(m["id"])

        why = {
            "code": "задача на код — первым идёт измеренный кодер",
            "analysis": "задача на анализ/рассуждение — первым идёт флагман или основной",
            "quick": "быстрый вопрос — первым идёт самая быстрая дежурная",
            "general": "общая задача — основным дежурным, остальные как усиление",
        }[task_type]
        names = " → ".join(f"{c['id']} ({ROLE_TITLES.get(c['primary_role'], c['primary_role'])})" for c in calls)
        rationale = f"Тип: {task_type}; {why}. Порядок вызова: {names}."
        return {"task_type": task_type, "calls": calls, "rationale": rationale}
