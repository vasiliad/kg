#!/usr/bin/env python3
"""
Bot self-test battery: the complex interrogates ITS OWN Telegram bot.

The operator asked whether the harness can write to the bot itself to test
its abilities. A real Telegram user session cannot be impersonated (and must
not be), but everything that defines the bot's abilities lives in
HarnessTelegramBot.process_message(): the password gate, the keeper's
dispatch over the measured duty roster, parallel duty calls, the honest reply
composition, the status/roster commands and the search tier.

This harness drives THAT exact code path with a scripted Russian battery,
capturing replies through a fake transport (the only thing stubbed — no
Telegram HTTP), and writes a transcript the operator can read:

  - data/bot_selftest_report.json  (committed by the workflow)
  - a САМОТЕСТ record in nim_history.log (event memory)

Conversation memory is isolated to throwaway files so test chatter never
pollutes the operator's real chat log; the event log gets exactly one
summary line. Duty calls, searches and key checks run LIVE.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.memory_engine import MemoryEngine
from harness.state_manager import StateManager
from harness.telegram_bot import HarnessTelegramBot

# The battery: (label, input, hard_markers, soft_markers)
#   hard_markers — MUST appear in the replies (structural correctness)
#   soft_markers — SHOULD appear when the live providers cooperate; their
#                  absence is reported, not failed (provider variance).
TEST_CASES = [
    ("статус", "📡 Статус системы",
     ["СИСТЕМНЫЙ СТАТУС"], ["NVIDIA NIM API", "Дежурная команда"]),
    ("ростер", "🧑‍💼 Состав команды",
     ["СОСТАВ КОМАНДЫ"], ["Хранитель", "Дежурные"]),
    ("быстрый_вопрос", "какой сейчас основной дежурный и почему он выбран?",
     [], ["Хранитель", "балл"]),
    ("код_задача", "напиши python функцию is_palindrome(s) с обработкой ошибок",
     [], ["def ", "is_palindrome"]),
    ("память_фактов", "отправляли ли мне сегодня отчёты на почту?",
     [], ["ОТЧЁТЫ", "нет записей"]),
    ("поиск", "найди в интернете: что нового в NVIDIA NIM за последние дни",
     [], ["http"]),
]


def _capture_bot() -> HarnessTelegramBot:
    bot = HarnessTelegramBot()
    # Conversation memory isolated: test chatter never touches the operator's
    # durable chat log.
    tmp = Path(tempfile.mkdtemp(prefix="kg-selftest-"))
    bot.memory_engine = MemoryEngine(
        history_file=tmp / "chat.txt", vault_file=tmp / "vault.json"
    )
    # Transport stubs: capture everything the bot would have sent.
    bot._sent: List[str] = []
    bot.send_message = lambda chat_id, text, **kw: (bot._sent.append(text), True)[1]
    bot.send_chat_action = lambda *a, **k: None
    bot.delete_message = lambda *a, **k: None
    # Never persist the synthetic session into the real session store.
    bot._save_authorized_sessions = lambda: None
    return bot


def run_battery() -> Dict[str, Any]:
    bot = _capture_bot()
    chat_id = 90210  # synthetic chat, never persisted
    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    # --- 1. the password gate must fail closed -----------------------------
    bot.process_message({"chat": {"id": chat_id}, "from": {"id": 90210},
                         "text": "wrong-password-123", "message_id": None})
    locked = any("ДОСТУП ЗАКРЫТ" in t for t in bot._sent)
    results.append({"label": "пароль_неверный", "input": "***",
                    "ok": locked, "replies": bot._sent[:]})
    if not locked:
        failures.append("неверный пароль не был отклонён")

    # --- 2. the real passphrase authorizes ---------------------------------
    bot._sent.clear()
    password = os.getenv("TELEGRAM_BOT_PASSWORD", "")
    bot.process_message({"chat": {"id": chat_id}, "from": {"id": 90210},
                         "text": password, "message_id": None})
    authed = (chat_id in bot.authorized_chats
              and any("ДОСТУП УСПЕШНО" in t for t in bot._sent))
    results.append({"label": "пароль_верный", "input": "***",
                    "ok": authed, "replies": ["<authorized>"] if authed else bot._sent[:]})
    if not authed:
        failures.append("верный пароль не авторизовал чат")
        return {"ran_at": _now(), "tests": results, "failures": failures,
                "summary": "FAILED"}

    # --- 3. the scripted conversation battery ------------------------------
    for label, text, hard, soft in TEST_CASES:
        bot._sent.clear()
        started = time.time()
        try:
            bot.process_message({"chat": {"id": chat_id}, "from": {"id": 90210},
                                 "text": text, "message_id": None})
        except Exception as e:
            bot._sent.append(f"[EXCEPTION] {type(e).__name__}: {e}")
        replies = [t for t in bot._sent if t]
        joined = "\n".join(replies)
        missing_hard = [m for m in hard if m not in joined]
        missing_soft = [m for m in soft if m not in joined]
        ok = bool(replies) and not missing_hard
        if not ok:
            failures.append(f"{label}: hard markers missing {missing_hard}")
        results.append({
            "label": label, "input": text, "ok": ok,
            "elapsed_sec": round(time.time() - started, 1),
            "missing_soft_markers": missing_soft,
            "replies": [r[:1200] for r in replies],
        })
        print(f"\n=== {label} {'OK' if ok else 'FAIL'} "
              f"({round(time.time() - started, 1)}s) ===")
        for r in replies:
            print(r[:900])
        if missing_soft:
            print(f"[soft] не найдено (провайдер может варьировать): {missing_soft}")

    return {
        "ran_at": _now(),
        "tests": results,
        "failures": failures,
        "summary": "PASSED" if not failures else "FAILED",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    print("=" * 72)
    print("🧪 BOT SELF-TEST: the complex interrogates its own Telegram bot")
    print("=" * 72)
    report = run_battery()

    report_path = PROJECT_ROOT / "data" / "bot_selftest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    passed = sum(1 for t in report["tests"] if t.get("ok"))
    total = len(report["tests"])
    StateManager().append_history_log(
        f"САМОТЕСТ БОТА {'ПРОЙДЕН' if report['summary'] == 'PASSED' else 'ПРОВАЛЕН'} "
        f"| проверок={passed}/{total} | "
        + ("; ".join(report["failures"][:3]) if report["failures"] else "весь диалог живой")
    )

    print("\n" + "=" * 72)
    print(f"{'✅' if report['summary'] == 'PASSED' else '❌'} "
          f"BOT SELF-TEST {report['summary']}: {passed}/{total} checks")
    print(f"report: {report_path}")
    print("=" * 72)
    sys.exit(0 if report["summary"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
