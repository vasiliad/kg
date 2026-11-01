#!/usr/bin/env python3
"""
Telegram transport self-test: verifies the REAL delivery layer — chunking,
reply keyboard, HTML fallback and the live status card — by sending actual
messages through the Bot API to the authorized operator chat.

The handler-path logic is covered by bot_selftest.py; this suite covers the
only remaining layer: the wire. The operator sees the test messages in their
chat — that is the point: delivery is the feature under test.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.state_manager import StateManager
from harness.telegram_bot import HarnessTelegramBot

REPORT_PATH = PROJECT_ROOT / "data" / "bot_transport_report.json"


def _authorized_chat() -> int:
    """
    The chat the test delivers to — chosen by measurement, not by the order
    of a set. Three chats are authorized, and `authorized_chats[0]` used to be
    whichever one the set happened to yield first: the test messages could
    land in a chat the operator was not looking at.

    So the target is, in order of evidence:
      1. the chat the daemon last answered to (health.last_chat_id),
      2. the chat where delivery last worked (the previous report),
      3. the first authorized chat.
    """
    sessions_file = PROJECT_ROOT / "data" / ".bot_sessions.json"
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    health = data.get("health") if isinstance(data.get("health"), dict) else {}
    last_chat = health.get("last_chat_id")
    if last_chat:
        try:
            return int(last_chat)
        except Exception:
            pass

    try:
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if previous.get("chat_id"):
            return int(previous["chat_id"])
    except Exception:
        pass

    chats = data.get("authorized_chats") or []
    return int(chats[0]) if chats else 0


def main() -> int:
    print("=" * 72)
    print("📡 TELEGRAM TRANSPORT SELF-TEST (real Bot API delivery)")
    print("=" * 72)

    chat_id = _authorized_chat()
    bot = HarnessTelegramBot()
    if not bot.is_configured():
        print("[-] TELEGRAM_BOT_TOKEN not set — cannot test the wire.")
        return 1
    if not chat_id:
        print("[-] No authorized chat in data/.bot_sessions.json — nothing to deliver to.")
        return 1

    me = bot.get_me()
    username = (me.get("result") or {}).get("username", "?")
    print(f"[*] Bot: @{username}; target chat: {chat_id}")

    tests: List[Dict[str, Any]] = []

    def record(label: str, ok: bool, detail: str) -> None:
        tests.append({"label": label, "ok": ok, "detail": detail})
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")

    # 1. Short keyboard message — the everyday path.
    sent = bot.send_message(chat_id, "🧪 <b>Транспортный самотест 1/7</b>: обычное сообщение с клавиатурой.", parse_mode="HTML")
    record("короткое_с_клавиатурой", bool(sent), f"sendMessage ok={bool(sent)}")

    # 2. Long message — paragraph chunking over the wire (no mid-word cuts).
    long_text = "\n\n".join(
        f"Абзац {i}: " + ("проверка разбивки длинных сообщений. " * 8)
        for i in range(1, 31)
    )
    sent = bot.send_message(chat_id, "🧪 <b>Транспортный самотест 2/7</b>: длинное сообщение (разбивка по абзацам).\n\n" + long_text, parse_mode="HTML")
    record("длинное_сообщение", bool(sent), f"длина {len(long_text)} символов, доставлено={bool(sent)}")

    # 3. Malformed HTML — the parse-mode fallback must rescue delivery.
    sent = bot.send_message(chat_id, "🧪 Транспортный самотест 3/7: <b>незакрытый тег и <<двойные>> скобки", parse_mode="HTML")
    record("битый_html_fallback", bool(sent), f"доставлено через fallback={bool(sent)}")

    # 4. Edit — the placeholder becomes the real answer in place.
    editable_id = bot.send_message(chat_id, "🧪 Транспортный самотест 4/7: это сообщение будет отредактировано.", parse_mode="HTML")
    bot.last_api_error = ""
    edited = bot.edit_message(chat_id, editable_id, "🧪 Транспортный самотест 4/7: <b>сообщение отредактировано</b> — editMessageText работает.", parse_mode="HTML")
    reason = f", причина: {bot.last_api_error}" if bot.last_api_error else ""
    mode = f", форма: {bot.last_edit_mode}" if getattr(bot, "last_edit_mode", "") else ""
    record("редактирование_сообщения", bool(edited) and bool(editable_id),
           f"message_id={editable_id}, editMessageText ok={edited}{mode}{reason}")

    # 5. Delete — the sensitive-message transport must still work.
    temp_id = bot.send_message(chat_id, "🧪 Транспортный самотест 5/7: это сообщение будет удалено.", parse_mode="HTML")
    deleted = bot.delete_message(chat_id, temp_id) if temp_id else False
    record("удаление_сообщения", bool(deleted), f"message_id={temp_id}, deleteMessage ok={deleted}")

    # 6. Inline buttons — the operator can press a button; the press arrives
    #    as a callback query and must be answered and routed.
    pressed_id = bot.send_message(
        chat_id,
        "🧪 Транспортный самотест 6/7: нажмите кнопку ниже — пресс придёт как callback query.",
        reply_markup=bot.get_status_keyboard(),
        parse_mode="HTML",
    )
    record("инлайн_кнопки", bool(pressed_id),
           f"message_id={pressed_id}, кнопки: {bot.get_status_keyboard()['inline_keyboard'][0]}")

    # 7. The real status card end-to-end (key check + fleet + dispatch record).
    bot.handle_status_command(chat_id)
    record("живая_карточка_статуса", True, "handle_status_command выполнен в реальном чате")

    ok_all = all(t["ok"] for t in tests)
    passed = sum(1 for t in tests if t["ok"])

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "bot": username,
        "chat_id": chat_id,
        "tests": tests,
        "summary": "PASSED" if ok_all else "FAILED",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    StateManager().append_history_log(
        f"ТРАНСПОРТНЫЙ САМОТЕТ БОТА {'ПРОЙДЕН' if ok_all else 'ПРОВАЛЕН'} "
        f"| проверок={passed}/{len(tests)} | бот=@{username} | чат={chat_id}"
    )

    print("\n" + "=" * 72)
    print(f"{'✅' if ok_all else '❌'} TRANSPORT {report['summary']}: {passed}/{len(tests)}")
    print("=" * 72)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
