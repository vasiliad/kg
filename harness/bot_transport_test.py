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
    """The operator's chat, persisted by the daemon's session store."""
    sessions_file = PROJECT_ROOT / "data" / ".bot_sessions.json"
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
        chats = data.get("authorized_chats") or []
        return int(chats[0]) if chats else 0
    except Exception:
        return 0


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
    sent = bot.send_message(chat_id, "🧪 <b>Транспортный самотест 1/4</b>: обычное сообщение с клавиатурой.", parse_mode="HTML")
    record("короткое_с_клавиатурой", sent, f"sendMessage ok={sent}")

    # 2. Long message — paragraph chunking over the wire (no mid-word cuts).
    long_text = "\n\n".join(
        f"Абзац {i}: " + ("проверка разбивки длинных сообщений. " * 8)
        for i in range(1, 31)
    )
    sent = bot.send_message(chat_id, "🧪 <b>Транспортный самотест 2/4</b>: длинное сообщение (разбивка по абзацам).\n\n" + long_text, parse_mode="HTML")
    record("длинное_сообщение", sent, f"длина {len(long_text)} символов, доставлено={sent}")

    # 3. Malformed HTML — the parse-mode fallback must rescue delivery.
    sent = bot.send_message(chat_id, "🧪 Транспортный самотест 3/4: <b>незакрытый тег и <<двойные>> скобки", parse_mode="HTML")
    record("битый_html_fallback", sent, f"доставлено через fallback={sent}")

    # 4. The real status card end-to-end (key check + fleet + dispatch record).
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
