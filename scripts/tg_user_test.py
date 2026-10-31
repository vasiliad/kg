#!/usr/bin/env python3
"""
Real-user Telegram test client: talks to the complex's bot AS A USER via
MTProto (Telethon), so the operator's assistant can run true end-to-end
chat tests — messages, replies, keyboard buttons — and then revoke the
session.

Credentials never touch the repository:
  TG_API_ID / TG_API_HASH  - environment variables only
  session file             - data/.tg_user_session.session (gitignored)

Commands:
  send-code --phone +7...        request the Telegram login code
  login    --phone +7... --code 12345
  chat     --text "..."          send a message to the bot
  read     [--n 5]               print the bot's latest replies
  press    --button "📡 Статус системы"
  logout                         revoke the session (access closed)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SESSION_PATH = PROJECT_ROOT / "data" / ".tg_user_session"
CODE_HASH_PATH = PROJECT_ROOT / "data" / ".tg_code_hash"
BOT_USERNAME = os.getenv("TEST_BOT_USERNAME", "immortal911bot")


def _make_client():
    from telethon import TelegramClient
    api_id = int(os.getenv("TG_API_ID", "0") or 0)
    api_hash = os.getenv("TG_API_HASH", "")
    if not api_id or not api_hash:
        print("[-] TG_API_ID / TG_API_HASH не заданы в окружении.")
        sys.exit(1)
    return TelegramClient(str(SESSION_PATH), api_id, api_hash)


async def cmd_send_code(args) -> None:
    client = _make_client()
    await client.connect()
    sent = await client.send_code_request(args.phone)
    CODE_HASH_PATH.write_text(sent.phone_code_hash, encoding="utf-8")
    print(f"[+] Код отправлен в Telegram аккаунта {args.phone}. ")
    print("[+] Скопируй код и выполни: login --phone ... --code <КОД>")
    await client.disconnect()


async def cmd_login(args) -> None:
    client = _make_client()
    await client.connect()
    phone_code_hash = CODE_HASH_PATH.read_text(encoding="utf-8").strip()
    await client.sign_in(
        phone=args.phone, code=args.code, phone_code_hash=phone_code_hash
    )
    me = await client.get_me()
    print(f"[+] Вход выполнен: {me.first_name} (@{me.username}).")
    print("[+] Сессия сохранена в (gitignored):", SESSION_PATH.name)
    await client.disconnect()


async def cmd_chat(args) -> None:
    async with _make_client() as client:
        msg = await client.send_message(BOT_USERNAME, args.text)
        print(f"[+] Отправлено боту @{BOT_USERNAME} (id={msg.id}): {args.text[:80]}")


async def cmd_read(args) -> None:
    async with _make_client() as client:
        dialog = await client.get_entity(BOT_USERNAME)
        msgs = await client.get_messages(dialog, limit=args.n)
        for m in reversed(msgs):
            who = "БОТ" if m.out is False else "Я"
            text = (m.message or "").replace("\n", " ")[:200]
            print(f"[{m.date:%H:%M:%S}] {who}: {text}")


async def cmd_press(args) -> None:
    async with _make_client() as client:
        dialog = await client.get_entity(BOT_USERNAME)
        msgs = await client.get_messages(dialog, limit=10)
        for m in msgs:
            if m.reply_markup and getattr(m.reply_markup, "rows", None):
                await m.click(text=args.button)
                print(f"[+] Нажата кнопка: {args.button}")
                return
        print(f"[-] Кнопка '{args.button}' не найдена в последних сообщениях.")


async def cmd_logout() -> None:
    async with _make_client() as client:
        await client.log_out()
    for p in (SESSION_PATH.with_suffix(".session"), SESSION_PATH.parent / (SESSION_PATH.name + ".session")):
        if p.exists():
            p.unlink()
    if CODE_HASH_PATH.exists():
        CODE_HASH_PATH.unlink()
    print("[+] Сессия отозвана, файл удалён. Доступ закрыт.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-user bot test client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("send-code"); p.add_argument("--phone", required=True)
    p = sub.add_parser("login"); p.add_argument("--phone", required=True); p.add_argument("--code", required=True)
    p = sub.add_parser("chat"); p.add_argument("--text", required=True)
    p = sub.add_parser("read"); p.add_argument("--n", type=int, default=5)
    p = sub.add_parser("press"); p.add_argument("--button", required=True)
    sub.add_parser("logout")

    args = parser.parse_args()
    handlers = {
        "send-code": cmd_send_code, "login": cmd_login, "chat": cmd_chat,
        "read": cmd_read, "press": cmd_press, "logout": cmd_logout,
    }
    asyncio.run(handlers[args.cmd](args) if args.cmd != "logout" else cmd_logout())


if __name__ == "__main__":
    main()
