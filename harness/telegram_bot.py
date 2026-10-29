#!/usr/bin/env python3
"""
Interactive Telegram Bot Bridge for Autonomous Harness.
Enables real-time human-in-the-loop interaction with active duty models:
- MiniMax M3.1 Flash (live streaming + unpruned Keenable search)
- DeepSeek-R1-32B / Sovereign CPU Engine
- System Vitals, Fleet Telemetry, and Storage Monitoring

Zero external library overhead: implemented via pure HTTP requests.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

from harness.disk_guard import DiskGuard
from harness.state_manager import StateManager
from harness.api_healthcheck import APIHealthChecker
from harness.minimax_executor import MiniMaxExecutor
from harness.tools import AgentToolbox

class HarnessTelegramBot:
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        self.keenable_key = os.getenv("KEENABLE_API_KEY", "")
        
        # Whitelisted Telegram user IDs
        raw_allowed = os.getenv("ALLOWED_TELEGRAM_USER_IDS", "")
        self.allowed_users = [int(u.strip()) for u in raw_allowed.split(",") if u.strip().isdigit()]
        
        self.state_mgr = StateManager()
        self.disk_guard = DiskGuard()
        self.executor = MiniMaxExecutor(nvidia_api_key=self.nvidia_key, keenable_api_key=self.keenable_key)
        self.user_states: Dict[int, str] = {} # chat_id -> awaiting_input_type

    def is_configured(self) -> bool:
        return bool(self.bot_token and len(self.bot_token) > 10)

    def get_me(self) -> Dict[str, Any]:
        """Validates bot token against Telegram API."""
        if not self.is_configured():
            return {"ok": False, "description": "Token not set"}
        try:
            res = requests.get(f"{self.base_url}/getMe", timeout=10)
            return res.json()
        except Exception as e:
            return {"ok": False, "description": str(e)}

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Sends status like 'typing...' to the user."""
        try:
            requests.post(f"{self.base_url}/sendChatAction", json={"chat_id": chat_id, "action": action}, timeout=5)
        except Exception:
            pass

    def get_main_keyboard(self) -> Dict[str, Any]:
        """Builds interactive reply keyboard with quick-access buttons."""
        return {
            "keyboard": [
                [{"text": "📡 Статус системы"}, {"text": "🚀 Флот моделей"}],
                [{"text": "🌐 Поиск Keenable"}, {"text": "📧 Проверка почты"}],
                [{"text": "ℹ️ Справка"}]
            ],
            "resize_keyboard": true,
            "is_persistent": true
        }

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: Optional[str] = None
    ) -> bool:
        """Sends a message, automatically chunking if length exceeds 4000 characters."""
        if not self.is_configured():
            print(f"[Telegram Bot] Token missing. Message to {chat_id}: {text[:100]}...")
            return False

        markup = reply_markup or self.get_main_keyboard()
        chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
        
        for idx, chunk in enumerate(chunks):
            # Only attach reply markup to the last chunk
            current_markup = markup if idx == len(chunks) - 1 else None
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "reply_markup": current_markup
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            try:
                res = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
                if not res.ok and parse_mode:
                    # Fallback to plain text if markdown formatting failed
                    payload.pop("parse_mode", None)
                    requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
            except Exception as e:
                print(f"[!] Error sending message to {chat_id}: {e}")
                return False
        return True

    def check_authorization(self, user_id: int, username: str = "") -> bool:
        """Checks if the user is authorized to send tasks."""
        if not self.allowed_users:
            # If no users configured yet, automatically whitelist first user and notify
            print(f"[*] No whitelist set. Auto-authorizing first user: {user_id} (@{username})")
            self.allowed_users.append(user_id)
            return True
        return user_id in self.allowed_users

    def handle_status_command(self, chat_id: int) -> None:
        """Handles /status command."""
        self.send_chat_action(chat_id, "typing")
        state = self.state_mgr.load_local_state()
        disk = self.disk_guard.inspect_disk_usage()

        checker = APIHealthChecker(api_key=self.nvidia_key) if self.nvidia_key else None
        key_res = checker.check_key_validity() if checker else {"valid": False, "message": "Key not set"}

        report = (
            "📡 <b>СИСТЕМНЫЙ СТАТУС АВТОНОМНОГО КОМПЛЕКСА</b>\n\n"
            f"👑 <b>Верховный арбитр:</b> <code>{state.get('local_arbiter', 'Unknown')}</code>\n"
            f"⚙️ <b>Режим:</b> {'🚨 Суверенный CPU (Zero-API)' if state.get('sovereign_zero_api_mode') else ('⚡ Экстренный CPU Fallback' if state.get('emergency_mode') else '💎 Штатный TPU v3-8 / NIM')}\n"
            f"🔑 <b>NVIDIA NIM API:</b> {key_res.get('message')} (HTTP {key_res.get('status_code', 0)})\n"
            f"💾 <b>Дисковое пространство:</b> {disk.get('free_gb')} ГБ свободно ({disk.get('percent_used')}% занято)\n"
            f"🕒 <b>Последнее обновление:</b> {state.get('last_updated', 'N/A')}\n\n"
            "🛡️ <i>Все подсистемы самовосстановления и лимитов активны.</i>"
        )
        self.send_message(chat_id, report, parse_mode="HTML")

    def handle_fleet_command(self, chat_id: int) -> None:
        """Handles /fleet command."""
        self.send_chat_action(chat_id, "typing")
        state = self.state_mgr.load_local_state()
        fleet = state.get("active_fleet", [])

        lines = ["🚀 <b>АКТИВНЫЙ БОЕВОЙ ФЛОТ МОДЕЛЕЙ:</b>\n"]
        for idx, m in enumerate(fleet[:5], 1):
            m_id = m.get("id", "Unknown") if isinstance(m, dict) else str(m)
            tier = m.get("tier", "Standard") if isinstance(m, dict) else ""
            score = m.get("quality_score", "N/A") if isinstance(m, dict) else ""
            status = m.get("status", "ACTIVE") if isinstance(m, dict) else ""
            lines.append(f"<b>{idx}. {m_id}</b>\n   • Категория: <code>{tier}</code>\n   • Рейтинг качества: <b>{score}</b> | Статус: {status}\n")

        self.send_message(chat_id, "\n".join(lines), parse_mode="HTML")

    def handle_keenable_search(self, chat_id: int, query: str) -> None:
        """Handles /search command."""
        self.send_chat_action(chat_id, "typing")
        self.send_message(chat_id, f"🔍 <i>Выполняю немодифицированный поиск через Keenable:</i>\n<code>{query}</code>", parse_mode="HTML")
        
        results = self.executor.search_web(query, num_results=3)
        if not results:
            self.send_message(chat_id, "[-] По вашему запросу ничего не найдено.")
            return

        resp = [f"🌐 <b>Результаты поиска Keenable ({len(results)} источника):</b>\n"]
        for idx, r in enumerate(results, 1):
            title = r.get("title", "Источник")
            url = r.get("url", "")
            snip = r.get("snippet", "").strip()[:500]
            resp.append(f"<b>{idx}. {title}</b>\n🔗 {url}\n{snip}\n")

        self.send_message(chat_id, "\n".join(resp), parse_mode="HTML")

    def handle_minimax_task(self, chat_id: int, prompt: str) -> None:
        """Delegates task to MiniMax M3.1 Flash with CoT and Keenable."""
        self.send_chat_action(chat_id, "typing")
        self.send_message(chat_id, "🧠 <i>Передаю задачу боевой модели MiniMax M3.1 Flash...</i>", parse_mode="HTML")

        # Periodically refresh typing action during generation
        start_t = time.time()
        res = self.executor.execute_task(
            task_prompt=prompt,
            enable_search=True,
            max_tokens=32768,
            temperature=0.6
        )
        elapsed = res.get("elapsed_sec", round(time.time() - start_t, 1))
        content = res.get("content", "Модель не сформировала ответ.")
        citations = res.get("citations", [])

        citations_text = ""
        if citations:
            citations_text = "\n\n🔗 <b>Использованные источники:</b>\n" + "\n".join(f"• {c}" for c in citations[:5])

        reply = f"🤖 <b>Ответ MiniMax M3.1 Flash ({elapsed}с):</b>\n\n{content}{citations_text}"
        self.send_message(chat_id, reply)

    def handle_email_dispatch(self, chat_id: int) -> None:
        """Triggers manual email dispatch."""
        self.send_chat_action(chat_id, "typing")
        from harness.daily_dispatcher import build_health_report_email
        from harness.email_sender import EmailSender

        mail = build_health_report_email()
        sender = EmailSender()
        res = sender.send_email(
            recipient_email="ivanvasiliadi@gmail.com",
            subject=mail["subject"],
            html_content=mail["html"],
            plain_text_content=mail["plain"]
        )
        if res.get("status") == "DELIVERED":
            self.send_message(chat_id, "✅ Письмо «Проверка связи» успешно отправлено на <b>ivanvasiliadi@gmail.com</b>!", parse_mode="HTML")
        else:
            self.send_message(chat_id, f"[-] Ошибка отправки: {res.get('error') or res.get('message')}")

    def process_message(self, message: Dict[str, Any]) -> None:
        """Processes a single incoming message from Telegram."""
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        username = message.get("from", {}).get("username", "")
        text = message.get("text", "").strip()

        if not chat_id or not text:
            return

        if not self.check_authorization(user_id, username):
            self.send_message(
                chat_id,
                f"⛔ <b>Доступ ограничен.</b>\nВаш User ID: <code>{user_id}</code>.\nДобавьте его в <code>ALLOWED_TELEGRAM_USER_IDS</code>.",
                parse_mode="HTML"
            )
            return

        # Check pending state
        if chat_id in self.user_states:
            state = self.user_states.pop(chat_id)
            if state == "awaiting_search_query":
                self.handle_keenable_search(chat_id, text)
                return
            elif state == "awaiting_task_prompt":
                self.handle_minimax_task(chat_id, text)
                return

        # Handle Commands & Button Clicks
        if text in ["/start", "/help", "ℹ️ Справка"]:
            welcome = (
                "👋 <b>Добро пожаловать в Интерактивный Штаб Автономного Комплекса!</b>\n\n"
                "Вы можете управлять дежурными моделями, запрашивать исследования и мониторить систему прямо отсюда.\n\n"
                "📌 <b>Быстрые команды:</b>\n"
                "• <b>📡 Статус системы</b> — проверка ключей, диска, режимов и квот.\n"
                "• <b>🚀 Флот моделей</b> — список 5 утвержденных моделей NIM.\n"
                "• <b>🌐 Поиск Keenable</b> — живой поиск информации без обрезки.\n"
                "• <b>📧 Проверка почты</b> — отправка отчета здоровья на ivanvasiliadi@gmail.com.\n\n"
                "💡 <i>Или просто напишите любой вопрос или задачу сообщением — дежурная модель MiniMax M3.1 Flash сразу приступит к решению!</i>"
            )
            self.send_message(chat_id, welcome, parse_mode="HTML")
        elif text in ["/status", "📡 Статус системы"]:
            self.handle_status_command(chat_id)
        elif text in ["/fleet", "🚀 Флот моделей"]:
            self.handle_fleet_command(chat_id)
        elif text in ["/email", "📧 Проверка почты"]:
            self.handle_email_dispatch(chat_id)
        elif text in ["/search", "🌐 Поиск Keenable"]:
            self.user_states[chat_id] = "awaiting_search_query"
            self.send_message(chat_id, "🔍 Введите поисковый запрос для поиска через Keenable:")
        elif text.startswith("/search "):
            query = text.replace("/search ", "", 1).strip()
            self.handle_keenable_search(chat_id, query)
        elif text.startswith("/task "):
            prompt = text.replace("/task ", "", 1).strip()
            self.handle_minimax_task(chat_id, prompt)
        else:
            # Default: treat any regular message as a task for MiniMax
            self.handle_minimax_task(chat_id, text)

    def start_polling(self) -> None:
        """Starts long-polling loop to listen for user commands."""
        me = self.get_me()
        if not me.get("ok"):
            print(f"[-] Telegram Bot Error: Cannot connect to Telegram. {me.get('description')}")
            print("[*] Please configure TELEGRAM_BOT_TOKEN in .env or environment.")
            return

        bot_username = me.get("result", {}).get("username", "Unknown")
        print("=" * 72)
        print(f"🤖 TELEGRAM BOT ACTIVATED: @{bot_username}")
        print(f"[*] Polling Telegram updates... Press Ctrl+C to terminate.")
        print("=" * 72)

        offset = None
        while True:
            try:
                params = {"timeout": 25}
                if offset:
                    params["offset"] = offset

                res = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        if "message" in update:
                            self.process_message(update["message"])
                elif res.status_code == 409:
                    print("[!] Conflict: Another bot instance is polling. Waiting 10s...")
                    time.sleep(10)
                else:
                    time.sleep(2)
            except requests.exceptions.Timeout:
                continue
            except KeyboardInterrupt:
                print("\n[*] Polling stopped by user.")
                break
            except Exception as e:
                print(f"[!] Polling network exception: {e}. Retrying in 5s...")
                time.sleep(5)

if __name__ == "__main__":
    bot = HarnessTelegramBot()
    bot.start_polling()
