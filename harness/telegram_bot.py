#!/usr/bin/env python3
"""
Interactive Telegram Bot Bridge with Password Protection.
Enables secure human-in-the-loop interaction with active duty models:
- MiniMax M3.1 Flash (live streaming + unpruned Keenable search)
- DeepSeek-R1-32B / Sovereign CPU Engine
- System Vitals, Fleet Telemetry, and Storage Monitoring

Features:
- Access-Control: Secret Passphrase / PIN protection
- Persistent Session Memory (.telegram_sessions.json)
- Automatic Password Deletion from chat history
- Zero external library overhead: implemented via pure HTTP requests.
"""

import os
import sys
import json
import time
import html
import signal
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

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
from harness.sovereign_cpu_engine import SovereignCPUEngine
from harness.memory_engine import MemoryEngine
from harness.tools import AgentToolbox
from harness import runtime_settings
from harness import operational_context

class HarnessTelegramBot:
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        self.keenable_key = os.getenv("KEENABLE_API_KEY", "")
        
        # Password protection configuration.
        # The passphrase lives only in the GitHub secret TELEGRAM_BOT_PASSWORD
        # (and optionally .env for local runs). This repository is public, so
        # no default is embedded in the source: an unset passphrase means the
        # bot rejects every connection.
        self.bot_password = os.getenv("TELEGRAM_BOT_PASSWORD", "").strip()
        if not self.bot_password:
            print("[!] WARNING: TELEGRAM_BOT_PASSWORD is empty! Bot will reject all connections.")
        # Sessions live in data/ so they can be persisted between GitHub Actions
        # runs (the runner filesystem is wiped on every job restart).
        self.sessions_file = PROJECT_ROOT / "data" / ".bot_sessions.json"
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        self.authorized_chats: Set[int] = self._load_authorized_sessions()

        self.state_mgr = StateManager()
        self.disk_guard = DiskGuard()
        self.executor = MiniMaxExecutor(nvidia_api_key=self.nvidia_key, keenable_api_key=self.keenable_key)
        self.arbiter_engine = SovereignCPUEngine()
        self.memory_engine = MemoryEngine(nvidia_api_key=self.nvidia_key)
        # Event memory: the operational log (nim_history.log) indexed with the
        # same protocol as the conversation, so the bot can recall what the
        # complex actually did (dispatches, key checks, fleet changes) by
        # meaning and not only by reading the log tail.
        self.event_memory = MemoryEngine(
            history_file=PROJECT_ROOT / "nim_history.log",
            vault_file=PROJECT_ROOT / "data" / ".event_vault.json",
            nvidia_api_key=self.nvidia_key,
            log_format="events",
        )
        self.user_states: Dict[int, str] = {} # chat_id -> awaiting_input_type
        self._state_lock = threading.Lock()
        self._stop = False
        # Model calls (MiniMax streaming can take minutes) run in a worker pool so
        # the polling loop keeps receiving and answering other messages meanwhile.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tg-handler")

        # Graceful shutdown so sessions/state survive a GitHub Actions job stop.
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except (ValueError, AttributeError, OSError):
            pass  # not in main thread / unsupported platform

    def log_startup_config(self) -> None:
        """Prints which credentials are configured so startup failures are obvious."""
        print("[Telegram Bot] Startup configuration:")
        print(f"    TELEGRAM_BOT_TOKEN: {'set' if self.is_configured() else 'MISSING (bot cannot start)'}")
        print(f"    NVIDIA_API_KEY:     {'set' if self.nvidia_key else 'MISSING (model calls will fail)'}")
        print(f"    KEENABLE_API_KEY:   {'set' if self.keenable_key else 'not set (public endpoint / DuckDuckGo fallback)'}")
        print(f"    TELEGRAM_BOT_PASSWORD: {'set' if self.bot_password else 'MISSING (bot will reject all connections)'}")

    def is_configured(self) -> bool:
        return bool(self.bot_token and len(self.bot_token) > 10)

    def _load_authorized_sessions(self) -> Set[int]:
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("authorized_chats", []))
            except Exception:
                pass
        return set()

    def _save_authorized_sessions(self) -> None:
        try:
            with open(self.sessions_file, "w", encoding="utf-8") as f:
                json.dump({"authorized_chats": list(self.authorized_chats)}, f, indent=2)
        except Exception:
            pass

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

    def delete_message(self, chat_id: int, message_id: int) -> None:
        """Deletes sensitive message (e.g. password) from chat history."""
        try:
            requests.post(f"{self.base_url}/deleteMessage", json={"chat_id": chat_id, "message_id": message_id}, timeout=5)
        except Exception:
            pass

    def get_main_keyboard(self) -> Dict[str, Any]:
        """Builds interactive reply keyboard with quick-access buttons."""
        return {
            "keyboard": [
                [{"text": "📡 Статус системы"}, {"text": "🚀 Флот моделей"}],
                [{"text": "👑 Запрос Арбитру"}, {"text": "🌐 Поиск Keenable"}],
                [{"text": "📧 Проверка почты"}, {"text": "📝 Сменить email"}],
                [{"text": "🔒 Выход (/logout)"}]
            ],
            "resize_keyboard": True,
            "is_persistent": True
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

        markup = reply_markup if reply_markup is not None else self.get_main_keyboard()
        chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
        
        for idx, chunk in enumerate(chunks):
            # Only attach reply markup to the last chunk
            current_markup = markup if idx == len(chunks) - 1 else None
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "reply_markup": current_markup,
                "disable_web_page_preview": True,
                "link_preview_options": {"is_disabled": True}
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

    def handle_status_command(self, chat_id: int) -> None:
        """Handles /status command."""
        self.send_chat_action(chat_id, "typing")
        state = self.state_mgr.load_local_state()
        disk = self.disk_guard.inspect_disk_usage()

        checker = APIHealthChecker(api_key=self.nvidia_key) if self.nvidia_key else None
        key_res = checker.check_key_validity() if checker else {"valid": False, "message": "Key not set"}

        # The last recorded report dispatch — the authoritative answer to
        # "were the reports sent?", taken from the operational log itself.
        dispatch_line = "нет записей об отправке"
        try:
            from harness.operational_context import _dispatch_events, _tail, _KEY_FILES
            records = _dispatch_events(_tail(_KEY_FILES["history"], 1000))
            if records:
                dispatch_line = records[-1]
        except Exception:
            pass

        report = (
            "📡 <b>СИСТЕМНЫЙ СТАТУС АВТОНОМНОГО КОМПЛЕКСА</b>\n\n"
            f"👑 <b>Верховный арбитр (CPU):</b> <code>{state.get('local_arbiter', 'Qwen/Qwen2.5-1.5B-Instruct')}</code>\n"
            "⚡ <b>Тактический исполнитель:</b> <code>MiniMax M3.1 Flash (256k окно / 128k ответ)</code>\n"
            "🧠 <b>Вечная память (RAG):</b> <code>NVIDIA Nemotron 3.5 Embed 8B (4096d)</code>\n"
            f"🔑 <b>NVIDIA NIM API:</b> {key_res.get('message')} (HTTP {key_res.get('status_code', 0)})\n"
            f"💾 <b>Дисковое пространство:</b> {disk.get('free_gb')} ГБ свободно ({disk.get('percent_used')}% занято)\n"
            f"📬 <b>Последняя отправка отчётов:</b> <code>{html.escape(dispatch_line[:220])}</code>\n"
            f"📨 <b>Получатель отчётов:</b> <code>{html.escape(runtime_settings.get_recipient_email('не задан'))}</code>\n"
            f"🕒 <b>Обновлено:</b> {state.get('last_updated', 'N/A')}\n\n"
            "🛡️ <i>Автономный суверенный комплекс полностью отвязан от очередей сторонних сервисов.</i>"
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
            resp.append(f"<b>{idx}. {html.escape(title)}</b>\n🔗 {url}\n{html.escape(snip)}\n")

        self.send_message(chat_id, "\n".join(resp), parse_mode="HTML")

    def handle_arbiter_query(self, chat_id: int, prompt: str) -> None:
        """
        Bidirectional Arbitration Channel:
        Queries Supreme On-Duty Arbiter Qwen2.5-1.5B (SovereignCPUEngine),
        persists verdict into eternal memory, and synthesizes tactical guidance via MiniMax.
        """
        self.send_chat_action(chat_id, "typing")
        self.send_message(chat_id, "👑 <i>Запрашиваю вердикт Верховного дежурного арбитра Qwen2.5-1.5B (CPU)...</i>", parse_mode="HTML")

        # 1. Append user prompt to eternal memory
        self.memory_engine.append_message("user", prompt, {"chat_id": chat_id, "intent": "arbiter_query"})

        # 2. Query Local Arbiter
        context = {
            "query": prompt,
            "minimax_window": "256,000 tokens total (Hard ceiling: 262,144 tokens)",
            "minimax_output": "Up to 128,000 tokens (131,072 tokens)",
            "pacing": "<= 30 RPM (min 2.0s slot intervals)",
            "supreme_arbiter": "Qwen/Qwen2.5-1.5B-Instruct on CPU Steady Active Duty",
            "eternal_memory": "NVIDIA Nemotron 3.5 Embed 8B active with data/telegram_history.txt"
        }
        arb_res = self.arbiter_engine.execute_sovereign_mission(
            task_description=f"Issue supreme arbiter ruling for commander query: {prompt}",
            context=context
        )
        arbiter_output = (
            arb_res.get("output")
            or arb_res.get("report", {}).get("decision")
            or "Директива подтверждена верховным арбитром: окно MiniMax составляет 256,000 токенов (ввод/контекст), генерация до 128,000 токенов, лимит <= 30 RPM."
        )

        # 3. Log Arbiter verdict to eternal memory
        self.memory_engine.append_message("arbiter", f"RULING: {arbiter_output}")

        # 4. Formulate consolidated tactical response via MiniMax with memory envelope
        memory_ctx = self.memory_engine.build_memory_envelope(prompt)
        combined_prompt = (
            f"Вопрос Командира: {prompt}\n\n"
            f"Официальный вердикт Верховного Арбитра Qwen2.5-1.5B:\n{arbiter_output}\n\n"
            f"Сформируй четкий, исчерпывающий и точный доклад для Командира в Telegram."
        )
        minimax_res = self.executor.execute_task(
            task_prompt=combined_prompt,
            context=memory_ctx,
            enable_search=False,
            max_tokens=16384,
            temperature=0.6
        )
        content = minimax_res.get("content", arbiter_output)
        self.memory_engine.append_message("minimax", content, {"chat_id": chat_id})

        reply = (
            f"👑 <b>Вердикт Верховного Арбитра (Qwen2.5-1.5B CPU):</b>\n"
            f"<blockquote>{html.escape(arbiter_output)}</blockquote>\n\n"
            f"⚡ <b>Доклад тактического дежурного MiniMax M3.1 Flash:</b>\n\n"
            f"{html.escape(content)}"
        )
        self.send_message(chat_id, reply, parse_mode="HTML")

    def handle_minimax_task(self, chat_id: int, prompt: str, enable_search: bool = False) -> None:
        """Delegates tasks to MiniMax M3.1 Flash with eternal memory recall and CoT."""
        # 1. Check if user is asking the Arbiter / Boss / Supervisor
        clean_prompt = prompt.lower().strip()
        arbiter_triggers = ["начальник", "начальнику", "начальника", "арбитр", "арбитру", "арбитра", "шеф", "босс", "/arbiter"]
        if any(t in clean_prompt for t in arbiter_triggers):
            return self.handle_arbiter_query(chat_id, prompt)

        self.send_chat_action(chat_id, "typing")
        self.send_message(chat_id, "🧠 <i>Передаю задачу боевой модели MiniMax M3.1 Flash...</i>", parse_mode="HTML")

        # 2. Append user prompt to eternal memory log
        self.memory_engine.append_message("user", prompt, {"chat_id": chat_id})

        # 3. Web search is STRICTLY DISABLED by default!
        search_prefixes = ["найди в интернете:", "найди в интернете ", "поиск в интернете:", "/search "]
        needs_search = enable_search or any(clean_prompt.startswith(p) for p in search_prefixes)

        # 4. Build semantic memory envelope from Nemotron vault + recent turns
        memory_envelope = self.memory_engine.build_memory_envelope(prompt)

        # 4b. Attach the verified system facts (operational log, active fleet,
        # last diagnostics). Without them the model answers questions about
        # what the complex actually did from the chat log alone, which is how
        # it ended up inventing an operator role that does not exist.
        operational = operational_context.build_operational_context(
            query=prompt,
            event_memory=self.event_memory,
        )

        start_t = time.time()
        res = self.executor.execute_task(
            task_prompt=prompt,
            context=f"{operational}\n\n{memory_envelope}",
            enable_search=needs_search,
            max_tokens=32768,
            temperature=0.6
        )
        elapsed = res.get("elapsed_sec", round(time.time() - start_t, 1))
        content = res.get("content", "Модель не сформировала ответ.")
        citations = res.get("citations", [])

        # 5. Append model output to eternal memory log
        self.memory_engine.append_message("minimax", content, {"chat_id": chat_id})

        citations_text = ""
        if citations and needs_search:
            citations_text = "\n\n🔗 <b>Использованные источники:</b>\n" + "\n".join(f"• {c}" for c in citations[:5])

        reply = f"🤖 <b>Ответ MiniMax M3.1 Flash ({elapsed}с):</b>\n\n{html.escape(content)}{citations_text}"
        self.send_message(chat_id, reply, parse_mode="HTML")

    def handle_email_dispatch(self, chat_id: int) -> None:
        """Triggers manual email dispatch."""
        self.send_chat_action(chat_id, "typing")
        from harness.daily_dispatcher import build_health_report_email
        from harness.email_sender import EmailSender

        mail = build_health_report_email()
        sender = EmailSender()
        recipient = runtime_settings.get_recipient_email()
        res = sender.send_email(
            recipient_email=recipient,
            subject=mail["subject"],
            html_content=mail["html"],
            plain_text_content=mail["plain"]
        )
        if res.get("status") == "DELIVERED":
            self.send_message(chat_id, f"✅ Письмо «Проверка связи» успешно отправлено на <b>{recipient}</b>!", parse_mode="HTML")
        else:
            self.send_message(chat_id, f"[-] Ошибка отправки: {res.get('error') or res.get('message')}")

    def handle_email_change(self, chat_id: int, new_email: str) -> None:
        """Updates the recipient email for daily dispatches."""
        new_email = new_email.strip()
        # Basic email validation
        if "@" not in new_email or "." not in new_email.split("@")[-1]:
            self.send_message(chat_id, "[-] Неверный формат email. Пример: user@example.com")
            return

        try:
            # Persisted outside .env: GitHub secrets are frozen at job start and
            # .env is gitignored, so either would lose the change on the next
            # runner restart. The daemon commits this file back to the repo.
            runtime_settings.set_setting("DISPATCH_RECIPIENT_EMAIL", new_email)

            self.send_message(chat_id, f"✅ Email для отчётов обновлён: <b>{new_email}</b>\n\nСледующая ежедневная рассылка (12:00 НСК) и ручная проверка почты пойдут на этот адрес. Изменение сохранено и переживёт перезапуск.", parse_mode="HTML")
            print(f"[+] Dispatch recipient email changed to: {new_email}")
        except Exception as e:
            self.send_message(chat_id, f"[-] Ошибка сохранения: {e}")

    def process_message(self, message: Dict[str, Any]) -> None:
        """Processes a single incoming message from Telegram with Password Guard."""
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        username = message.get("from", {}).get("username", "")
        message_id = message.get("message_id")
        text = message.get("text", "").strip()

        if not chat_id or not text:
            return

        # 1. Password Verification Layer
        if chat_id not in self.authorized_chats:
            candidate = text.replace("/login ", "", 1).strip() if text.startswith("/login ") else text

            if self.bot_password and candidate == self.bot_password:
                with self._state_lock:
                    self.authorized_chats.add(chat_id)
                self._save_authorized_sessions()
                if message_id:
                    self.delete_message(chat_id, message_id)

                welcome = (
                    "🔓 <b>ДОСТУП УСПЕШНО РАЗРЕШЕН!</b>\n\n"
                    "Пароль подтвержден. Вы авторизованы в командном центре автономного комплекса.\n"
                    "Клавиатура быстрого управления активирована внизу экрана."
                )
                self.send_message(chat_id, welcome, parse_mode="HTML")
                print(f"[+] User {user_id} (@{username}) successfully authenticated chat {chat_id} with password.")
                return
            else:
                lock_msg = (
                    "🔒 <b>ДОСТУП ЗАКРЫТ.</b>\n\n"
                    "Этот бот защищен паролем.\n"
                    "Введите секретный пароль для авторизации:"
                )
                self.send_message(chat_id, lock_msg, reply_markup={"remove_keyboard": True}, parse_mode="HTML")
                return

        # 2. Logout Command
        if text in ["/logout", "🔒 Выход (/logout)"]:
            with self._state_lock:
                self.authorized_chats.discard(chat_id)
            self._save_authorized_sessions()
            self.send_message(
                chat_id,
                "🔒 <b>Вы вышли из системы.</b>\nЧат заблокирован. Для повторного входа введите пароль:",
                reply_markup={"remove_keyboard": True},
                parse_mode="HTML"
            )
            print(f"[*] Chat {chat_id} logged out.")
            return

        # 3. Pending Interaction States
        with self._state_lock:
            state = self.user_states.pop(chat_id, None)
        if state:
            if state == "awaiting_search_query":
                self.handle_keenable_search(chat_id, text)
                return
            elif state == "awaiting_arbiter_query":
                self.handle_arbiter_query(chat_id, text)
                return
            elif state == "awaiting_task_prompt":
                self.handle_minimax_task(chat_id, text)
                return
            elif state == "awaiting_email_change":
                self.handle_email_change(chat_id, text)
                return

        # 4. Standard Commands & Buttons
        if text in ["/start", "/help", "ℹ️ Справка"]:
            recipient = runtime_settings.get_recipient_email("configured-recipient")
            welcome = (
                "👋 <b>Добро пожаловать в Интерактивный Штаб Автономного Комплекса!</b>\n\n"
                "Вы можете управлять дежурными моделями, запрашивать исследования и мониторить систему прямо отсюда.\n\n"
                "📌 <b>Быстрые команды:</b>\n"
                "• <b>📡 Статус системы</b> — проверка ключей, диска, режимов и квот.\n"
                "• <b>👑 Запрос Арбитру</b> — прямое обращение к Верховному арбитру (Qwen2.5-1.5B CPU).\n"
                "• <b>🚀 Флот моделей</b> — список 5 утвержденных моделей NIM.\n"
                "• <b>🌐 Поиск Keenable</b> — живой поиск информации без обрезки.\n"
                f"• <b>📧 Проверка почты</b> — отправка отчета здоровья на {recipient}.\n"
                "• <b>📝 Сменить email</b> — изменить получателя ежедневных отчётов.\n"
                "• <b>🔒 Выход (/logout)</b> — заблокировать доступ.\n\n"
                "💡 <i>Или просто напишите любой вопрос или задачу сообщением — дежурные модели сразу приступят к решению с вечной памятью!</i>"
            )
            self.send_message(chat_id, welcome, parse_mode="HTML")
        elif text in ["/status", "📡 Статус системы"]:
            self.handle_status_command(chat_id)
        elif text in ["/arbiter", "👑 Запрос Арбитру"]:
            with self._state_lock:
                self.user_states[chat_id] = "awaiting_arbiter_query"
            self.send_message(chat_id, "👑 <b>Введите вопрос или директиву для Верховного дежурного арбитра (Qwen2.5-1.5B CPU):</b>", parse_mode="HTML")
        elif text in ["/fleet", "🚀 Флот моделей"]:
            self.handle_fleet_command(chat_id)
        elif text in ["/email", "📧 Проверка почты"]:
            self.handle_email_dispatch(chat_id)
        elif text in ["/setemail", "📝 Сменить email"]:
            self.user_states[chat_id] = "awaiting_email_change"
            current = runtime_settings.get_recipient_email("не задан")
            self.send_message(chat_id, f"📝 <b>Текущий email для отчётов:</b> <code>{current}</code>\n\nВведите новый email-адрес получателя:", parse_mode="HTML")
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

    def _handle_signal(self, signum, frame) -> None:
        """Graceful shutdown: GitHub Actions sends SIGTERM when the job ends."""
        print(f"\n[*] Signal {signum} received — shutting down gracefully...")
        self._stop = True
        self._save_authorized_sessions()

    def start_polling(self) -> None:
        """Starts long-polling loop to listen for user commands."""
        self.log_startup_config()
        me = self.get_me()
        if not me.get("ok"):
            print(f"[-] Telegram Bot Error: Cannot connect to Telegram. {me.get('description')}")
            print("[*] Please configure TELEGRAM_BOT_TOKEN in .env or environment.")
            return

        bot_username = me.get("result", {}).get("username", "Unknown")
        print("=" * 72)
        print(f"🤖 TELEGRAM BOT ACTIVATED: @{bot_username}")
        print(f"[*] Password protection: ACTIVE (Passphrase set)")

        # The runner filesystem is wiped between jobs, so the semantic indexes
        # are re-derived from the durable logs before serving messages.
        try:
            self.memory_engine.ensure_vault()
        except Exception as e:
            print(f"[!] Memory index warmup skipped: {e}")
        try:
            self.event_memory.ensure_vault(max_entries=300)
        except Exception as e:
            print(f"[!] Event index warmup skipped: {e}")

        print(f"[*] Polling Telegram updates... Press Ctrl+C to terminate.")
        print("=" * 72)

        offset = None
        while not self._stop:
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
                            # Handle in a worker pool: long model generations must not
                            # freeze polling (the bot stays responsive to everyone else).
                            self._pool.submit(self._safe_process, update["message"])
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

    def _safe_process(self, message: Dict[str, Any]) -> None:
        """Runs in worker threads; keeps handler exceptions from killing the pool."""
        try:
            self.process_message(message)
        except Exception as e:
            print(f"[!] Message processing error: {e}")

if __name__ == "__main__":
    bot = HarnessTelegramBot()
    bot.start_polling()
