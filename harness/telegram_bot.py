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
import re
import signal
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlsplit

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

from harness.atomic_io import write_json_atomic
from harness.disk_guard import DiskGuard
from harness.state_manager import StateManager
from harness.api_healthcheck import APIHealthChecker
from harness.minimax_executor import MiniMaxExecutor
from harness.sovereign_cpu_engine import SovereignCPUEngine
from harness.memory_engine import MemoryEngine
from harness.tools import AgentToolbox
from harness import runtime_settings
from harness import redaction
from harness import operational_context
from harness import tool_call


class _NoMarkup:
    """
    Sentinel: "send this message without any reply markup".

    The plain `None` means "attach the persistent keyboard" (the default for
    every operator-facing message). A transient placeholder needs the opposite:
    Telegram refuses to edit a message that carries a reply keyboard
    ("Bad Request: message can't be edited" — measured in the cloud), so the
    placeholder must go out bare and the answer may then replace it in place.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<NO_MARKUP>"


NO_MARKUP = _NoMarkup()


def _parse_chat_ids(raw: str) -> List[int]:
    """
    The operator's chat ids, taken from the `TELEGRAM_OPERATOR_CHAT_ID`
    secret — the ONLY place the real id may live.

    Measured 2026-11-03: the authorized chats and `health.last_chat_id` were
    committed to the PUBLIC repository, and the id was printed into the
    public Actions log. The repository now keeps a fingerprint; the secret
    keeps the address.

    The value is a list because the operator may write from more than one
    account; a comma, a space or a semicolon separates the ids.
    """
    out: List[int] = []
    for part in re.split(r"[\s,;]+", (raw or "").strip()):
        if not part:
            continue
        try:
            chat_id = int(part)
        except ValueError:
            continue
        if chat_id not in out:
            out.append(chat_id)
    return out


def _now_iso() -> str:
    """UTC timestamp, second precision — the unit the health block speaks in."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _operator_safe_text(text: str) -> str:
    """
    The answer a human may read.

    Measured on 2026-11-01: the duty model answered «Найди на озон» with a raw
    JSON tool call and the operator received "only curly braces". The executor
    now runs such a call and asks again, but the bot is the last door to the
    operator: whatever comes out of a model is checked here. A machine-shaped
    answer becomes a sentence about what the model wanted to do.
    """
    if not text:
        return ""
    call = tool_call.parse_tool_call(text)
    if not call:
        return text

    wanted = tool_call.tool_query(call) or tool_call.tool_url(call) or "—"
    warning = (
        "⚠️ Модель вернула вызов инструмента вместо ответа "
        f"({tool_call.describe_call(call)}).\n"
        f"Запрос, который она хотела выполнить: {wanted}"
    )
    prose = tool_call.strip_tool_calls(text)
    if prose:
        # The answer carried prose AND a call: the prose is the answer.
        return f"{prose}\n\n{warning}"
    return warning


def _short_link(url: str) -> str:
    """`https://ozon.ru/item/1234567890/` -> `ozon.ru/item/1234567890`."""
    try:
        parsed = urlsplit(url)
        host = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip("/")
        if not path:
            return host or url[:60]
        return f"{host}{path[-48:]}"
    except Exception:
        return url[:60]


def _sources_block(citations: List[Any]) -> str:
    """
    The links the duty model actually used — clickable, and honest.

    `res["citations"]` used to be dropped on the floor: the operator got a
    claim about the live web with no way to check it. Only http(s) links are
    rendered; anything else is not a source.
    """
    seen: List[str] = []
    for raw in citations or []:
        url = str(raw or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        if url not in seen:
            seen.append(url)
    if not seen:
        return ""
    lines = ["🔗 <b>Источники</b>"]
    for url in seen[:10]:
        lines.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(_short_link(url))}</a>')
    if len(seen) > 10:
        lines.append(f"…ещё {len(seen) - 10}")
    return "\n".join(lines)


def health_line(health: Optional[Dict[str, Any]]) -> str:
    """
    The bot's own liveness as one card line. Empty when there is nothing
    measured yet — the card must never invent an activity.
    """
    if not isinstance(health, dict) or not health:
        return ""
    parts: List[str] = []
    last_send = health.get("last_send_at")
    last_poll = health.get("last_poll_at")
    if last_send:
        parts.append(f"последний ответ {last_send}")
    elif last_poll:
        parts.append(f"последний опрос {last_poll}")
    else:
        parts.append("активности ещё не было")
    sent = health.get("messages_sent")
    if isinstance(sent, int):
        parts.append(f"ответов {sent}")
    errors = health.get("errors")
    if isinstance(errors, int) and errors:
        parts.append(f"ошибок {errors}")
    return f"🫀 <b>Активность бота:</b> {html.escape('; '.join(parts))}\n"


def fleet_drift_line(state: Dict[str, Any]) -> str:
    """
    The measured fleet drift as one card line. Empty when nothing was
    measured yet — the card must never invent a number.
    """
    drift = (state or {}).get("fleet_drift")
    if not isinstance(drift, dict) or not drift.get("status"):
        return ""
    try:
        from harness.fleet_drift import describe
        return f"📉 <b>Дрейф флота:</b> {html.escape(describe(drift))}\n"
    except Exception:
        return ""


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
        # The operator's real chat ids live in the SECRET, never in the
        # repository: the tracked session file keeps only fingerprints
        # («чат#1a2b3c4d»), the secret keeps the address. A chat learned
        # from the passphrase is authorized for this run and remembered as a
        # fingerprint; the operator's own chat is authorized by the secret
        # and therefore survives every restart.
        self.operator_chat_ids: List[int] = _parse_chat_ids(
            os.getenv("TELEGRAM_OPERATOR_CHAT_ID", ""))
        self.authorized_chats: Set[int] = (
            self._load_authorized_sessions() | set(self.operator_chat_ids))
        # Telegram's update cursor. It MUST survive a restart: without it the
        # bot re-reads every unconfirmed update of the last 24 hours on each
        # cold start and re-answers old questions — duplicate replies, wasted
        # NIM quota, and the operator's real message buried under the replay.
        self.last_update_id: int = self._load_last_update_id()
        # Updates dispatched to a worker but not finished yet, plus the highest
        # update id ever seen. The cursor may only pass an update that is
        # really done — and update ids are arbitrary large numbers, not a
        # sequence starting at 1.
        self._pending_update_ids: Set[int] = set()
        self._dispatched_max: int = self.last_update_id
        # The bot's own health: what it last polled, what it last answered,
        # what it last failed on. Without it "the bot does not answer" has
        # two possible causes — an absent bot and a broken one — and from
        # the outside they look exactly the same.
        self.health: Dict[str, Any] = self._load_health()
        # The chat the control copy goes to: the operator's own chat from the
        # secret. Without it the digest would have no addressee after a
        # restart — the tracked file cannot hold one.
        if not self.health.get("last_chat_id") and self.operator_chat_ids:
            self.health["last_chat_id"] = self.operator_chat_ids[0]
            self.health["last_chat_tag"] = redaction.chat_tag(self.operator_chat_ids[0])
        self.health["runs"] = int(self.health.get("runs") or 0) + 1
        self.health["started_at"] = _now_iso()
        self._last_health_save: float = 0.0
        # The last Telegram refusal, verbatim. A failed call that says nothing
        # is a call that cannot be diagnosed.
        self.last_api_error: str = ""
        # Which form of the edit payload Telegram accepted ("markup", "plain",
        # "bare") — a silent downgrade must be visible.
        self.last_edit_mode: str = ""
        # How many chunks the last send produced and how many were actually
        # delivered — a long answer that lost a chunk must be visible, not
        # silent.
        self.last_message_chunks: int = 0
        self.last_message_chunks_sent: int = 0

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
        # The id of the last message the bot sent — the handle for editing
        # a "thinking" placeholder into the real answer.
        self.last_message_id: Optional[int] = None
        # Re-entrant: saving the session file takes the same lock that
        # advances the update cursor, and both live in one code path.
        self._state_lock = threading.RLock()
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

    def _load_sessions(self) -> Dict[str, Any]:
        """Reads the persisted session state; a broken file yields empty state."""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _load_authorized_sessions(self) -> Set[int]:
        """
        The chats authorized by a PREVIOUS run.

        The tracked file holds fingerprints («чат#1a2b3c4d»), not ids — so it
        can only confirm that a chat WAS authorized, never which one. The ids
        come from the secret. A legacy file that still holds raw ids is read
        (it is a real authorization), and the next write replaces them with
        fingerprints.
        """
        chats: Set[int] = set()
        try:
            for chat in self._load_sessions().get("authorized_chats", []) or []:
                if isinstance(chat, bool):
                    continue
                if isinstance(chat, int):
                    chats.add(int(chat))
                elif isinstance(chat, str) and chat.strip().lstrip("-").isdigit():
                    chats.add(int(chat.strip()))
        except Exception:
            return set()
        return chats

    def _load_health(self) -> Dict[str, Any]:
        """
        The health block of the previous runs. A broken file yields an empty
        block — the bot must still start and must not invent history.
        """
        raw = self._load_sessions().get("health")
        raw = raw if isinstance(raw, dict) else {}
        health = {
            "runs": 0,
            "started_at": None,
            "last_poll_at": None,
            "last_update_at": None,
            "last_send_at": None,
            "last_chat_id": None,
            "last_chat_tag": None,
            "updates_seen": 0,
            "messages_sent": 0,
            "errors": 0,
            "last_error": None,
        }
        for key in health:
            if key in raw:
                health[key] = raw[key]
        for key in ("runs", "updates_seen", "messages_sent", "errors"):
            try:
                health[key] = int(health[key] or 0)
            except Exception:
                health[key] = 0
        # `last_chat_id` is an ADDRESS. The tracked file keeps the
        # fingerprint («чат#1a2b3c4d»); the secret keeps the id. A legacy
        # file that still holds a raw id is read, and the next write
        # replaces it with a fingerprint.
        raw_chat = health.get("last_chat_id")
        if isinstance(raw_chat, str) and not raw_chat.strip().lstrip("-").isdigit():
            health["last_chat_tag"] = raw_chat.strip()
            health["last_chat_id"] = None
        elif raw_chat is not None:
            try:
                chat_id = int(raw_chat)
                health["last_chat_id"] = chat_id
                health["last_chat_tag"] = redaction.chat_tag(chat_id)
            except (TypeError, ValueError):
                health["last_chat_id"] = None
                health["last_chat_tag"] = None
        return health

    def _save_health(self, force: bool = False) -> None:
        """
        Persists the health block. Throttled, so a 25-second poll does not
        write the file on every tick — but an update, an answer or an error
        is always written through.
        """
        now = time.time()
        if not force and now - self._last_health_save < 120:
            return
        self._last_health_save = now
        self._save_authorized_sessions()

    def _record_poll(self) -> None:
        """The bot reached Telegram — this is the liveness signal."""
        with self._state_lock:
            self.health["last_poll_at"] = _now_iso()
        self._save_health()

    def _record_update(self) -> None:
        """An update arrived from Telegram."""
        with self._state_lock:
            self.health["last_update_at"] = _now_iso()
            self.health["updates_seen"] = int(self.health.get("updates_seen") or 0) + 1
        self._save_health(force=True)

    def _record_send(self, chat_id: int = 0) -> None:
        """A message went out to the operator."""
        with self._state_lock:
            self.health["last_send_at"] = _now_iso()
            self.health["messages_sent"] = int(self.health.get("messages_sent") or 0) + 1
            if chat_id:
                # Which chat the bot actually answered — measured, not guessed.
                # The id stays in memory; the tracked file keeps the tag.
                self.health["last_chat_id"] = int(chat_id)
                self.health["last_chat_tag"] = redaction.chat_tag(int(chat_id))
        self._save_health(force=True)

    def _operator_chat(self) -> int:
        """
        The chat to answer to, as a real id — 0 when there is none.

        The id may come from the secret or from this run's memory; the
        tracked file holds only a fingerprint, which is NOT an address and
        must never be cast to an int.
        """
        for candidate in (self.health.get("last_chat_id"),
                          self.operator_chat_ids[0] if self.operator_chat_ids else None):
            if candidate is None:
                continue
            try:
                chat_id = int(candidate)
            except (TypeError, ValueError):
                continue
            if chat_id:
                return chat_id
        return 0

    def _record_error(self, error: str) -> None:
        """Something broke — the reason must outlive the window."""
        with self._state_lock:
            self.health["errors"] = int(self.health.get("errors") or 0) + 1
            self.health["last_error"] = str(error)[:300]
        self._save_health(force=True)

    def _load_last_update_id(self) -> int:
        try:
            return int(self._load_sessions().get("last_update_id") or 0)
        except Exception:
            return 0

    def _merge_sessions_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges this run's state with what is on disk.

        Two processes write this file: the daemon (which owns the cursor) and
        the self-test (which only adds health numbers). A writer that started
        earlier must never roll the cursor back — that would make Telegram
        replay messages the operator has already seen, which is exactly the
        duplicate-answer bug the cursor was introduced to kill. So the cursor
        only ever moves forward, authorizations are a union, and the health
        counters keep the largest value each side has seen.
        """
        disk = self._load_sessions()
        merged = dict(payload)

        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except Exception:
                return 0

        merged["last_update_id"] = max(
            _int(disk.get("last_update_id")), _int(payload.get("last_update_id"))
        )

        chats: List[str] = []
        for chat in list(disk.get("authorized_chats") or []) + list(payload.get("authorized_chats") or []):
            # The tracked file holds FINGERPRINTS: «чат#1a2b3c4d» proves that
            # a chat was authorized, not which one. The ids live in the
            # secret and in memory only. The union matters: an authorization
            # must never be dropped by another writer.
            tag = chat if redaction.is_chat_tag(chat) else redaction.chat_tag(chat)
            if tag not in chats:
                chats.append(tag)
        merged["authorized_chats"] = chats

        disk_health = disk.get("health") if isinstance(disk.get("health"), dict) else {}
        own_health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
        health = dict(disk_health)
        health.update(own_health)
        for key in ("runs", "updates_seen", "messages_sent", "errors"):
            health[key] = max(_int(disk_health.get(key)), _int(own_health.get(key)))
        for key in ("started_at", "last_poll_at", "last_update_at", "last_send_at"):
            # The stamps are UTC ISO-8601 — they compare correctly as text.
            stamps = [str(v) for v in (disk_health.get(key), own_health.get(key)) if v]
            health[key] = max(stamps) if stamps else None
        # The chat the bot last answered to: keep whichever side knows it.
        if not own_health.get("last_chat_id"):
            health["last_chat_id"] = disk_health.get("last_chat_id")
        if not own_health.get("last_chat_tag"):
            health["last_chat_tag"] = disk_health.get("last_chat_tag")
        health.setdefault("last_error", None)
        if not own_health.get("last_error"):
            health["last_error"] = disk_health.get("last_error")
        merged["health"] = health
        return merged

    def _save_authorized_sessions(self) -> None:
        """Persists authorized chats, the Telegram update cursor and the health block together."""
        try:
            with self._state_lock:
                payload = {
                    "authorized_chats": list(self.authorized_chats),
                    "last_update_id": self.last_update_id,
                    # The health block is a bonus, never a precondition: the
                    # cursor must be persisted even if it is missing.
                    "health": dict(getattr(self, "health", {}) or {}),
                }
                payload = self._merge_sessions_payload(payload)
            # The session file is committed to a public repository: no address
            # may travel in it (see harness/redaction.py) — neither the
            # e-mail nor the Telegram chat.
            payload = redaction.mask_sensitive_deep(payload)
            # Atomic: THIS file is the bot's memory — the Telegram cursor,
            # the authorized chats, the health. A half-written file would
            # reset the cursor and replay old updates.
            write_json_atomic(self.sessions_file, payload)
        except Exception:
            pass

    def _mark_update_dispatched(self, update_id: int) -> None:
        """Records that an update is being handled right now."""
        if update_id <= 0:
            return
        with self._state_lock:
            self._pending_update_ids.add(update_id)
            if update_id > self._dispatched_max:
                self._dispatched_max = update_id

    def _mark_update_handled(self, update_id: int) -> None:
        """
        Advances the persisted cursor over the updates that are really done.
        The cursor may only pass an update that finished; a crash must never
        swallow the operator's message. Telegram's own confirmation is what
        actually deletes an update, so a slightly conservative cursor costs one
        extra scan, never a lost message.
        """
        if update_id <= 0:
            return
        with self._state_lock:
            self._pending_update_ids.discard(update_id)
            if self._pending_update_ids:
                # The oldest unfinished update holds the cursor back.
                cursor = min(self._pending_update_ids) - 1
            else:
                cursor = max(self.last_update_id, self._dispatched_max)
            if cursor > self.last_update_id:
                self.last_update_id = cursor
                self._save_authorized_sessions()

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

    def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Deletes sensitive message (e.g. password) from chat history."""
        if not self.is_configured() or not message_id:
            return False
        try:
            res = requests.post(
                f"{self.base_url}/deleteMessage",
                json={"chat_id": chat_id, "message_id": message_id},
                timeout=5,
            )
            return bool(res.ok)
        except Exception:
            return False

    def get_main_keyboard(self) -> Dict[str, Any]:
        """Builds interactive reply keyboard with quick-access buttons."""
        return {
            "keyboard": [
                [{"text": "📡 Статус системы"}, {"text": "🧑‍💼 Состав команды"}],
                [{"text": "🔄 Обновить флот"}, {"text": "👑 Запрос Арбитру"}],
                [{"text": "🌐 Поиск Keenable"}, {"text": "📧 Проверка почты"}],
                [{"text": "📝 Сменить email"}, {"text": "🔒 Выход (/logout)"}]
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
    ) -> Optional[int]:
        """
        Sends a message, automatically chunking if length exceeds 4000
        characters. Returns the id of the LAST chunk (or None when nothing
        was delivered) so a caller can edit that message later instead of
        sending another one.
        """
        if not self.is_configured():
            print(f"[Telegram Bot] Token missing. Message to {redaction.chat_tag(chat_id)}: {text[:100]}...")
            return None

        markup = reply_markup if reply_markup is not None else self.get_main_keyboard()
        if markup is NO_MARKUP:
            markup = None
        # Split on paragraph boundaries, never mid-word: a message cut like
        # "…ко | нфигурацию развёртывания" reads as corruption to the user.
        chunks = []
        current = ""
        for para in text.split("\n\n"):
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= 3900:
                current = candidate
                continue
            if current:
                chunks.append(current)
            while len(para) > 3900:
                cut = para.rfind("\n", 2400, 3900)
                cut = cut if cut != -1 else 3900
                chunks.append(para[:cut])
                para = para[cut:].lstrip("\n")
            current = para
        if current:
            chunks.append(current)
        chunks = chunks or [text[:3900]]

        last_id: Optional[int] = None
        sent_chunks = 0
        for idx, chunk in enumerate(chunks):
            # Only attach reply markup to the last chunk
            current_markup = markup if idx == len(chunks) - 1 else None
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
                "link_preview_options": {"is_disabled": True}
            }
            if current_markup is not None:
                # A null reply_markup is not "no markup" for Telegram — it is
                # an invalid object ("Bad Request: object expected as reply
                # markup"), and the message is refused. The field must be
                # absent, not empty. Measured in the cloud: a long answer
                # silently lost every chunk but the last.
                payload["reply_markup"] = current_markup
            if parse_mode:
                payload["parse_mode"] = parse_mode
            try:
                res = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
                if not res.ok and parse_mode:
                    # Fallback to plain text if markdown formatting failed
                    payload.pop("parse_mode", None)
                    res = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
                if not res.ok:
                    # A silent failure is not a delivery — say why.
                    print(f"[!] sendMessage failed (HTTP {res.status_code}): {self._api_error(res)}")
                if res.ok:
                    sent_chunks += 1
                    try:
                        mid = (res.json().get("result") or {}).get("message_id")
                        if mid:
                            last_id = int(mid)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[!] Error sending message to {redaction.chat_tag(chat_id)}: {e}")
                return None
        self.last_message_id = last_id
        self.last_message_chunks = len(chunks)
        self.last_message_chunks_sent = sent_chunks
        if last_id:
            # The operator actually received something — that is the answer
            # side of the health signal.
            self._record_send(chat_id)
        return last_id

    def get_chat(self, chat_id: int) -> Dict[str, Any]:
        """
        The chat's type and identity ("private", "group", "channel"). The
        transport behaves differently per type, so a delivery problem must be
        read against the kind of chat it happened in.
        """
        if not self.is_configured():
            return {}
        try:
            res = requests.post(f"{self.base_url}/getChat", json={"chat_id": chat_id}, timeout=10)
            if not res.ok:
                return {}
            result = res.json().get("result")
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def _api_error(self, res: Any) -> str:
        """
        The Telegram reason behind a non-OK response. "The edit failed" is not
        a diagnosis — the description in the body is. The last reason is kept
        so a report can carry it.
        """
        description = ""
        try:
            body = res.json()
            if isinstance(body, dict):
                description = str(body.get("description") or body.get("error_code") or "")
        except Exception:
            pass
        if not description:
            try:
                description = str(res.text or "")[:200]
            except Exception:
                description = ""
        self.last_api_error = description
        return description

    def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = None
    ) -> bool:
        """
        Edits a message that was already sent — the transport that turns a
        "the keeper is thinking" placeholder into the real answer without
        adding another message to the operator's chat.

        editMessageText accepts a narrower parameter set than sendMessage:
        `disable_web_page_preview` is a sendMessage field, and on this method
        the request is rejected as it stands (measured in the cloud: HTTP 400,
        the edit never landed). So the payload here is built for the method,
        and the fallbacks strip one layer at a time — markup first, then the
        preview options — because a failed edit must never cost the operator
        the text: the caller falls back to sending it as a new message.
        """
        if not self.is_configured() or not message_id:
            return False
        base = {"chat_id": chat_id, "message_id": message_id, "text": text}
        attempts: List[Dict[str, Any]] = []
        if parse_mode:
            attempts.append({**base, "parse_mode": parse_mode,
                             "link_preview_options": {"is_disabled": True}})
            # Same rescue as sendMessage: broken markup must not lose the text.
            attempts.append({**base, "link_preview_options": {"is_disabled": True}})
        else:
            attempts.append({**base, "link_preview_options": {"is_disabled": True}})
        attempts.append(dict(base))  # the bare minimum — always legal

        for payload in attempts:
            try:
                res = requests.post(f"{self.base_url}/editMessageText", json=payload, timeout=15)
            except Exception as e:
                print(f"[!] Error editing message {message_id} in {redaction.chat_tag(chat_id)}: {e}")
                return False
            if res.ok:
                # Which form Telegram accepted: the markup, the plain text, or
                # the bare request. A silent downgrade to plain text must be
                # visible, not invisible.
                self.last_edit_mode = ("markup" if payload.get("parse_mode")
                                       else ("plain" if payload.get("link_preview_options")
                                             else "bare"))
                return True
            print(f"[!] editMessageText failed (HTTP {res.status_code}): {self._api_error(res)}")
        self.last_edit_mode = ""
        return False

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        """
        Answers a button press. Without this the button keeps spinning and
        the operator concludes that the bot is stuck.
        """
        if not self.is_configured() or not callback_query_id:
            return False
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text[:200]
        try:
            res = requests.post(f"{self.base_url}/answerCallbackQuery", json=payload, timeout=10)
            return bool(res.ok)
        except Exception as e:
            print(f"[!] Error answering callback: {e}")
            return False

    def get_status_keyboard(self) -> Dict[str, Any]:
        """Inline buttons for the status card: a press is a command."""
        return {
            "inline_keyboard": [
                [
                    {"text": "🧑‍💼 Состав команды", "callback_data": "roster"},
                    {"text": "🔄 Обновить флот", "callback_data": "retune"},
                ]
            ]
        }

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
            f"🧑‍💼 <b>Дежурная команда:</b> {len(state.get('active_fleet', []) or [])} моделей, источник: <code>{state.get('fleet_source', 'статический сид')}</code>\n"
            f"{fleet_drift_line(state)}"
            f"{health_line(self.health)}"
            "⚡ <b>Тактический исполнитель:</b> <code>по замеру флота (см. «Состав команды»)</code>\n"
            "🧠 <b>Вечная память (RAG):</b> <code>NVIDIA Nemotron 3.5 Embed 8B (4096d)</code>\n"
            f"🔑 <b>NVIDIA NIM API:</b> {key_res.get('message')} (HTTP {key_res.get('status_code', 0)})\n"
            f"💾 <b>Дисковое пространство:</b> {disk.get('free_gb')} ГБ свободно ({disk.get('percent_used')}% занято)\n"
            f"📬 <b>Последняя отправка отчётов:</b> <code>{html.escape(dispatch_line[:220])}</code>\n"
            f"📨 <b>Получатель отчётов:</b> <code>{html.escape(runtime_settings.get_recipient_email('не задан'))}</code>\n"
            f"🕒 <b>Обновлено:</b> {state.get('last_updated', 'N/A')}\n\n"
            "🛡️ <i>Автономный суверенный комплекс полностью отвязан от очередей сторонних сервисов.</i>"
        )
        self.send_message(chat_id, report, reply_markup=self.get_status_keyboard(), parse_mode="HTML")

    def handle_fleet_command(self, chat_id: int) -> None:
        """Handles the team roster command — who serves and why."""
        self.send_chat_action(chat_id, "typing")
        from harness.duty_team import DutyTeam
        self.send_message(chat_id, DutyTeam().format_roster(), parse_mode="HTML")

    def handle_fleet_refresh(self, chat_id: int) -> None:
        """
        Runs the autonomous fleet tuner in-process: live probes of the NIM
        catalogue, measured scores, parameter sweep, new duty roster. Takes
        several minutes; the handler thread keeps polling responsive.
        """
        self.send_message(
            chat_id,
            "🔄 <i>Запускаю живой замер каталога NIM: пинг кандидатов, объективные "
            "проверки (математика / код / JSON), свип параметров. Это займёт "
            "несколько минут — доложу по готовности.</i>",
            parse_mode="HTML",
        )
        try:
            from harness.fleet_tuner import FleetTuner
            tuner = FleetTuner(api_key=self.nvidia_key)
            report = tuner.run()
            fleet = report.get("fleet") or []
            if fleet:
                lines = ["✅ <b>ФЛОТ ПЕРЕОБРАН ПО ЖИВОМУ ЗАМЕРУ</b>\n"]
                for e in fleet:
                    roles = ", ".join(e.get("roles", [])) or e.get("tier", "")
                    lat = (e.get("measured") or {}).get("latency_sec")
                    lat_txt = f", {lat}с" if lat else ""
                    lines.append(
                        f"• <b>{e['id']}</b> — {roles}\n"
                        f"   балл <b>{e['quality_score']}</b> (замер{lat_txt}), статус: {e.get('status')}"
                    )
                lines.append("\nХранитель теперь распределяет задачи по этому составу.")
                self.send_message(chat_id, "\n".join(lines), parse_mode="HTML")
            else:
                reason = report.get("status", "неизвестно")
                self.send_message(
                    chat_id,
                    f"[-] Замер не дал утверждённого состава (причина: {reason}). "
                    "Предыдущий флот сохранён без изменений — команда продолжает работу.",
                    parse_mode="HTML",
                )
        except Exception as e:
            self.send_message(chat_id, f"[-] Ошибка замера флота: {html.escape(str(e))}\nПредыдущий флот сохранён.")

    def handle_keenable_search(self, chat_id: int, query: str) -> None:
        """Handles /search command."""
        self.send_chat_action(chat_id, "typing")
        self.send_message(chat_id, f"🔍 <i>Ищу в живом вебе:</i>\n<code>{html.escape(query)}</code>", parse_mode="HTML")

        results = self.executor.search_web(query, num_results=3)
        if not results:
            self.send_message(chat_id, "[-] По вашему запросу ничего не найдено.")
            return

        # The tier that actually served the answer — not a hardcoded brand.
        tier = getattr(self.executor, "last_search_tier", "") or "неизвестный тир"
        resp = [f"🌐 <b>Результаты поиска ({len(results)} источника, тир: {html.escape(str(tier))}):</b>\n"]
        for idx, r in enumerate(results, 1):
            title = r.get("title", "Источник")
            url = str(r.get("url", "") or "")
            snip = r.get("snippet", "").strip()[:500]
            link = (
                f'<a href="{html.escape(url, quote=True)}">{html.escape(_short_link(url))}</a>'
                if url.lower().startswith(("http://", "https://")) else ""
            )
            resp.append(f"<b>{idx}. {html.escape(title)}</b>\n{link}\n{html.escape(snip)}\n")

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
        content = _operator_safe_text(minimax_res.get("content", arbiter_output))
        arbiter_output = _operator_safe_text(arbiter_output)
        self.memory_engine.append_message("minimax", content, {"chat_id": chat_id})

        reply = (
            f"👑 <b>Вердикт Верховного Арбитра (Qwen2.5-1.5B CPU):</b>\n"
            f"<blockquote>{html.escape(arbiter_output)}</blockquote>\n\n"
            f"⚡ <b>Доклад тактического дежурного MiniMax M3.1 Flash:</b>\n\n"
            f"{html.escape(content)}"
        )
        self.send_message(chat_id, reply, parse_mode="HTML")

    def handle_minimax_task(self, chat_id: int, prompt: str, enable_search: bool = False) -> None:
        """
        Free-form task handling, the architecture the operator asked for:
        the keeper (local CPU chief) plans the dispatch, two or three duty
        models from the measured fleet answer in parallel, the keeper
        synthesises the verdict. Every reply states WHO answered and on what
        measured evidence.
        """
        # 1. Check if user is asking the Arbiter / Boss / Supervisor
        clean_prompt = prompt.lower().strip()
        arbiter_triggers = ["начальник", "начальнику", "начальника", "арбитр", "арбитру", "арбитра", "шеф", "босс", "/arbiter"]
        if any(t in clean_prompt for t in arbiter_triggers):
            return self.handle_arbiter_query(chat_id, prompt)

        self.send_chat_action(chat_id, "typing")
        placeholder_id = self.send_message(
            chat_id,
            "🧠 <i>Хранитель передаёт задачу дежурной модели смены...</i>",
            parse_mode="HTML",
            # No reply keyboard: Telegram refuses to edit a message that
            # carries one ("Bad Request: message can't be edited" — measured
            # in the cloud), and the whole point of this placeholder is to be
            # replaced by the answer in place.
            reply_markup=NO_MARKUP,
        )

        # 2. Append user prompt to eternal memory log. The previous turn is
        #    read BEFORE the append: «Ищи дальше» and «Найди на озон» refer to
        #    the topic of the turn before them.
        previous_user = self.memory_engine.get_previous_user_turn()
        self.memory_engine.append_message("user", prompt, {"chat_id": chat_id})

        # 3. Web search: only when the question actually needs the live web.
        #    A question about our own complex is answered by our own state.
        from harness import search_query as query_formulation
        search_plan = query_formulation.plan(prompt, previous=previous_user)
        needs_search = enable_search or search_plan["search"]
        if needs_search:
            print(f"[Bot] Поиск: '{search_plan['query']}' ({search_plan['reason']})")

        # 3b. The model gets the inherited topic too — otherwise it repeats
        #     the same question as the planner did.
        effective_prompt = prompt
        if search_plan.get("inherited"):
            effective_prompt = (
                f"{prompt}\n\n"
                f"(Предыдущая тема оператора: {search_plan['inherited']}. "
                "Если реплика сама темы не содержит, продолжай именно её.)"
            )

        # 4. Verified system facts + semantic memory envelope
        memory_envelope = self.memory_engine.build_memory_envelope(effective_prompt)
        operational = operational_context.build_operational_context(
            query=prompt,
            event_memory=self.event_memory,
        )
        task_context = f"{operational}\n\n{memory_envelope}"

        # 5. THE DUTY SHIFT. The operator's rule (2026-11-02): ONE model serves
        #    every task for six hours; the keeper picks it; it changes only if
        #    it stops answering quickly. No distribution across models, no
        #    second opinion — the same voice answers everything.
        from harness.duty_team import DutyTeam, ROLE_TITLES
        from harness import duty_shift

        team = DutyTeam()
        roster = team.load_roster()
        task_type = team.classify(prompt)
        shift, _shift_info = duty_shift.current_duty(roster)

        duty_model_id = str(shift.get("model_id") or "")
        # The holder of the shift, with its measured evidence.
        duty_entry = next(
            (m for m in (roster.get("duty") or []) if str(m.get("id")) == duty_model_id),
            {},
        )
        # The bench is NOT a distribution: it is the replacement used only
        # when the holder stopped answering.
        bench = [
            m for m in (roster.get("duty") or [])
            if str(m.get("id")) != duty_model_id
        ][:2]

        # Answer budget follows the task type: a quick question gets a crisp
        # reply, code/analysis get room to work.
        budget = {"quick": 4096, "general": 8192, "code": 16384, "analysis": 16384}.get(
            task_type, 8192
        )

        answers: List[Dict[str, Any]] = []

        def run_call(idx: int, call: Dict[str, Any]) -> Dict[str, Any]:
            params = call.get("params") or {}
            res = self.executor.execute_task(
                task_prompt=effective_prompt,
                context=task_context,
                enable_search=(needs_search and idx == 0),
                max_tokens=budget if idx == 0 else min(budget, 8192),
                temperature=float(params.get("temperature", 0.6)),
                top_p=float(params.get("top_p", 0.9)),
                model_id=call["id"],
                # The planner's query wins: the executor would otherwise
                # re-formulate the prompt and lose the inherited topic.
                search_query=search_plan["query"] if needs_search else "",
                num_results=5,
            )
            raw_content = res.get("content", "") or ""
            safe_content = _operator_safe_text(raw_content)
            is_call = bool(tool_call.parse_tool_call(raw_content))
            elapsed = res.get("elapsed_sec")
            ok = res.get("status") == "SUCCESS" and bool(safe_content) and not is_call
            # The shift breaks only when the model failed or answered slowly.
            duty_shift.note_outcome(
                call["id"],
                elapsed,
                ok,
                measured_latency=(call.get("measured") or {}).get("latency_sec")
                or call.get("latency_sec"),
            )
            return {
                "model": call["id"],
                "role": call.get("primary_role", ""),
                "score": call.get("score"),
                "verified": call.get("verified"),
                # A raw tool call is NOT an answer: it must never reach the
                # operator as an answer.
                "ok": ok,
                "status": res.get("status"),
                "elapsed": elapsed,
                "content": safe_content,
                "tool_call": is_call,
                "citations": [c for c in (res.get("citations") or []) if str(c or "").strip()],
            }

        if duty_model_id:
            try:
                answers.append(run_call(0, {
                    **duty_entry,
                    "id": duty_model_id,
                    "primary_role": "duty_primary",
                }))
            except Exception as e:
                answers.append({"model": duty_model_id, "ok": False,
                                "status": f"HANDLER_ERROR:{type(e).__name__}", "content": ""})
        else:
            answers.append({"model": "?", "ok": False,
                            "status": "NO_DUTY_MODEL", "content": ""})

        # 5b. The holder did not answer (or answered too slowly): the shift is
        #     broken and the bench replaces it. This is the ONE case where a
        #     second model speaks — a replacement, not a second opinion.
        if not any(a.get("ok") for a in answers) and bench:
            self.send_chat_action(chat_id, "typing")
            replacement = bench[0]
            try:
                answers.append(run_call(1, {
                    **replacement,
                    "id": str(replacement.get("id")),
                    "primary_role": "duty_fallback",
                }))
            except Exception as e:
                answers.append({"model": str(replacement.get("id")), "ok": False,
                                "status": f"HANDLER_ERROR:{type(e).__name__}", "content": ""})

        # 5c. The network tier is GONE (both duty models failed, or there
        #     was no roster at all): the complex TAKES OVER with the local
        #     sovereign engine. Survival protocol, stage 1 — the operator
        #     gets an answer, not a shrug; the reply is labelled with the
        #     engine that produced it.
        if not any(a.get("ok") for a in answers):
            takeover = self.sovereign_takeover_answer(effective_prompt, search_plan["query"])
            if takeover:
                answers.append(takeover)


        # 6. Persist duty answers into eternal memory (the event protocol).
        for a in answers:
            if a.get("ok"):
                self.memory_engine.append_message(
                    "duty", f"[{a['model']}] {a['content'][:800]}", {"chat_id": chat_id}
                )

        # 7. Compose the honest reply: who answered, on what evidence. The
        # keeper's shift line is the only "verdict" — the deterministic CPU
        # kernel's canned English reports proved to be noise here.
        ok_answers = [a for a in answers if a.get("ok")]
        if ok_answers:
            def label(a: Dict[str, Any]) -> str:
                role_txt = ROLE_TITLES.get(a.get("role", ""), "дежурный")
                score_txt = f"{a['score']}" if a.get("score") is not None else "UNRANKED"
                mark = "замер ✅" if a.get("verified") else "сид ⚠️"
                return f"{html.escape(a['model'])} — {role_txt} · балл {score_txt} ({mark}) · {a.get('elapsed', '?')}с"

            primary = ok_answers[0]
            parts = [
                f"🧑‍💼 <i>Хранитель:</i> {html.escape(duty_shift.describe_shift(shift))}",
                f"\n⚡ <b>{label(primary)}</b>\n\n{html.escape(primary['content'])}",
            ]
            sources = _sources_block(primary.get("citations") or [])
            if sources:
                parts.append(f"\n\n{sources}")
            if len(ok_answers) > 1:
                # The holder was replaced because it stopped answering quickly.
                alt = ok_answers[1]
                alt_body = alt["content"][:600] + ("…" if len(alt["content"]) > 600 else "")
                parts.append(
                    f"\n\n🔁 <b>Смена дежурного</b> — {label(alt)}:\n{html.escape(alt_body)}"
                )
            self._deliver_answer(chat_id, placeholder_id, "\n".join(parts), parse_mode="HTML")
        else:
            failed = ", ".join(a.get("model", "?") for a in answers) or "нет измеренных дежурных"
            # A model that answered with a tool call instead of an answer is
            # not a silent failure: the operator must see what it wanted.
            call_notes = [
                a for a in answers
                if a.get("tool_call") and str(a.get("content") or "").strip()
            ]
            if call_notes:
                # The models DID answer — with a machine-shaped answer. Saying
                # "the network models are unavailable" here would be a lie.
                notes = "\n".join(
                    f"• {html.escape(a.get('model', '?'))}: {html.escape(a['content'])}"
                    for a in call_notes
                )
                self._deliver_answer(
                    chat_id,
                    placeholder_id,
                    "[-] <b>Дежурные ответили вызовом инструмента вместо ответа.</b>\n"
                    f"Попытки: {html.escape(failed)}\n\n{notes}\n\n"
                    "Запрос уже сформулирован — повторите его, и я выполню поиск сам.",
                    parse_mode="HTML",
                )
                return
            self._deliver_answer(
                chat_id,
                placeholder_id,
                "[-] <b>Ни одна дежурная модель не ответила.</b>\n"
                f"Попытки: {html.escape(failed)}\n\n"
                "Хранитель в строю, но сетевые модели недоступны. "
                "Нажмите «🔄 Обновить флот» — замер выберет работающие модели из живого каталога.",
                parse_mode="HTML",
            )

    def _deliver_answer(
        self,
        chat_id: int,
        placeholder_id: Optional[int],
        text: str,
        parse_mode: Optional[str] = None
    ) -> None:
        """
        The answer replaces the "thinking" placeholder when the transport can
        edit it; otherwise it goes out as a new message and the placeholder is
        removed. The operator always gets the answer — never a message stuck
        on "the keeper is thinking".
        """
        if isinstance(placeholder_id, int) and placeholder_id > 0 and len(text) <= 4096:
            if self.edit_message(chat_id, placeholder_id, text, parse_mode=parse_mode):
                return
        self.send_message(chat_id, text, parse_mode=parse_mode)
        if placeholder_id:
            # The placeholder is noise once the answer is out.
            self.delete_message(chat_id, placeholder_id)

    def _mail_red_line(self, hours: float = 24.0) -> Dict[str, Any]:
        """
        The red line, read from the event log: errors, deliveries, failures.

        This lives in the BOT on purpose: when the mail channel dies, the
        Telegram pipe must still carry the numbers — an operator who reads
        the control copy must see the day's strain even if no letter
        arrived. Self-contained (no dispatcher import): the bot must not
        load the whole mail pipeline to count log lines.

        And when the log cannot be read, the answer is NOT "zero". A missing
        log read as an empty tally is a calm day that nobody measured — the
        exact lie the red line exists to prevent. So the failure travels
        with the numbers and is shown as such.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        counts: Dict[str, Any] = {"errors": 0, "delivered": 0, "failed": 0}
        try:
            log = PROJECT_ROOT / "nim_history.log"
            for line in (log.read_text(encoding="utf-8", errors="ignore")).splitlines():
                if not line.startswith("["):
                    continue
                try:
                    ts_str = line[1:line.find("]")].replace(" UTC", "")
                    stamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if stamp < cutoff:
                    continue
                msg = line[line.find("]") + 1:].upper()
                if "ОТЧЁТЫ ОТПРАВЛЕНЫ" in msg:
                    counts["delivered"] += 1
                elif "ОТЧЁТЫ НЕ ОТПРАВЛЕНЫ" in msg:
                    counts["failed"] += 1
                if any(kw in msg for kw in ("ERROR", "FAIL", "429", "TIMEOUT", "DANGER", "ALERT")):
                    counts["errors"] += 1
        except Exception as e:
            # The numbers are UNKNOWN, not zero. The caller must say so.
            counts["error"] = f"журнал недоступен ({type(e).__name__})"
            return counts
        return counts

    def maybe_push_daily_digest(self) -> None:
        """
        Once a day, sends the digest artefact to the operator's chat as a
        CONTROL copy.

        The email says «DELIVERED» only to SMTP. This is the second pipe:
        the artefact must also be quotable through the bot, or the
        "что было в утреннем дайджесте?" question answers from nothing. A
        failure here is recorded in the bot's own error counters and never
        interrupts polling.
        """
        chat_id = self._operator_chat()
        if not chat_id:
            return  # nobody to confirm to yet
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.health.get("last_digest_push") == today:
            return  # already pushed today
        artefact = PROJECT_ROOT / "data" / "daily_digest.json"
        try:
            data = json.loads(artefact.read_text(encoding="utf-8")) if artefact.exists() else {}
        except Exception:
            data = {}
        items = data.get("items") or []
        if not items:
            return  # nothing generated yet today
        subject = str(data.get("subject") or "Ежедневный дайджест ИИ")
        lines = [f"📰 <b>Контрольная копия — {html.escape(subject)}</b>", ""]
        for index, item in enumerate(items[:6], 1):
            item = item or {}
            title = str(item.get("title") or "").strip() or "без заголовка"
            url = str(item.get("url") or "").strip()
            mark = "" if str(item.get("detail") or "") == "full" else " (краткое изложение)"
            if url and url.startswith(("http://", "https://")):
                lines.append(f"{index}. <a href=\"{html.escape(url, quote=True)}\">{html.escape(title)}</a>{mark}")
            else:
                lines.append(f"{index}. {html.escape(title)}{mark}")
        full = sum(1 for i in items if str((i or {}).get("detail") or "") == "full")
        lines.append("")
        lines.append(
            html.escape(
                f"Полных описаний: {full} из {len(items)}. "
                "Письмо ушло отдельно; это контроль доставки через Telegram."
            )
        )
        # The red line rides along: if the mail channel is the thing that
        # died, this is where the operator still sees the day's strain.
        strain = self._mail_red_line()
        if strain.get("error"):
            # Silence that was never measured must not wear the mask of a
            # calm day. The numbers are withheld, and the reason is shown.
            lines.append(
                html.escape(
                    f"Красная линия: НЕИЗВЕСТНО — {strain['error']}. "
                    "Числа не показаны: отсутствие показаний не равно «всё спокойно»."
                )
            )
        else:
            lines.append(
                html.escape(
                    f"Красная линия: ошибок за 24ч {strain['errors']} · "
                    f"писем доставлено {strain['delivered']}, не доставлено {strain['failed']}."
                )
            )
        try:
            self.send_message(chat_id, "\n".join(lines), parse_mode="HTML", reply_markup=NO_MARKUP)
            self.health["last_digest_push"] = today
            self._save_health()
            print(f"[*] Control digest copy pushed to Telegram ({today}).")
        except Exception as e:
            self._record_error(f"digest push: {type(e).__name__}: {e}")
            print(f"[!] Control digest push failed: {e}")

    def sovereign_takeover_answer(self, task: str, query: str = "") -> Optional[Dict[str, Any]]:
        """
        Survival protocol, stage 1: the local sovereign engine answers a
        duty task when the network tier is gone.

        The reply is labelled with the engine that produced it: a neural
        GGUF/transformers answer is a real answer; the deterministic kernel
        produces an honest «сеть недоступна» frame instead of pretending.
        """
        try:
            sov = self.arbiter_engine.execute_sovereign_mission(
                task_description=task,
                context={"mode": "duty_takeover", "query": query},
            )
        except Exception as e:
            self._record_error(f"sovereign takeover: {type(e).__name__}: {e}")
            return None

        engine = str(sov.get("engine") or "")
        text = str(sov.get("output") or "").strip()
        if engine == "deterministic_kernel":
            text = (
                "Сетевые модели недоступны, локальная нейросеть не загружена — "
                "отвечает детерминированное ядро Хранителя. "
                "Полный ответ появится, когда сетевой тир восстановится: "
                "супервизор переиграет циклы, дайджест уйдёт с честной пометкой."
            )
        if not text:
            return None
        self._record_error("duty takeover: сеть недоступна, отвечает локальное ядро")
        return {
            "model": f"локальное ядро ({engine})",
            "ok": True,
            "status": "SOVEREIGN_TAKEOVER",
            "content": _operator_safe_text(text),
            "elapsed": sov.get("elapsed_sec"),
            "sovereign": True,
        }

    def handle_code_order(self, chat_id: int, task: str) -> None:
        """
        The operator's ORDER to write code — «напиши программу, которая …».

        The model writes, the forge RUNS it in the sandbox (its own empty
        dir, isolated interpreter, no secrets, hard timeout), and the
        operator gets the MEASURED result: the exit code, the output and
        the error, as they were — never a promise.

        The reserve LOCAL model does the same when the network tier is
        gone: that is the operator's own question («может ли резервная
        локальная модель писать программы»), and it must be answerable
        without the network.

        The self-improvement order («улучши себя») takes the other path:
        the model proposes a PATCH, and only a fully green offline suite
        turns it into a branch.
        """
        from harness import code_forge

        self.send_chat_action(chat_id, "typing")
        apply_proposal = code_forge.is_apply_proposal_order(task)
        placeholder_id = self.send_message(
            chat_id,
            ("⚙️ <i>Кузница: применяю сохранённое улучшение, конституция решает…</i>"
             if apply_proposal else
             "⚙️ <i>Кузница: модель пишет код, песочница запускает…</i>"),
            parse_mode="HTML",
            reply_markup=NO_MARKUP,
        )

        # The operator's «да» to a proposal that ALREADY exists: the stored
        # diff goes through the same constitution. No model is asked to
        # invent a second patch — the proposal is applied as it was written.
        if apply_proposal:
            self._apply_stored_proposal(chat_id, placeholder_id)
            return

        self_improve = code_forge.is_self_improvement_order(task)
        if self_improve:
            # The GROUNDED proposal — the very same path the autopilot uses.
            # One truth about how a proposal is made: the writer names a REAL
            # file of this repository, sees its real content, and the
            # constitution judges the diff.
            self._handle_grounded_proposal(chat_id, placeholder_id, task)
            return

        ask = code_forge.CODE_ORDER_PROMPT + "\n\nЗадача оператора:\n" + task

        # 1. The duty holder writes. The bench replaces it only if it stays
        #    silent — one voice, the keeper's shift.
        writer = self._duty_writer(ask)
        # 2. The network tier is gone: the RESERVE LOCAL model writes.
        local: Dict[str, Any] = {}
        if not writer.get("ok"):
            local = self._local_writer(ask)
            if local.get("ok"):
                writer = local

        if not writer.get("ok"):
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Ни одна модель не написала код.</b>\n"
                f"Сетевая дежурная: {html.escape(str(writer.get('status') or 'нет ответа'))}\n"
                f"Локальная резервная: {html.escape(str((local or {}).get('status') or 'нет ответа'))}\n\n"
                "Без локальной нейросети на этой машине кузница не пишет код: "
                "детерминированное ядро Хранителя не выдумывает программы.",
                parse_mode="HTML",
            )
            return

        # 3. The artifact, as written.
        content = str(writer.get("content") or "")

        code = code_forge.extract_code(content)
        if not code:
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Модель ответила без кода.</b>\n"
                f"Автор: {html.escape(str(writer.get('model') or '?'))}\n\n"
                "Песочница запускает только код: без него результата нет.",
                parse_mode="HTML",
            )
            return

        # 4. The sandbox MEASURES.
        run = code_forge.run_script(code, timeout_sec=code_forge.FORGE_TIMEOUT_SEC)
        self._deliver_code_result(chat_id, placeholder_id, writer, code, run)

    def _duty_writer(self, ask: str) -> Dict[str, Any]:
        """The holder of the duty shift writes the code."""
        from harness import forge_writers

        return forge_writers.duty_writer(ask, self.executor)

    def _local_writer(self, ask: str) -> Dict[str, Any]:
        """
        The RESERVE LOCAL model writes the code when the network is gone.
        The deterministic kernel cannot write programs — and says so.
        """
        from harness import forge_writers

        result = forge_writers.local_writer(ask, self.arbiter_engine)
        if str(result.get("status") or "").startswith("LOCAL_ENGINE_ERROR"):
            self._record_error(f"local writer: {result.get('status')}")
        return result

    def _deliver_code_result(
        self,
        chat_id: int,
        placeholder_id: Optional[int],
        writer: Dict[str, Any],
        code: str,
        run: Dict[str, Any],
    ) -> None:
        """The measured run, as it was — plus who wrote the code."""
        from harness import duty_shift
        from harness import code_forge

        score_txt = f"балл {writer['score']}" if writer.get("score") is not None else "балл не измерен"
        mark = "замер ✅" if writer.get("verified") else "сид ⚠️"
        shift_line = ""
        if writer.get("shift"):
            try:
                shift_line = f"🧑‍💼 <i>Хранитель:</i> {html.escape(duty_shift.describe_shift(writer['shift']))}\n"
            except Exception:
                shift_line = ""

        shown = code if len(code) <= 1800 else code[:1800] + "\n… (код обрезан для показа)"
        exit_txt = "таймаут" if run.get("timed_out") else str(run.get("exit_code"))
        verdict = "✅" if run.get("ok") else "❌"
        out = (run.get("stdout") or "").strip() or "(пусто)"
        err = (run.get("stderr") or "").strip()
        if len(out) > 1500:
            out = out[:1500] + "\n… (вывод обрезан)"
        if len(err) > 800:
            err = err[:800] + "\n… (ошибка обрезана)"

        parts = [
            f"⚙️ <b>Кузница — код написан и выполнен</b> {verdict}",
            shift_line,
            f"✍️ <b>Автор:</b> {html.escape(str(writer.get('model') or '?'))} — "
            f"{score_txt} ({mark}) · {writer.get('elapsed', '?')}с",
            "",
            "<b>Код:</b>",
            f"<pre>{html.escape(shown)}</pre>",
            "",
            f"🧪 <b>Песочница:</b> exit={html.escape(exit_txt)} · "
            f"{run.get('elapsed_sec', '?')}с · изолированный каталог, без секретов, "
            f"таймаут {code_forge.FORGE_TIMEOUT_SEC:.0f}с",
            "",
            "<b>Вывод:</b>",
            f"<pre>{html.escape(out)}</pre>",
        ]
        if err:
            parts += ["", "<b>Ошибка:</b>", f"<pre>{html.escape(err)}</pre>"]
        if not run.get("ok"):
            parts += ["", "<i>Код не прошёл — результат выше измерен, а не обещан.</i>"]
        self._deliver_answer(chat_id, placeholder_id, "\n".join(p for p in parts if p is not None),
                             parse_mode="HTML")

    def _handle_grounded_proposal(
        self,
        chat_id: int,
        placeholder_id: Optional[int],
        task: str,
    ) -> None:
        """
        The operator's order to improve the complex: the GROUNDED proposal.

        The writer names a REAL file of this repository, sees that file's
        real content, and writes the diff; the constitution (the whole
        offline suite in a scratch clone) judges it. Measured 2026-11-03 in
        the cloud: a proposal to a file that did not exist, with a hunk
        header that did not match its own body, could never apply — the
        complex was asking for something it had not made possible.
        """
        from harness import code_forge
        from harness import forge_writers

        proposal = forge_writers.propose_patch(
            task=task,
            executor=self.executor,
            engine=self.arbiter_engine,
            gate=code_forge.apply_self_improvement,
            duty=self._duty_writer,
            local=self._local_writer,
        )
        self._deliver_self_improvement(chat_id, placeholder_id, proposal)

    def _deliver_self_improvement(
        self,
        chat_id: int,
        placeholder_id: Optional[int],
        proposal: Dict[str, Any],
    ) -> None:
        """The grounded proposal, as it was judged — or why it was not."""
        stage = str(proposal.get("stage") or "?")
        writer = proposal.get("writer") or {}
        author = str(writer.get("model") or "?")
        target = str(proposal.get("target") or "")

        if stage == "no_writer":
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Ни одна модель не предложила улучшение.</b>\n"
                f"Сетевая дежурная: {html.escape(str(proposal.get('duty_status') or 'нет ответа'))}\n"
                f"Локальная резервная: {html.escape(str(proposal.get('local_status') or 'нет ответа'))}\n\n"
                "Детерминированное ядро Хранителя не выдумывает программы.",
                parse_mode="HTML",
            )
            return

        if stage in ("no_patch", "no_target"):
            preview = str(proposal.get("answer_preview") or "")[:600]
            why = ("Модель ответила без патча" if stage == "no_patch"
                   else "Модель не назвала ни одного РЕАЛЬНОГО файла комплекса")
            body = (
                f"[-] <b>{html.escape(why)}.</b>\n"
                f"✍️ Автор: {html.escape(author)}\n\n"
                + (f"<pre>{html.escape(preview)}</pre>\n\n" if preview else "")
                + "<i>Улучшать нечего — комплекс не придумывает патчи из воздуха.</i>"
            )
            self._deliver_answer(chat_id, placeholder_id, body, parse_mode="HTML")
            return

        if proposal.get("ok"):
            body = (
                "✅ <b>Патч прошёл весь офлайн-suite в песочнице.</b>\n"
                f"✍️ Автор: {html.escape(author)}\n"
                f"🎯 Файл: <code>{html.escape(target)}</code>\n"
                f"🌿 Ветка: <code>{html.escape(str(proposal.get('branch') or ''))}</code>\n\n"
                "<i>Ветка ждёт оператора или облачную проверку — в main она не попала.</i>"
            )
        else:
            error = str(proposal.get("error") or "патч отклонён")
            repair = proposal.get("repair") or {}
            repair_line = ""
            if repair.get("attempted"):
                repair_line = (
                    f"🛠 Ремонт: {'помог' if repair.get('ok') else 'не помог'} "
                    f"({html.escape(str(repair.get('status') or '?'))})\n"
                )
            body = (
                "[-] <b>Патч отклонён конституцией.</b>\n"
                f"✍️ Автор: {html.escape(author)}\n"
                f"🎯 Файл: <code>{html.escape(target)}</code>\n"
                f"🔎 Этап: {html.escape(stage)}\n"
                f"Причина: {html.escape(error[:400])}\n"
                f"{repair_line}\n"
                "<i>Ветка не создана: без зелёного suite улучшение не становится кодом.</i>"
            )
        self._deliver_answer(chat_id, placeholder_id, body, parse_mode="HTML")

    def _apply_stored_proposal(self, chat_id: int, placeholder_id: Optional[int]) -> None:
        """
        The operator's «да» to the proposal the complex already wrote down.

        The stored diff goes through the SAME constitution as any other
        patch: vetted, applied in a scratch clone of the committed tree,
        and the whole offline suite must be green before a branch appears.
        Nothing is written to main, nothing is pushed outward.
        """
        from harness import code_forge
        from harness import forge_autopilot

        verdict = forge_autopilot.read_verdict()
        patch_file = str(verdict.get("patch_file") or "")
        if not patch_file:
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Сохранённого улучшения нет.</b>\n\n"
                "Автопилот ещё не предлагал патч — нечего применять. "
                "Скажите «улучши себя», и я предложу.",
                parse_mode="HTML",
            )
            return

        patch_path = Path(patch_file)
        if not patch_path.is_absolute():
            patch_path = PROJECT_ROOT / patch_path
        try:
            diff = patch_path.read_text(encoding="utf-8")
        except Exception as e:
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Не удалось прочитать предложение.</b>\n"
                f"{html.escape(str(e)[:200])}",
                parse_mode="HTML",
            )
            return

        try:
            outcome = code_forge.apply_self_improvement(diff)
        except Exception as e:
            self._deliver_answer(
                chat_id, placeholder_id,
                "[-] <b>Гейт не отработал.</b>\n"
                f"{html.escape(f'{type(e).__name__}: {e}'[:300])}",
                parse_mode="HTML",
            )
            return

        if outcome.get("ok"):
            body = (
                "✅ <b>Улучшение применено и прошло конституцию.</b>\n"
                f"🌿 Ветка: {html.escape(str(outcome.get('branch') or '?'))}\n"
                f"Автор предложения: {html.escape(str((verdict.get('writer') or {}).get('model') or '?'))}\n\n"
                "<i>Ветка ждёт вашего решения: в main она не вошла и наружу не отправлена.</i>"
            )
        else:
            body = (
                "[-] <b>Улучшение отклонено конституцией.</b>\n"
                f"🔎 Этап: {html.escape(str(outcome.get('stage') or '?'))}\n"
                f"Причина: {html.escape(str(outcome.get('error') or 'патч отклонён')[:400])}\n\n"
                "<i>Ветка не создана: без зелёного suite улучшение не становится кодом.</i>"
            )
        self._deliver_answer(chat_id, placeholder_id, body, parse_mode="HTML")

    def maybe_push_forge_verdict(self) -> None:
        """
        The complex's own verdict about itself — pushed once to the operator.

        The autopilot works without an order, so nobody asked for its
        result. A verdict nobody sees is a verdict nobody can accept: the
        proposal is written into the artefact, and THIS is the pipe that
        tells the operator that the complex tried, what it proposed and
        what the constitution said about it.
        """
        chat_id = self._operator_chat()
        if not chat_id:
            return  # nobody to tell yet
        from harness import forge_autopilot

        try:
            verdict = forge_autopilot.read_verdict()
        except Exception as e:
            self._record_error(f"forge verdict: {type(e).__name__}: {e}")
            return

        verdict_id = str(verdict.get("verdict_id") or "")
        if not verdict_id:
            return
        if self.health.get("last_forge_push") == verdict_id:
            return  # already pushed

        writer = verdict.get("writer") or {}
        model = str(writer.get("model") or "?")
        score = writer.get("score")
        score_txt = f" — балл {score}" if score is not None else ""
        ok = bool(verdict.get("ok"))
        mark = "✅" if ok else "⛔"

        lines = [
            f"🛠 <b>Автопилот кузницы: попытка улучшить себя</b> {mark}",
            f"Когда: {html.escape(str(verdict.get('attempted_at') or '?'))}",
            f"✍️ Автор: {html.escape(model)}{html.escape(score_txt)}",
        ]
        # The age of the ATTEMPT, not the age of the file. A verdict pushed
        # hours after the attempt must not look like the result of the
        # current cycle (measured 2026-11-06).
        freshness = str(verdict.get("freshness") or "")
        if freshness:
            lines.append(f"⏳ {html.escape(freshness)}")
        if verdict.get("branch"):
            lines.append(f"🌿 Ветка: {html.escape(str(verdict.get('branch')))}")
        if verdict.get("patch_file"):
            lines.append(f"📄 Патч: {html.escape(str(verdict.get('patch_file')))}")
        lines.append("")
        lines.append(html.escape(str(verdict.get("summary") or "")))
        if verdict.get("error"):
            lines.append("")
            lines.append(f"Причина: {html.escape(str(verdict.get('error'))[:400])}")

        facts = str(verdict.get("facts_digest") or "")
        if facts:
            lines.append("")
            lines.append("<i>На каких замерах это было построено:</i>")
            lines.append(f"<pre>{html.escape(facts[:700])}</pre>")

        if ok:
            lines.append("")
            lines.append(
                "<i>Скажите «примени последнее улучшение» — патч пройдёт "
                "конституцию ещё раз и станет веткой.</i>"
            )

        try:
            self.send_message(chat_id, "\n".join(lines), parse_mode="HTML",
                              reply_markup=NO_MARKUP)
            self.health["last_forge_push"] = verdict_id
            self._save_health()
            print(f"[*] Forge autopilot verdict pushed ({verdict_id}).")
        except Exception as e:
            self._record_error(f"forge verdict push: {type(e).__name__}: {e}")
            print(f"[!] Forge verdict push failed: {e}")

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

    def handle_sender_change(self, chat_id: int, new_name: str) -> None:
        """Updates the display name shown in the From: field of outgoing reports."""
        try:
            applied = runtime_settings.set_sender_name(new_name)
            self.send_message(
                chat_id,
                f"✍️ <b>Подпись отправителя обновлена:</b> <code>{html.escape(applied)}</code>\n\n"
                "Изменение сохранено и переживёт перезапуск раннера. "
                "Следующие письма придут с этой подписью.",
                parse_mode="HTML",
            )
            print(f"[+] Sender signature changed to: {applied}")
        except Exception as e:
            self.send_message(chat_id, f"[-] Ошибка сохранения подписи: {e}")

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
                print(f"[+] User {redaction.chat_tag(user_id)} (@{username}) successfully authenticated chat {redaction.chat_tag(chat_id)} with password.")
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
            print(f"[*] Chat {redaction.chat_tag(chat_id)} logged out.")
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
            elif state == "awaiting_sender_change":
                self.handle_sender_change(chat_id, text)
                return

        # 4. Standard Commands & Buttons
        if text in ["/start", "/help", "ℹ️ Справка"]:
            recipient = runtime_settings.get_recipient_email("configured-recipient")
            welcome = (
                "👋 <b>Добро пожаловать в Интерактивный Штаб Автономного Комплекса!</b>\n\n"
                "Вы можете управлять дежурными моделями, запрашивать исследования и мониторить систему прямо отсюда.\n\n"
                "📌 <b>Быстрые команды:</b>\n"
                "• <b>📡 Статус системы</b> — проверка ключей, диска, режимов и квот.\n"
                "• <b>🧑‍💼 Состав команды</b> — хранитель и дежурные модели с измеренными баллами.\n"
                "• <b>🔄 Обновить флот</b> — живой замер каталога NIM и переотбор дежурных.\n"
                "• <b>👑 Запрос Арбитру</b> — прямое обращение к Верховному арбитру (Qwen2.5-1.5B CPU).\n"
                "• <b>⚙️ Кузница</b> — напишите «напиши программу, которая …»: модель пишет код, "
                "песочница запускает и присылает измеренный результат. «Улучши себя» — тот же путь, "
                "но через патч и полный офлайн-suite.\n"
                "• <b>🌐 Поиск Keenable</b> — живой поиск информации без обрезки.\n"
                f"• <b>📧 Проверка почты</b> — отправка отчета здоровья на {recipient}.\n"
                "• <b>📝 Сменить email</b> — изменить получателя ежедневных отчётов.\n"
                "• <b>✍️ Подпись отправителя</b> (/setsender) — изменить имя отправителя писем.\n"
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
        elif text in ["/fleet", "/team", "🧑‍💼 Состав команды"]:
            self.handle_fleet_command(chat_id)
        elif text in ["/retune", "🔄 Обновить флот"]:
            self.handle_fleet_refresh(chat_id)
        elif text in ["/email", "📧 Проверка почты"]:
            self.handle_email_dispatch(chat_id)
        elif text in ["/setemail", "📝 Сменить email"]:
            self.user_states[chat_id] = "awaiting_email_change"
            current = runtime_settings.get_recipient_email("не задан")
            self.send_message(chat_id, f"📝 <b>Текущий email для отчётов:</b> <code>{current}</code>\n\nВведите новый email-адрес получателя:", parse_mode="HTML")
        elif text in ["/setsender", "✍️ Подпись отправителя"]:
            self.user_states[chat_id] = "awaiting_sender_change"
            current = runtime_settings.get_sender_name()
            self.send_message(chat_id, f"✍️ <b>Текущая подпись отправителя:</b> <code>{html.escape(current)}</code>\n\nВведите новую подпись отправителя:", parse_mode="HTML")
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
            # The ORDER to write code — free-form, no @role: «напиши
            # программу, которая …». The detector names both the act and
            # the artifact, so «напиши письмо про код» stays a letter.
            from harness import code_forge
            if code_forge.looks_like_code_order(text):
                self.handle_code_order(chat_id, text)
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

        # Resume from the persisted cursor: only messages that arrived since
        # the last handled update are fetched. Without this the bot replays
        # the whole 24-hour backlog on every cold start — which is exactly
        # what made it look "silent then bursty": old questions answered
        # again, hours late, while the fresh one waited in the queue.
        offset = self.last_update_id or None
        print(f"[*] Telegram cursor starts at update_id={self.last_update_id}")
        while not self._stop:
            try:
                params = {"timeout": 25}
                if offset:
                    params["offset"] = offset

                res = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=30)
                if res.status_code == 200:
                    self._record_poll()
                    # The daily control copy of the digest: throttled to
                    # once per UTC day inside, never blocks polling.
                    try:
                        self.maybe_push_daily_digest()
                    except Exception as e:
                        self._record_error(f"digest push: {type(e).__name__}: {e}")
                    # The complex's own verdict about itself: nobody ordered
                    # it, so nobody would see it unless the bot says so.
                    try:
                        self.maybe_push_forge_verdict()
                    except Exception as e:
                        self._record_error(f"forge verdict push: {type(e).__name__}: {e}")
                    data = res.json()
                    for update in data.get("result", []):
                        update_id = int(update.get("update_id", 0))
                        offset = update_id + 1
                        self._record_update()
                        self._mark_update_dispatched(update_id)
                        if "message" in update:
                            # Handle in a worker pool: long model generations must not
                            # freeze polling (the bot stays responsive to everyone else).
                            self._pool.submit(self._safe_process, update["message"], update_id)
                        elif "callback_query" in update:
                            # A button press is not a message: it arrives as its
                            # own update type. Dropping it silently would leave
                            # the operator staring at a spinning button.
                            self._pool.submit(
                                self._safe_process_callback, update["callback_query"], update_id
                            )
                        else:
                            self._mark_update_handled(update_id)
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
                self._record_error(f"poll: {type(e).__name__}: {e}")
                print(f"[!] Polling network exception: {e}. Retrying in 5s...")
                time.sleep(5)

    def _safe_process(self, message: Dict[str, Any], update_id: int = 0) -> None:
        """Runs in worker threads; keeps handler exceptions from killing the pool."""
        try:
            self.process_message(message)
        except Exception as e:
            self._record_error(f"message: {type(e).__name__}: {e}")
            print(f"[!] Message processing error: {e}")
        finally:
            # The cursor advances only AFTER the handler finished, so a crash
            # in the middle of an answer can never swallow the message.
            self._mark_update_handled(update_id)

    def _safe_process_callback(self, callback: Dict[str, Any], update_id: int = 0) -> None:
        """Same discipline for a button press."""
        try:
            self.process_callback_query(callback)
        except Exception as e:
            self._record_error(f"callback: {type(e).__name__}: {e}")
            print(f"[!] Callback processing error: {e}")
        finally:
            self._mark_update_handled(update_id)

    def process_callback_query(self, callback: Dict[str, Any]) -> None:
        """
        A button press is a command. It must be answered (or the button spins
        forever) and it must be routed to the same handler as the equivalent
        text command — and only for an authorized chat.
        """
        chat = ((callback.get("message") or {}).get("chat")) or {}
        chat_id = int(chat.get("id") or 0)
        query_id = str(callback.get("id") or "")
        data = str(callback.get("data") or "")

        if not chat_id:
            self.answer_callback_query(query_id, "не удалось определить чат")
            return
        if chat_id not in self.authorized_chats:
            # Fail closed: an unauthorized chat must not be able to trigger a
            # fleet re-measure or read the roster.
            self.answer_callback_query(query_id, "нет доступа")
            return

        self.answer_callback_query(query_id)
        action = data.split(":", 1)[0]
        if action == "status":
            self.handle_status_command(chat_id)
        elif action == "roster":
            self.handle_fleet_command(chat_id)
        elif action == "retune":
            self.handle_fleet_refresh(chat_id)
        elif action == "search":
            with self._state_lock:
                self.user_states[chat_id] = "awaiting_search_query"
            self.send_message(chat_id, "🔍 Введите поисковый запрос для поиска через Keenable:")
        else:
            self.send_message(chat_id, f"[?] Неизвестная кнопка: {html.escape(data)}")

if __name__ == "__main__":
    bot = HarnessTelegramBot()
    bot.start_polling()
