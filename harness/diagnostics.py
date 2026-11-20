#!/usr/bin/env python3
"""
Autonomous Self-Diagnostics.

Probes every external dependency the complex relies on and reports exactly
what is alive, what is degraded and what is broken. Designed to run from
GitHub Actions (the same cloud environment the complex operates in), so the
result reflects the real runtime, not the developer's laptop.

Exit code: 0 = all critical dependencies healthy, 1 = at least one broken.
"""
import os
import sys
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import atomic_io
from harness import redaction
from harness import succession

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# Model IDs the project believes are live. The fleet IDs are kept in their
# original form on purpose: the scanner resolves supplier-side renames
# against the live catalog, so a renamed model is reported as a rename
# instead of a missing model.
PROJECT_MODEL_IDS = [
    "minimaxai/minimax-m3.1-flash",
    "zhipu/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/550b-ultra",
    "stepfun/step-5-preview",
]

# Models that must exist under exactly these ids (local embeddings and the
# sovereign CPU fallback are served locally, not through NIM).
LOCAL_ONLY_MODEL_IDS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
]

# The OPTIONAL tiers of the complex. Each one is a promise the documentation
# makes and the operator fulfils with a secret; until the secret exists the
# tier is DORMANT, and the report must say so in those words. Measured
# 2026-11-04: the failover account and the archive copy existed in the
# sender and in NO workflow, and nothing told the operator — the backup
# sender was a promise kept only on paper.
TIER_CREDENTIALS = {
    "GMAIL_SENDER_2": "резервный отправитель (второй Google-аккаунт)",
    "GMAIL_APP_PASSWORD_2": "пароль резервного отправителя",
    "ARCHIVE_RECIPIENT_EMAIL": "копия письма в архивный ящик",
    "TAVILY_API_KEY": "поисковый тариф модели",
    "FIRECRAWL_API_KEY": "резервный поисковый ярус (Firecrawl)",
}


def _hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_nvidia() -> Dict[str, Any]:
    """Validates the NVIDIA NIM key and captures the live model catalog."""
    _hdr("[1] NVIDIA NIM API (integrate.api.nvidia.com)")
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        print("  STATUS: MISSING - NVIDIA_API_KEY secret is not set.")
        return {"ok": False, "reason": "missing_key", "models": []}

    result: Dict[str, Any] = {"ok": False, "models": []}
    try:
        res = requests.get(
            f"{NIM_BASE}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=25,
        )
        print(f"  GET /v1/models -> HTTP {res.status_code}")
        if res.status_code == 200:
            data = res.json().get("data", [])
            ids = [m.get("id", "") for m in data]
            result["models"] = ids
            result["ok"] = True
            print(f"  STATUS: OK - {len(ids)} models live in the catalog.")
            print(f"  First 12: {ids[:12]}")
            missing = [m for m in PROJECT_MODEL_IDS if m not in ids]
            if missing:
                # A renamed model is not a broken configuration: the scanner
                # resolves it automatically at runtime.
                try:
                    from harness.nim_scanner import NIMScanner
                    resolution = NIMScanner(api_key=os.getenv("NVIDIA_API_KEY", "")).resolve_fleet_ids(
                        [{"id": mid} for mid in ids]
                    )
                    result["id_resolution"] = resolution
                    renamed = resolution.get("renamed", {})
                    if renamed:
                        print("  [!] Supplier-side renames detected (resolved automatically):")
                        for old, new in renamed.items():
                            print(f"        {old}  ->  {new}")
                    still_missing = resolution.get("missing", [])
                    if still_missing:
                        print(f"  [!] Model IDs with no live counterpart: {still_missing}")
                        result["missing_models"] = still_missing
                    else:
                        print("  [+] Every fleet model resolves to a live catalog entry.")
                        result["missing_models"] = []
                except Exception as e:
                    print(f"  [!] Project model IDs NOT present in the live catalog: {missing}")
                    print(f"      (resolver unavailable: {e})")
                    result["missing_models"] = missing
            else:
                print("  [+] Every project model ID is present in the live catalog.")
                result["missing_models"] = []
        elif res.status_code == 401:
            print("  STATUS: BROKEN - key rejected (401 Unauthorized).")
            result["reason"] = "invalid_key"
        elif res.status_code == 403:
            print("  STATUS: BROKEN - key lacks permissions (403 Forbidden).")
            result["reason"] = "forbidden"
        elif res.status_code == 451:
            print("  STATUS: BLOCKED - HTTP 451 (geo/legal block from this runner region).")
            result["reason"] = "geo_blocked"
        else:
            print(f"  STATUS: BROKEN - unexpected response: {res.text[:200]}")
            result["reason"] = f"http_{res.status_code}"
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        result["reason"] = "network_error"
    return result


def check_keenable() -> Dict[str, Any]:
    """Validates the Keenable web-search key."""
    _hdr("[2] Keenable Search API (api.keenable.ai)")
    key = os.getenv("KEENABLE_API_KEY", "")
    if not key:
        print("  STATUS: MISSING - KEENABLE_API_KEY secret is not set (DuckDuckGo fallback used).")
        return {"ok": False, "reason": "missing_key"}

    try:
        res = requests.post(
            "https://api.keenable.ai/v1/search",
            headers={
                "X-API-Key": key,
                "Content-Type": "application/json",
                "X-Keenable-Title": "DeepSeek-TPU-Harness",
            },
            json={"query": "NVIDIA NIM model catalog", "mode": "realtime"},
            timeout=20,
        )
        print(f"  POST /v1/search -> HTTP {res.status_code}")
        if res.status_code == 200:
            body = res.json()
            n = len(body.get("results", []))
            print(f"  STATUS: OK - {n} search results returned.")
            return {"ok": True, "results": n}
        print(f"  STATUS: BROKEN - {res.text[:200]}")
        return {"ok": False, "reason": f"http_{res.status_code}"}
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "network_error"}


def check_tinyfish() -> Dict[str, Any]:
    """Probes the TinyFish search pool (tier 0) with one live query."""
    _hdr("[2b] TinyFish Search pool (api.search.tinyfish.ai)")
    try:
        from harness.tinyfish import get_tinyfish_pool
        pool = get_tinyfish_pool()
        if not pool.is_configured:
            print("  STATUS: MISSING - TINYFISH_API_KEYS secret is not set.")
            return {"ok": False, "reason": "missing_keys", "keys": 0}
        results = pool.search("NVIDIA NIM inference platform", num_results=3)
        print(f"  keys in pool: {len(pool)} | results: {len(results)}")
        if results:
            print(f"  STATUS: OK - top result: {results[0].get('url', '')[:80]}")
            return {"ok": True, "keys": len(pool), "results": len(results)}
        print("  STATUS: BROKEN - the pool returned no results (all keys rejected?).")
        return {"ok": False, "reason": "no_results", "keys": len(pool)}
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "network_error"}


def check_langsearch() -> Dict[str, Any]:
    """Probes the LangSearch client (tier 0.5) with one live query."""
    _hdr("[2c] LangSearch (api.langsearch.com)")
    try:
        from harness.langsearch import get_langsearch_client
        client = get_langsearch_client()
        if not client.is_configured:
            print("  STATUS: MISSING - LANGSEARCH_API_KEY secret is not set.")
            return {"ok": False, "reason": "missing_key"}
        results = client.search("NVIDIA NIM inference platform", num_results=3)
        print(f"  results: {len(results)}")
        if results:
            print(f"  STATUS: OK - top result: {results[0].get('url', '')[:80]}")
            return {"ok": True, "results": len(results)}
        print("  STATUS: BROKEN - no results returned.")
        return {"ok": False, "reason": "no_results"}
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "network_error"}


def check_firecrawl() -> Dict[str, Any]:
    """Probes the Firecrawl search tier (the reserve between Tavily and DuckDuckGo)."""
    _hdr("[2d] Firecrawl (api.firecrawl.dev)")
    key = os.getenv("FIRECRAWL_API_KEY", "")
    if not key:
        print("  STATUS: MISSING - FIRECRAWL_API_KEY secret is not set (tier skipped).")
        return {"ok": False, "reason": "missing_key"}
    try:
        res = requests.post(
            "https://api.firecrawl.dev/v2/search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": "NVIDIA NIM model catalog", "limit": 3},
            timeout=20,
        )
        print(f"  POST /v2/search -> HTTP {res.status_code}")
        if res.status_code == 200:
            data = res.json().get("data")
            items = (data.get("web") if isinstance(data, dict) else None) or []
            print(f"  STATUS: OK - {len(items)} search results returned.")
            return {"ok": True, "results": len(items)}
        print(f"  STATUS: BROKEN - {res.text[:200]}")
        return {"ok": False, "reason": f"http_{res.status_code}"}
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "network_error"}


def check_telegram() -> Dict[str, Any]:
    """Validates the Telegram bot token and confirms the bot identity."""
    _hdr("[3] Telegram Bot API (api.telegram.org)")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("  STATUS: MISSING - TELEGRAM_BOT_TOKEN secret is not set.")
        return {"ok": False, "reason": "missing_token"}

    try:
        res = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20)
        print(f"  GET /getMe -> HTTP {res.status_code}")
        if res.status_code == 200:
            body = res.json()
            if body.get("ok"):
                bot = body.get("result", {})
                print(f"  STATUS: OK - bot @{bot.get('username')} (id={bot.get('id')}).")
                return {"ok": True, "username": bot.get("username")}
            print(f"  STATUS: BROKEN - {body}")
            return {"ok": False, "reason": "api_not_ok"}
        print(f"  STATUS: BROKEN - {res.text[:200]}")
        return {"ok": False, "reason": f"http_{res.status_code}"}
    except Exception as e:
        print(f"  STATUS: UNREACHABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "network_error"}


def check_github_state() -> Dict[str, Any]:
    """Reports the persistence layer that survives runner restarts."""
    _hdr("[4] State persistence (repository files)")
    checks = {
        "data/.bot_sessions.json": "authorised chats (bot login survives restarts)",
        "data/telegram_history.txt": "conversation log (source of truth for memory)",
        "active_models.json": "active fleet state",
    }
    out: Dict[str, Any] = {}
    for rel, desc in checks.items():
        p = PROJECT_ROOT / rel
        if p.exists():
            size = p.stat().st_size
            print(f"  [+] {rel} - {size} bytes ({desc})")
            out[rel] = size
        else:
            print(f"  [-] {rel} - MISSING ({desc})")
            out[rel] = None

    # The semantic index is derived from the conversation log at startup, so its
    # absence on a fresh runner is expected, not a fault.
    vault = PROJECT_ROOT / "data" / ".memory_vault.json"
    if vault.exists():
        size = vault.stat().st_size
        print(f"  [+] data/.memory_vault.json - {size} bytes (derived semantic index, rebuilt on demand)")
        out["data/.memory_vault.json"] = size
    else:
        print("  [i] data/.memory_vault.json - absent (derived cache; rebuilt from the log on startup)")
        out["data/.memory_vault.json"] = None
    return {"ok": True, "files": out}


def check_credentials_present(measured: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Lists which credentials are configured without leaking their values.

    Measured 2026-11-03: a DEAD key was reported as «[+] KEENABLE_API_KEY:
    43 chars» in the same report whose live probe said DEGRADED — "the
    variable is set" and "the credential works" are different facts, and the
    inventory used to publish the first as if it were the second. Now each
    key that has a live probe in this report carries the probe's verdict
    beside the char count, and a key without one says «не измерялся здесь».
    """
    _hdr("[5] Credential inventory")
    # The key -> the check in THIS report that measured it live.
    measured_by = {
        "NVIDIA_API_KEY": "nvidia",
        "KEENABLE_API_KEY": "keenable",
        "TELEGRAM_BOT_TOKEN": "telegram",
    }
    names = [
        "NVIDIA_API_KEY", "KEENABLE_API_KEY", "HF_TOKEN", "GH_PAT",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_PASSWORD",
        "GMAIL_SENDER", "GMAIL_APP_PASSWORD", "DISPATCH_RECIPIENT_EMAIL",
    ]
    # The OPTIONAL tiers live in their own section (check_tiers_armed): they
    # are not required for the primary wire, so mixing them into «ok» would
    # turn every honest report into a red one over a missing BACKUP.
    measured = measured if isinstance(measured, dict) else {}
    out: Dict[str, bool] = {}
    verdicts: Dict[str, str] = {}
    for n in names:
        val = os.getenv(n, "")
        present = bool(val.strip())
        out[n] = present
        if not present:
            print(f"  [-] {n}: MISSING")
            verdicts[n] = "missing"
            continue
        check_name = measured_by.get(n)
        probe = measured.get(check_name) if check_name else None
        if isinstance(probe, dict) and "ok" in probe:
            verdict = "живой (замер пройден)" if probe.get("ok") else \
                f"МЁРТВ ИЛИ ДЕГРАДАЦИЯ (замер: {probe.get('reason', 'без причины')})"
            print(f"  {'[+]' if probe.get('ok') else '[!]'} {n}: {len(val)} chars — {verdict}")
        else:
            verdict = "присутствует (в этом отчёте не измерялся)"
            print(f"  [+] {n}: {len(val)} chars — {verdict}")
        verdicts[n] = verdict
    return {"ok": all(out.values()), "inventory": out, "verdicts": verdicts}


def check_tiers_armed() -> Dict[str, Any]:
    """
    Which of the OPTIONAL tiers are armed.

    Measured 2026-11-04: the failover account and the archive copy were
    implemented in the sender and wired into no workflow at all — and
    nothing anywhere told the operator that. A «backup sender» that was
    never configured is not a backup; it is a promise in the documentation
    and nowhere else.

    These tiers are not required: without them the primary wire still
    works. So their absence is not a FAILURE — it is a FACT, and the
    report says it plainly, with the exact secret the operator has to set.
    """
    _hdr("[5b] Optional tiers (the primary wire works without them)")
    armed: Dict[str, bool] = {}
    dormant: List[str] = []
    for name, what in TIER_CREDENTIALS.items():
        present = bool(os.getenv(name, "").strip())
        armed[name] = present
        if present:
            print(f"  [+] {name}: настроен ({what})")
        else:
            dormant.append(name)
            print(f"  [-] {name}: НЕ НАСТРОЕН ({what}) — секрет не задан")
    if dormant:
        print(f"  [!] {len(dormant)} tier(s) dormant: {', '.join(dormant)}")
    return {"armed": armed, "dormant": dormant, "ok": not dormant}


def check_bot_health() -> Dict[str, Any]:
    """
    Reads the bot's own health block — what it last polled, what it last
    answered, what it last failed on. This is what separates "the bot is
    not running" from "the bot is running and failing": without it both
    look like silence in the operator's chat.
    """
    _hdr("[6] Bot health (last measured activity)")
    path = PROJECT_ROOT / "data" / ".bot_sessions.json"
    if not path.exists():
        print("  STATUS: UNKNOWN - data/.bot_sessions.json is missing.")
        return {"ok": False, "reason": "no_session_file"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  STATUS: UNREADABLE - {type(e).__name__}: {e}")
        return {"ok": False, "reason": "broken_session_file"}

    health = data.get("health")
    if not isinstance(health, dict) or not health:
        print("  STATUS: UNKNOWN - the bot has not recorded its activity yet.")
        return {"ok": False, "reason": "no_health_block"}

    last_poll = health.get("last_poll_at")
    last_send = health.get("last_send_at")
    updates = int(health.get("updates_seen") or 0)
    sent = int(health.get("messages_sent") or 0)
    errors = int(health.get("errors") or 0)

    age_min: Optional[float] = None
    if last_poll:
        try:
            stamp = datetime.strptime(str(last_poll), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            age_min = round((datetime.now(timezone.utc) - stamp).total_seconds() / 60.0, 1)
        except Exception:
            age_min = None

    print(f"  Last poll      : {last_poll or '—'}" + (f" ({age_min} min ago)" if age_min is not None else ""))
    print(f"  Last answer    : {last_send or '—'}"
          + (f" (chat {redaction.chat_tag(health.get('last_chat_id'))})"
             if health.get("last_chat_id") else ""))
    print(f"  Updates seen   : {updates}")
    print(f"  Messages sent  : {sent}")
    print(f"  Errors         : {errors}" + (f" — {health.get('last_error')}" if health.get("last_error") else ""))

    # A poll older than 15 minutes means the bot is not talking to Telegram.
    if age_min is not None and age_min <= 15:
        print("  STATUS: OK - the bot polled Telegram recently.")
        return {"ok": True, "last_poll_at": last_poll, "age_min": age_min,
                "last_send_at": last_send, "updates_seen": updates,
                "messages_sent": sent, "errors": errors,
                "last_chat_id": redaction.chat_tag(health.get("last_chat_id")),
                "last_error": health.get("last_error")}

    print("  STATUS: SILENT - the bot has not polled Telegram recently.")
    return {"ok": False, "reason": "no_recent_poll", "last_poll_at": last_poll,
            "age_min": age_min, "last_send_at": last_send, "updates_seen": updates,
            "messages_sent": sent, "errors": errors,
            "last_chat_id": redaction.chat_tag(health.get("last_chat_id")),
            "last_error": health.get("last_error")}


def main() -> int:
    print("=" * 72)
    print("AUTONOMOUS COMPLEX - CLOUD SELF-DIAGNOSTICS")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Runner:    {os.getenv('GITHUB_ACTIONS', 'local')}")
    print("=" * 72)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nvidia": check_nvidia(),
        "tinyfish": check_tinyfish(),
        "langsearch": check_langsearch(),
        "keenable": check_keenable(),
        "firecrawl": check_firecrawl(),
        "telegram": check_telegram(),
        "state": check_github_state(),
        "bot_health": check_bot_health(),
    }
    # The inventory runs AFTER the live probes, so «присутствует» can carry
    # the probe's verdict instead of standing in for it.
    report["credentials"] = check_credentials_present({
        "nvidia": report["nvidia"],
        "keenable": report["keenable"],
        "telegram": report["telegram"],
    })
    # The optional tiers: which promises are still dormant. This is not a
    # verdict on the complex's health — it is the list of what the operator
    # has NOT armed yet, so nobody reads «the backup sender» in the
    # documentation and believes in a failover that was never configured.
    report["tiers"] = check_tiers_armed()
    # The heartbeat: every probed credential's last success / first failure,
    # so decay is visible in the report BEFORE the channel dies.
    report["heartbeat"] = record_heartbeat(report)
    # The mailbox inventory, measured here — because this is the cycle that
    # holds the vault's salt. The dispatch job does NOT (measured
    # 2026-11-05: REDACTION_SALT does not open the vault; only VAULT_SALT
    # does), so the letter cannot measure the inventory itself. It reads
    # this artefact instead — the same pattern as the credential heartbeat.
    report["mailboxes"] = record_mailbox_tiers()
    # The succession protocol: who governs today — the founder or the
    # steward (STANDING_ORDERS.md). A complex built for the day nobody
    # comes must measure the silence, not assume the presence.
    report["governance"] = succession.measure()

    _hdr("SUMMARY")
    nvidia = report["nvidia"]
    print(f"  NVIDIA NIM : {'OK' if nvidia.get('ok') else 'DEGRADED (' + str(nvidia.get('reason')) + ')'}")
    print(f"  TinyFish   : {'OK' if report['tinyfish'].get('ok') else 'DEGRADED (' + str(report['tinyfish'].get('reason')) + ')'}")
    print(f"  LangSearch : {'OK' if report['langsearch'].get('ok') else 'DEGRADED (' + str(report['langsearch'].get('reason')) + ')'}")
    print(f"  Keenable   : {'OK' if report['keenable'].get('ok') else 'DEGRADED (' + str(report['keenable'].get('reason')) + ')'}")
    print(f"  Firecrawl  : {'OK' if report['firecrawl'].get('ok') else 'DEGRADED (' + str(report['firecrawl'].get('reason')) + ')'}")
    print(f"  Telegram   : {'OK' if report['telegram'].get('ok') else 'DEGRADED (' + str(report['telegram'].get('reason')) + ')'}")
    print(f"  Bot health : {'OK' if report['bot_health'].get('ok') else 'SILENT (' + str(report['bot_health'].get('reason')) + ')'}")
    creds = report["credentials"]
    dead = [n for n, v in (creds.get("verdicts") or {}).items()
            if "МЁРТВ" in v or "ДЕГРАДАЦИЯ" in v]
    if not creds.get("ok"):
        cred_line = "INCOMPLETE"
    elif dead:
        cred_line = f"SET, НО {len(dead)} МЁРТВ ({', '.join(dead)})"
    else:
        cred_line = "ALL SET (живость — см. пробы выше)"
    print(f"  Credentials: {cred_line}")
    print(f"  Governance : {succession.describe(report['governance'])}")

    _hdr("CREDENTIAL HEARTBEAT (last success / first failure since)")
    for name, entry in (report.get("heartbeat") or {}).items():
        entry = entry or {}
        if entry.get("ok"):
            print(f"  [+] {name}: живой, последний успех {entry.get('last_ok_at') or entry.get('at')}")
        else:
            print(f"  [!] {name}: МЁРТВ с {entry.get('dead_since') or entry.get('at')} "
                  f"({entry.get('reason') or 'без причины'})")

    out_path = PROJECT_ROOT / "diagnostics_report.json"
    # Atomic: the supervisor reads this file's age to decide whether the
    # diagnostics cycle is alive.
    if atomic_io.write_json_atomic(out_path, report):
        print(f"\n  Report written to {out_path}")
    else:
        print("  Could not write report")

    critical_ok = nvidia.get("ok") and report["telegram"].get("ok")
    return 0 if critical_ok else 1


# ---------------------------------------------------------------------------
# The credential heartbeat — the complex sees its own decay BEFORE it dies.
#
# Measured over the audit: a dead key is only ever noticed when its channel
# goes quiet (Keenable has been http_401 for days and the first thing anyone
# saw was a missing search tier). The heartbeat records, for every probed
# credential, the LAST SUCCESS and the FIRST FAILURE since — so the letter
# and the report can say «KEENABLE_API_KEY мёртв с 2026-10-30», not
# «что-то не ищет». Statuses and timestamps only: never a value.
# ---------------------------------------------------------------------------

HEARTBEAT_PATH = PROJECT_ROOT / "data" / ".credential_heartbeat.json"

CREDENTIAL_PROBES: Tuple[tuple, ...] = (
    ("NVIDIA_API_KEY", "nvidia"),
    ("TINYFISH_API_KEYS", "tinyfish"),
    ("LANGSEARCH_API_KEY", "langsearch"),
    ("KEENABLE_API_KEY", "keenable"),
    ("FIRECRAWL_API_KEY", "firecrawl"),
    ("TELEGRAM_BOT_TOKEN", "telegram"),
)

_MAIL_LINE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\][^\n]*"
    r"ОТЧЁТЫ ОТПРАВЛЕНЫ[^\n]*статус=DELIVERED"
)


def _last_mail_delivery() -> Optional[str]:
    """The newest DELIVERED dispatch stamp in the operational log, if any."""
    try:
        text = (PROJECT_ROOT / "nim_history.log").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    stamps = _MAIL_LINE.findall(text)
    return max(stamps) if stamps else None


def record_heartbeat(report: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Merges this run's probe verdicts into data/.credential_heartbeat.json.

    A success keeps the previous `last_ok_at` history and clears
    `dead_since`; a failure keeps the OLDEST `dead_since` (when the key
    first died) — that is the number the operator needs. No credential
    value ever enters the file.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat()
    try:
        old = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
        old = old if isinstance(old, dict) else {}
    except Exception:
        old = {}

    heartbeat: Dict[str, Any] = {}
    for env_name, probe_key in CREDENTIAL_PROBES:
        probe = report.get(probe_key) or {}
        ok = bool(probe.get("ok"))
        prev = old.get(env_name) if isinstance(old.get(env_name), dict) else {}
        entry = {
            "ok": ok,
            "at": stamp,
            "reason": "" if ok else str(probe.get("reason") or "")[:120],
            "last_ok_at": stamp if ok else prev.get("last_ok_at"),
            "last_fail_at": stamp if not ok else prev.get("last_fail_at"),
            "dead_since": (prev.get("dead_since") if not ok else None),
        }
        if not ok and not entry["dead_since"]:
            entry["dead_since"] = stamp
        heartbeat[env_name] = entry

    # The mail channel is measured by the dispatch itself: its heartbeat is
    # the newest DELIVERED line in the log.
    last_mail = _last_mail_delivery()
    mail_ok = False
    if last_mail:
        try:
            delivered_at = datetime.strptime(last_mail, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            mail_ok = (now - delivered_at).total_seconds() <= 26 * 3600
        except ValueError:
            mail_ok = False
    prev_mail = old.get("GMAIL_APP_PASSWORD") if isinstance(old.get("GMAIL_APP_PASSWORD"), dict) else {}
    mail_entry = {
        "ok": mail_ok,
        "at": stamp,
        "reason": "" if mail_ok else "письма не доставлялись более 26 часов",
        "last_ok_at": last_mail or prev_mail.get("last_ok_at"),
        "last_fail_at": stamp if not mail_ok else prev_mail.get("last_fail_at"),
        "dead_since": prev_mail.get("dead_since") if not mail_ok else None,
    }
    if not mail_ok and not mail_entry["dead_since"]:
        mail_entry["dead_since"] = stamp
    heartbeat["GMAIL_APP_PASSWORD"] = mail_entry

    atomic_io.write_json_atomic(HEARTBEAT_PATH, heartbeat)
    return heartbeat


# ---------------------------------------------------------------------------
# The mailbox inventory — what the complex can actually USE.
#
# Measured 2026-11-05: the vault is sealed under VAULT_SALT, and
# REDACTION_SALT (the salt the dispatch and diagnostics jobs carry) does NOT
# open it. So the letter's job cannot measure the inventory itself — it can
# only read what a cycle that HOLDS the salt measured. This cycle holds it.
#
# The file carries COUNTS and tiers, never an address and never a password:
# the operator already has the addresses (they are his), and the letter only
# needs to say «1 mailbox I can read unattended, 2 that need a human, 5 of
# my own that are not sealed, 5 that were session-only from the start».
# ---------------------------------------------------------------------------

MAILBOX_TIERS_PATH = PROJECT_ROOT / "data" / ".mailbox_tiers.json"


def record_mailbox_tiers(now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Measures the mailbox inventory and writes it as an artefact.

    Counts and tiers only. If the vault cannot be opened the result says
    «not measured» — never «zero»: a cycle that could not look must not
    report an empty shelf.
    """
    from harness import mail_accounts
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    try:
        tiers = mail_accounts.mailbox_tiers()
    except Exception as exc:
        tiers = {
            "measured": False,
            "api_readable": 0, "human_only": 0,
            "own_durable_sealed": 0, "own_durable_local": 0,
            "own_session": 0, "own_total": 0, "sealed_in_vault": 0,
            "why": f"the inventory could not be measured: {exc}",
        }
    tiers["checked_at"] = stamp
    atomic_io.write_json_atomic(MAILBOX_TIERS_PATH, tiers)
    return tiers


if __name__ == "__main__":
    sys.exit(main())
