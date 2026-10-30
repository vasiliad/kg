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
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# Model IDs the project believes are live. Verified against the real catalog.
PROJECT_MODEL_IDS = [
    "minimaxai/minimax-m3.1-flash",
    "zhipu/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/550b-ultra",
    "stepfun/step-5-preview",
    "nvidia/nemotron-3.5-embed-8b",
    "nvidia/nemotron-3.5-embed-1b",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


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
                print(f"  [!] Project model IDs NOT present in the live catalog: {missing}")
                result["missing_models"] = missing
            else:
                print("  [+] Every project model ID is present in the live catalog.")
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
        "data/telegram_history.txt": "conversation log",
        "data/.memory_vault.json": "semantic memory vault",
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
    return {"ok": True, "files": out}


def check_credentials_present() -> Dict[str, Any]:
    """Lists which credentials are configured without leaking their values."""
    _hdr("[5] Credential inventory")
    names = [
        "NVIDIA_API_KEY", "KEENABLE_API_KEY", "HF_TOKEN", "GH_PAT",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_PASSWORD",
        "GMAIL_SENDER", "GMAIL_APP_PASSWORD", "DISPATCH_RECIPIENT_EMAIL",
    ]
    out: Dict[str, bool] = {}
    for n in names:
        val = os.getenv(n, "")
        present = bool(val.strip())
        out[n] = present
        shown = f"{len(val)} chars" if present else "MISSING"
        print(f"  {'[+]' if present else '[-]'} {n}: {shown}")
    return {"ok": all(out.values()), "inventory": out}


def main() -> int:
    print("=" * 72)
    print("AUTONOMOUS COMPLEX - CLOUD SELF-DIAGNOSTICS")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Runner:    {os.getenv('GITHUB_ACTIONS', 'local')}")
    print("=" * 72)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nvidia": check_nvidia(),
        "keenable": check_keenable(),
        "telegram": check_telegram(),
        "state": check_github_state(),
        "credentials": check_credentials_present(),
    }

    _hdr("SUMMARY")
    nvidia = report["nvidia"]
    print(f"  NVIDIA NIM : {'OK' if nvidia.get('ok') else 'DEGRADED (' + str(nvidia.get('reason')) + ')'}")
    print(f"  Keenable   : {'OK' if report['keenable'].get('ok') else 'DEGRADED (' + str(report['keenable'].get('reason')) + ')'}")
    print(f"  Telegram   : {'OK' if report['telegram'].get('ok') else 'DEGRADED (' + str(report['telegram'].get('reason')) + ')'}")
    print(f"  Credentials: {'ALL SET' if report['credentials'].get('ok') else 'INCOMPLETE'}")

    out_path = PROJECT_ROOT / "diagnostics_report.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report written to {out_path}")
    except Exception as e:
        print(f"  Could not write report: {e}")

    critical_ok = nvidia.get("ok") and report["telegram"].get("ok")
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
