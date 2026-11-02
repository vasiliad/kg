#!/usr/bin/env python3
"""
Daily Dispatcher: Scheduled 12:00 NOVT (05:00 UTC) Dual-Email Pipeline.
1. Local Arbiter Health & Incident Audit: "Проверка связи"
2. Daily AI News Intelligence Digest: "Ежедневный дайджест ИИ"

Route: configured via GMAIL_SENDER (sender) and DISPATCH_RECIPIENT_EMAIL (recipient)

WHAT CHANGED (2026-11-02) — the operator called the letters «просто ужас
ужасный», in form AND in content. Both defects were measured in this file:

  FORM. The digest was the MODEL's markdown, "converted" with a chain of
  `.replace()` calls: `##` stayed literal, `<b>` was opened and never
  closed, links were not clickable, and whatever HTML the model produced was
  pasted in unescaped. The health letter used `display:flex` and a CSS
  gradient, which Outlook and Windows Mail drop entirely.

  CONTENT. The footer credited «Keenable» whatever tier actually served the
  search; the incident column was the constant «Автоматически локализовано
  (NIMPacer / Circuit Breaker / Emergency Fallback)» on every single row —
  including rows that say the reports were NOT sent; the API key status was
  an English sentence; the plain-text part dropped the fleet and the
  incidents and printed `None` for the arbiter; and the recipient's own
  address was printed in the footer of a letter that lands in a public
  repository's log.

The new pipeline is the plan's Workstream B:

  1. targeted queries (RU + EN) go to the search chain, each result keeping
     the tier that served it;
  2. ONE model call selects and compresses the found sources into strict
     JSON, picking sources BY NUMBER — it never supplies a URL;
  3. the layout is OURS (`harness/email_render.py`), everything the model
     wrote is escaped, and every link is checked against the whitelist of
     what was actually found;
  4. the letter says which tier served the search, how many queries ran and
     how many sources were read — and carries the recipient's FINGERPRINT,
     never the address;
  5. no letter leaves without `validate_html`.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.email_sender import EmailSender
from harness import redaction
from harness import email_render
from harness import runtime_settings
from harness import tool_call
from harness.disk_guard import DiskGuard
from harness.state_manager import StateManager
from harness.api_healthcheck import APIHealthChecker
from harness.minimax_executor import MiniMaxExecutor

DIGEST_ARTIFACT = PROJECT_ROOT / "data" / "daily_digest.json"

# The recipient is resolved at send time (not import time) so a change made
# from Telegram is picked up by the scheduled dispatcher.
def resolve_recipient() -> str:
    return runtime_settings.get_recipient_email()

RECIPIENT_EMAIL = resolve_recipient()
SENDER_EMAIL = os.getenv("GMAIL_SENDER", "")
SENDER_NAME = runtime_settings.get_sender_name()
if not SENDER_EMAIL:
    print("[!] WARNING: GMAIL_SENDER not set! Email dispatch will be skipped.")


def get_recent_history_logs(history_file: str = "nim_history.log", hours: int = 24) -> List[Dict[str, Any]]:
    """
    Parses entries from nim_history.log within the last N hours.
    """
    if not os.path.exists(history_file):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    parsed = []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("["):
                    continue
                try:
                    # Format: [2026-10-29 09:30:30 UTC] message
                    end_idx = line.find("]")
                    if end_idx != -1:
                        ts_str = line[1:end_idx].replace(" UTC", "")
                        entry_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        msg = line[end_idx + 1:].strip()
                        if entry_dt >= cutoff:
                            parsed.append({
                                "timestamp": ts_str,
                                "message": msg,
                                "is_error": any(kw in msg.lower() for kw in ["error", "fail", "429", "timeout", "abort", "danger", "alert", "не отправлены"]),
                                "is_success": any(kw in msg.lower() for kw in ["success", "approved", "completed", "elected", "healthy", "отправлены"])
                            })
                except Exception:
                    continue
    except Exception as e:
        print(f"[!] Warning reading history log: {e}")

    return parsed


# --------------------------------------------------------------------------
# Incident state — derived from the log line, never a constant
# --------------------------------------------------------------------------

def incident_state(message: str) -> str:
    """
    What actually happened, read out of the log line itself.

    The old letter printed «Автоматически локализовано (NIMPacer / Circuit
    Breaker / Emergency Fallback)» on every row, including the rows that say
    the reports were NOT sent — a claim the letter's own evidence
    contradicts. The state is now the state of the event.
    """
    text = (message or "").upper()
    if "ОТЧЁТЫ НЕ ОТПРАВЛЕНЫ" in text:
        return "Не отправлено"
    if "ОТЧЁТЫ ОТПРАВЛЕНЫ" in text:
        return "Доставлено"
    return "Требует внимания"


def key_state_ru(key_status: Dict[str, Any]) -> Tuple[str, bool]:
    """
    The API key's state in Russian, by the status code that was measured.

    The old letter pasted the English `message` ("Key is active and
    authorized.") into a Russian letter.
    """
    key_status = key_status or {}
    code = key_status.get("status_code")
    if key_status.get("valid"):
        return "Ключ активен, доступ разрешён", True
    if code == 401:
        return "Ключ отозван или истёк (401) — нужна замена", False
    if code == 403:
        return "Доступ запрещён для этого ключа (403)", False
    if code == 451:
        return "Регион раннера заблокирован (451) — менять регион, не ключ", False
    if code == 429:
        return "Исчерпан лимит запросов (429)", False
    if code in (0, None):
        return "Ключ не настроен или сеть недоступна", False
    return f"Ключ не прошёл проверку (HTTP {code})", False


def _fleet_rows(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """The measured fleet, with the honesty contract about the score."""
    rows: List[Dict[str, str]] = []
    for model in (state.get("active_fleet") or [])[:8]:
        if not isinstance(model, dict):
            rows.append({"id": str(model), "roles": [], "score_text": "UNRANKED", "latency": ""})
            continue
        if model.get("score_verified") and model.get("quality_score") is not None:
            score_txt = f"{model['quality_score']} (живой замер)"
        elif model.get("quality_score") is not None:
            score_txt = f"{model['quality_score']} (статический сид, не проверено)"
        else:
            score_txt = "UNRANKED"
        measured = model.get("measured") or {}
        latency = measured.get("latency_sec")
        rows.append({
            "id": str(model.get("id", "—")),
            "roles": [str(r) for r in (model.get("roles") or [])] or [str(model.get("tier") or "")],
            "score_text": score_txt,
            "latency": f"{latency}с" if isinstance(latency, (int, float)) else "",
        })
    return rows


def build_health_report_email() -> Dict[str, str]:
    """
    Builds Email 1: system health, disk metrics, incidents, and the fleet.

    Content rule: every line is something that was MEASURED. No incident is
    declared "automatically localised" unless the log says it was resolved.
    """
    state_mgr = StateManager()
    state = state_mgr.load_local_state()

    dg = DiskGuard()
    disk_metrics = dg.inspect_disk_usage()

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    key_status = {"valid": False, "status_code": 0, "message": "Key not configured"}
    if nvidia_key:
        checker = APIHealthChecker(api_key=nvidia_key)
        key_status = checker.check_key_validity()
    api_state, api_ok = key_state_ru(key_status)

    # The incidents the operator must be able to check: everything that
    # failed AND every delivery record (those are the answers to "were the
    # reports sent?").
    recent_logs = get_recent_history_logs(hours=24)
    interesting = [
        log for log in recent_logs
        if log["is_error"] or "ОТЧЁТЫ" in log["message"].upper()
    ]

    if state.get("sovereign_zero_api_mode"):
        mode = "Экстренный суверенный CPU (без сети)"
    elif state.get("emergency_mode"):
        mode = "Экстренный CPU fallback"
    else:
        mode = "Штатный сетевой режим (NIM)"

    now_novt = datetime.now(timezone(timedelta(hours=7))).strftime("%d.%m.%Y %H:%M НСК")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = {
        "subject": "Проверка связи",
        "now_label": f"{now_novt} ({now_utc})",
        "api": {"state": api_state, "ok": api_ok},
        "disk": {
            "free_gb": disk_metrics.get("free_gb"),
            "percent_used": disk_metrics.get("percent_used"),
        },
        "arbiter": {
            "model": state.get("local_arbiter") or "не зафиксирован",
            "mode": mode,
        },
        "incidents": [
            {
                "timestamp": log["timestamp"],
                "message": log["message"],
                "state": incident_state(log["message"]),
            }
            for log in interesting
        ],
        "fleet": _fleet_rows(state),
        "recipient_tag": redaction.recipient_tag(resolve_recipient()),
        "sender_name": SENDER_NAME,
    }

    html, plain = email_render.render_health(report)
    return {"subject": report["subject"], "html": html, "plain": plain}


# --------------------------------------------------------------------------
# The digest: targeted queries -> model selection -> OUR layout
# --------------------------------------------------------------------------

# Targeted queries, RU and EN. The old pipeline sent ONE vague prompt
# («подробный, высокопрофессиональный ... 4 раздела с эмодзи») and let the
# model search however it liked — which is how a letter gets written from
# the model's memory instead of from the live web.
DIGEST_QUERIES: List[Tuple[str, str, str]] = [
    ("Модели", "новые модели ИИ релиз открытые веса", "new AI model release open weights LLM"),
    ("Агенты", "агентные ИИ системы tool calling фреймворки", "AI agents frameworks autonomous tool calling"),
    ("Инфраструктура", "NVIDIA GPU TPU инфраструктура ИИ чипы", "AI infrastructure chips NVIDIA GPU TPU datacenter"),
    ("Исследования", "исследования ИИ reasoning бенчмарки модели", "AI research reasoning benchmark LLM paper"),
]

MAX_ITEMS = 6
PER_QUERY_RESULTS = 5


def collect_digest_sources(executor: MiniMaxExecutor) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Asks the search chain the targeted queries and returns
    (candidates, tiers, queries).

    Every candidate keeps the tier that served it, so the letter can say
    where the facts came from instead of naming a brand that was never
    asked.
    """
    candidates: List[Dict[str, Any]] = []
    seen_urls = set()
    tiers: List[str] = []
    queries: List[str] = []

    for category, ru_query, en_query in DIGEST_QUERIES:
        for query in (ru_query, en_query):
            print(f"[*] Digest search [{category}]: '{query}'")
            try:
                results = executor.search_web(query, num_results=PER_QUERY_RESULTS) or []
            except Exception as e:
                print(f"[!] Search failed for '{query}': {type(e).__name__}: {e}")
                results = []
            tier = getattr(executor, "last_search_tier", "") or ""
            if tier and tier not in tiers:
                tiers.append(tier)
            queries.append(query)
            for result in results:
                url = str(result.get("url") or "").strip()
                title = str(result.get("title") or "").strip()
                snippet = str(result.get("snippet") or "").strip()
                if not url or not title or url in seen_urls:
                    continue
                seen_urls.add(url)
                candidates.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "category": category,
                    "tier": tier,
                })
            print(f"    -> {len(results)} results (tier: {tier or 'none'})")

    return candidates, tiers, queries


SELECTION_PROMPT = (
    "Ты редактор ежедневного дайджеста новостей искусственного интеллекта.\n"
    "Ниже — список источников, найденных живым веб-поиском за последние сутки.\n\n"
    "Отбери до {max_items} самых существенных новостей и ответь СТРОГО одним JSON-объектом:\n"
    '{{"items": [{{"n": <номер источника>, "title": "заголовок по-русски", '
    '"summary": "2-3 предложения по-русски", "date": "дата, если она есть в источнике, иначе пустая строка", '
    '"category": "Модели|Агенты|Инфраструктура|Исследования"}}]}}\n\n'
    "Правила:\n"
    "1. n — номер источника из списка. Ссылку НЕ пиши: её подставит харнесс.\n"
    "2. title и summary — только факты из этого источника. Ничего не добавляй из своей памяти.\n"
    "3. Разные новости — разные источники; не бери два пункта из одного источника.\n"
    "4. Если существенных новостей нет — верни {{\"items\": []}}.\n"
    "5. Никакого текста вокруг JSON: ни пояснений, ни markdown-заборов."
)


def _sources_block(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for index, candidate in enumerate(candidates, 1):
        snippet = candidate["snippet"][:600].replace("\n", " ")
        lines.append(
            f"[{index}] {candidate['title']}\n"
            f"    url: {candidate['url']}\n"
            f"    раздел: {candidate['category']}\n"
            f"    текст: {snippet}"
        )
    return "\n\n".join(lines)


def parse_selection(text: str) -> List[Dict[str, Any]]:
    """The model's JSON answer, whatever wrapper it arrived in."""
    if not text:
        return []
    span = tool_call.outermost_json_span(text)
    for blob in [span] + [json.dumps(obj, ensure_ascii=False) for obj in tool_call.iter_json_objects(text)]:
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
    return []


def select_digest_items(
    executor: MiniMaxExecutor,
    candidates: List[Dict[str, Any]],
    model_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    ONE model call that selects and compresses. Returns (items, note).

    The model picks sources BY NUMBER and never supplies a URL, so a
    fabricated link cannot reach the letter: the URL, the source name and
    the category are attached here, from what was actually found. An item
    whose number is not in the list is dropped.
    """
    if not candidates:
        return [], "источников не найдено"

    prompt = SELECTION_PROMPT.format(max_items=MAX_ITEMS)
    context = f"Найденные источники ({len(candidates)}):\n\n{_sources_block(candidates)}"

    print(f"[*] Selecting the digest items via {model_id or executor.model_id}...")
    try:
        res = executor.execute_task(
            task_prompt=prompt,
            context=context,
            enable_search=False,
            max_tokens=8192,
            temperature=0.2,
            model_id=model_id,
        )
    except Exception as e:
        return [], f"модель недоступна ({type(e).__name__})"

    if res.get("status") != "SUCCESS":
        return [], f"модель не ответила ({res.get('error', 'unknown')})"

    raw_items = parse_selection(res.get("content", ""))
    if not raw_items:
        return [], "модель не вернула разборчивый JSON"

    items: List[Dict[str, Any]] = []
    used_numbers = set()
    for raw in raw_items:
        try:
            number = int(raw.get("n"))
        except Exception:
            continue
        if number < 1 or number > len(candidates) or number in used_numbers:
            continue
        used_numbers.add(number)
        source = candidates[number - 1]
        title = str(raw.get("title") or "").strip() or source["title"]
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            continue
        items.append({
            "title": title,
            "summary": summary,
            "url": source["url"],
            "source": source["title"],
            "date": str(raw.get("date") or "").strip(),
            "category": str(raw.get("category") or "").strip() or source["category"],
            "tier": source["tier"],
        })
        if len(items) >= MAX_ITEMS:
            break

    if not items:
        return [], "ни один выбранный источник не подтвердился"
    return items, ""


def fallback_items(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    What the letter shows when the model could not select: the sources
    themselves, with their own text. Honest and still useful — the operator
    gets the links the search actually returned.
    """
    items: List[Dict[str, Any]] = []
    seen_categories = set()
    for candidate in candidates:
        category = candidate["category"]
        if category in seen_categories:
            continue
        seen_categories.add(category)
        snippet = candidate["snippet"].strip() or "Источник найден, но без описания."
        items.append({
            "title": candidate["title"],
            "summary": snippet[:400],
            "url": candidate["url"],
            "source": candidate["title"],
            "date": "",
            "category": category,
            "tier": candidate["tier"],
        })
        if len(items) >= MAX_ITEMS:
            break
    return items


def build_ai_news_digest_email() -> Dict[str, Any]:
    """
    Builds Email 2: the daily AI digest — targeted search, one selection
    call, our own layout.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    keenable_key = os.getenv("KEENABLE_API_KEY", "")
    now = datetime.now(timezone(timedelta(hours=7)))
    now_date = now.strftime("%d.%m.%Y")
    now_label = now.strftime("%d.%m.%Y 12:00 НСК")

    subject = f"Ежедневный дайджест ИИ: {now_date}"
    executor = MiniMaxExecutor(nvidia_api_key=nvidia_key, keenable_api_key=keenable_key)

    print("[*] Collecting the digest sources from the live web...")
    candidates, tiers, queries = collect_digest_sources(executor)

    items, note = select_digest_items(executor, candidates)
    if not items and candidates:
        print(f"[!] Selection failed ({note}); falling back to the found sources.")
        items = fallback_items(candidates)
        note = note or "отбор не удался"

    meta = {
        "subject": subject,
        "date_label": now_label,
        # The tier that ACTUALLY served the search — the old footer said
        # «Keenable» even when TinyFish answered.
        "tier": ", ".join(tiers),
        "queries": queries,
        "sources_count": len(candidates),
        "recipient_tag": redaction.recipient_tag(resolve_recipient()),
        "sender_name": SENDER_NAME,
        "elapsed": None,
    }
    html, plain = email_render.render_digest(items, meta)
    if note and not items:
        plain += f"\n({note})"

    return {
        "subject": subject,
        "html": html,
        "plain": plain,
        "items": items,
        "meta": meta,
    }


def write_digest_artifact(mail: Dict[str, Any]) -> None:
    """
    Keeps the digest's content in the repository so the bot can quote it
    later. Masked before writing: the artefact is committed, the repository
    is public (harness/redaction.py).
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": mail.get("subject"),
        "tier": (mail.get("meta") or {}).get("tier", ""),
        "queries": (mail.get("meta") or {}).get("queries", []),
        "sources_count": (mail.get("meta") or {}).get("sources_count", 0),
        "items": [
            {
                "title": item.get("title"),
                "summary": item.get("summary"),
                "url": item.get("url"),
                "source": item.get("source"),
                "date": item.get("date"),
                "category": item.get("category"),
            }
            for item in (mail.get("items") or [])
        ],
    }
    try:
        DIGEST_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        with open(DIGEST_ARTIFACT, "w", encoding="utf-8") as f:
            json.dump(redaction.mask_emails_deep(payload), f, ensure_ascii=False, indent=2)
        print(f"[*] Digest artefact written: {DIGEST_ARTIFACT}")
    except Exception as e:
        print(f"[!] Could not write the digest artefact: {e}")


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------

def _log_dispatch(label: str, recipient: str, result: Dict[str, Any]) -> None:
    """
    Records the outcome of a report dispatch in the operational log.

    Without this record nobody — including the Command Center itself — can
    answer whether the daily reports were actually sent. The wording is
    natural language on purpose: the log is searched both semantically and
    lexically, so "отчёты отправлены?" must match the record literally.

    The recipient is recorded as a FINGERPRINT, not an address: this log is
    committed to a public repository (see harness/redaction.py).
    """
    status = (result or {}).get("status", "UNKNOWN")
    error = (result or {}).get("error") or (result or {}).get("message") or ""
    detail = f" | причина={str(error)[:180]}" if error else ""
    if status == "DELIVERED":
        headline = "ОТЧЁТЫ ОТПРАВЛЕНЫ"
    else:
        headline = "ОТЧЁТЫ НЕ ОТПРАВЛЕНЫ"
    try:
        StateManager().append_history_log(
            f"{headline} | тип={label} | получатель={redaction.recipient_tag(recipient)} | "
            f"статус={status}{detail}"
        )
        print(f"[*] Dispatch logged: {label} -> {status} (recipient={redaction.recipient_tag(recipient)})")
    except Exception as e:
        print(f"[!] Could not log dispatch outcome: {e}")


def _dispatch_one(
    sender: EmailSender,
    label: str,
    mail: Dict[str, str],
    recipient: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Validates and sends ONE letter.

    No letter leaves without `email_render.validate_html`: an unbalanced tag
    is exactly what made the previous letters unreadable.
    """
    problems = email_render.validate_html(mail.get("html", ""))
    if problems:
        result = {"status": "ERROR", "error": "HTML письма повреждён: " + "; ".join(problems[:3])}
        print(f"[-] {label}: {result['error']}")
        _log_dispatch(label, recipient, result)
        return result

    if dry_run:
        print(f"[DRY-RUN] {label} ready. Length: {len(mail.get('html', ''))} chars.")
        return {"status": "DRY_RUN_OK", "subject": mail.get("subject")}

    result = sender.send_email(
        recipient_email=recipient,
        subject=mail["subject"],
        html_content=mail["html"],
        plain_text_content=mail["plain"],
    )
    _log_dispatch(label, recipient, result)
    return result


def run_daily_dispatch(dry_run: bool = False, health_only: bool = False, news_only: bool = False) -> Dict[str, Any]:
    print("=" * 72)
    print("📬 STARTING DAILY 12:00 NOVT DISPATCH PIPELINE")
    print(f"[*] Dispatcher timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    sender = EmailSender()
    results: Dict[str, Any] = {}

    # Resolved at send time so a recipient changed from Telegram is honoured
    # by the scheduled dispatch, not just by manual checks.
    recipient = resolve_recipient()
    if not recipient:
        print("[!] WARNING: no dispatch recipient configured (DISPATCH_RECIPIENT_EMAIL unset).")
    else:
        print(f"[*] Dispatch recipient: {redaction.recipient_tag(recipient)}")

    # 1. Email 1: Проверка связи (Health & Incidents).
    # Each letter is built and sent under its OWN try/except: a broken health
    # letter used to cancel the digest as well, and the operator got nothing.
    if not news_only:
        print("\n--- GENERATING EMAIL 1: Проверка связи ---")
        try:
            health_mail = build_health_report_email()
            results["health_report"] = _dispatch_one(
                sender, "Проверка связи", health_mail, recipient, dry_run
            )
        except Exception as e:
            print(f"[-] Error generating the health report: {e}")
            results["health_report"] = {"status": "ERROR", "error": str(e)}
            _log_dispatch("Проверка связи", recipient, results["health_report"])

    # 2. Email 2: Ежедневный дайджест ИИ
    if not health_only:
        print("\n--- GENERATING EMAIL 2: Ежедневный дайджест ИИ ---")
        try:
            news_mail = build_ai_news_digest_email()
            write_digest_artifact(news_mail)
            results["news_digest"] = _dispatch_one(
                sender, "Ежедневный дайджест ИИ", news_mail, recipient, dry_run
            )
        except Exception as e:
            print(f"[-] Error generating news digest: {e}")
            results["news_digest"] = {"status": "ERROR", "error": str(e)}
            _log_dispatch("Ежедневный дайджест ИИ", recipient, results["news_digest"])

    print("\n" + "=" * 72)
    print("✅ DAILY DISPATCH PIPELINE COMPLETE.")
    print("=" * 72)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily 12:00 NOVT Email Dispatcher")
    parser.add_argument("--dry-run", action="store_true", help="Generate emails without sending via SMTP")
    parser.add_argument("--health-only", action="store_true", help="Dispatch only 'Проверка связи'")
    parser.add_argument("--news-only", action="store_true", help="Dispatch only 'Ежедневный дайджест ИИ'")
    parser.add_argument("--preview", metavar="PATH", help="Write both letters to PATH (HTML) and exit")
    args = parser.parse_args()

    if args.preview:
        preview_path = Path(args.preview)
        health = build_health_report_email()
        preview_path.write_text(health["html"], encoding="utf-8")
        print(f"[*] Health letter preview: {preview_path}")
        print("[*] validate_html:", email_render.validate_html(health["html"]) or "OK")
    else:
        run_daily_dispatch(dry_run=args.dry_run, health_only=args.health_only, news_only=args.news_only)
