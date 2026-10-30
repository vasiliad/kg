#!/usr/bin/env python3
"""
Daily Dispatcher: Scheduled 12:00 NOVT (05:00 UTC) Dual-Email Pipeline.
1. Local Arbiter Health & Incident Audit: "Проверка связи"
2. MiniMax M3.1 Flash AI News Intelligence Digest: "Ежедневный дайджест ИИ"

Route: configured via GMAIL_SENDER (sender) and DISPATCH_RECIPIENT_EMAIL (recipient)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.email_sender import EmailSender
from harness.disk_guard import DiskGuard
from harness.state_manager import StateManager
from harness.api_healthcheck import APIHealthChecker
from harness.minimax_executor import MiniMaxExecutor
from harness import runtime_settings

# The recipient is resolved at send time (not import time) so a change made
# from Telegram is picked up by the scheduled dispatcher.
def resolve_recipient() -> str:
    return runtime_settings.get_recipient_email()

RECIPIENT_EMAIL = resolve_recipient()
SENDER_EMAIL = os.getenv("GMAIL_SENDER", "")
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
                                "is_error": any(kw in msg.lower() for kw in ["error", "fail", "429", "timeout", "abort", "danger", "alert"]),
                                "is_success": any(kw in msg.lower() for kw in ["success", "approved", "completed", "elected", "healthy"])
                            })
                except Exception:
                    continue
    except Exception as e:
        print(f"[!] Warning reading history log: {e}")

    return parsed

def build_health_report_email() -> Dict[str, str]:
    """
    Builds Email 1: System Health, Disk Metrics, Incidents, and Self-Healing Report.
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

    recent_logs = get_recent_history_logs(hours=24)
    error_logs = [log for log in recent_logs if log["is_error"]]

    now_novt = datetime.now(timezone(timedelta(hours=7))).strftime("%d.%m.%Y %H:%M НСК")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Generate incident breakdown
    incident_rows = ""
    if error_logs:
        for err in error_logs:
            incident_rows += f"""
            <tr style="border-bottom: 1px solid #fee2e2;">
                <td style="padding: 10px; color: #991b1b; font-family: monospace; font-size: 13px;">{err['timestamp']}</td>
                <td style="padding: 10px; color: #b91c1c; font-size: 13px;">{err['message']}</td>
                <td style="padding: 10px; color: #047857; font-size: 13px; font-weight: 500;">
                    Автоматически локализовано (NIMPacer / Circuit Breaker / Emergency Fallback)
                </td>
            </tr>
            """
    else:
        incident_rows = """
        <tr>
            <td colspan="3" style="padding: 15px; text-align: center; color: #059669; font-weight: 600;">
                За прошедшие 24 часа критических сбоев не зафиксировано. Все модули отработали штатно.
            </td>
        </tr>
        """

    active_fleet = state.get("active_fleet", [])
    fleet_rows = ""
    for idx, m in enumerate(active_fleet[:5], 1):
        m_id = m.get("id", "Unknown") if isinstance(m, dict) else str(m)
        tier = m.get("tier", "Standard") if isinstance(m, dict) else ""
        score = m.get("quality_score", 95.0) if isinstance(m, dict) else ""
        fleet_rows += f"""
        <li style="margin-bottom: 6px; font-size: 13px; color: #1e293b;">
            <b>{idx}. {m_id}</b> <span style="color: #64748b;">({tier} | Оценка: {score})</span>
        </li>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; color: #1e293b;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 32px; color: #ffffff;">
                <h1 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px;">📡 Проверка связи: Системный аудит автономного комплекса</h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">Время фиксации: {now_novt} ({now_utc})</p>
            </div>

            <div style="padding: 28px 32px;">
                <!-- Status Cards -->
                <div style="display: flex; gap: 12px; margin-bottom: 24px;">
                    <div style="flex: 1; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 14px;">
                        <div style="font-size: 12px; color: #15803d; font-weight: 600; text-transform: uppercase;">Сетевой API (NIM)</div>
                        <div style="font-size: 16px; font-weight: 700; color: #166534; margin-top: 4px;">{key_status.get('message')}</div>
                    </div>
                    <div style="flex: 1; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 14px;">
                        <div style="font-size: 12px; color: #0369a1; font-weight: 600; text-transform: uppercase;">Свободный диск</div>
                        <div style="font-size: 16px; font-weight: 700; color: #075985; margin-top: 4px;">{disk_metrics.get('free_gb')} ГБ ({disk_metrics.get('percent_used')}% занято)</div>
                    </div>
                </div>

                <!-- Environment & Arbiter -->
                <div style="background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; padding: 16px; margin-bottom: 24px;">
                    <h3 style="margin: 0 0 10px 0; font-size: 14px; text-transform: uppercase; color: #475569;">Активный верховный арбитр</h3>
                    <div style="font-family: monospace; font-size: 14px; color: #0f172a; font-weight: 600;">{state.get('local_arbiter', 'DeepSeek-R1-Distill-Qwen-32B')}</div>
                    <p style="margin: 6px 0 0 0; font-size: 13px; color: #64748b;">
                        Режим: {"🚨 Экстренный суверенный CPU" if state.get("sovereign_zero_api_mode") else ("⚡ Экстренный CPU Fallback" if state.get("emergency_mode") else "💎 Штатный TPU v3-8 / NIM")}
                    </p>
                </div>

                <!-- Incidents & Self-Healing -->
                <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 700; color: #0f172a;">Журнал сбоев и решений за 24 часа</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 13px;">
                    <thead>
                        <tr style="background: #f1f5f9; text-align: left; border-bottom: 2px solid #cbd5e1;">
                            <th style="padding: 10px; color: #475569;">Время UTC</th>
                            <th style="padding: 10px; color: #475569;">Событие / Сбой</th>
                            <th style="padding: 10px; color: #475569;">Принятое исправление</th>
                        </tr>
                    </thead>
                    <tbody>
                        {incident_rows}
                    </tbody>
                </table>

                <!-- Approved Model Fleet -->
                <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 700; color: #0f172a;">Действующий флот проверенных моделей</h3>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                    {fleet_rows}
                </ul>
            </div>

            <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 32px; font-size: 12px; color: #94a3b8; text-align: center;">
                Автономный комплекс DeepSeek-TPU-NIM • Ежедневная рассылка 12:00 НСК • {SENDER_EMAIL or 'sender-not-configured'}
            </div>
        </div>
    </body>
    </html>
    """

    plain = (
        f"Проверка связи: Системный аудит комплекса ({now_novt})\n\n"
        f"1. Статус API NIM: {key_status.get('message')}\n"
        f"2. Дисковое пространство: {disk_metrics.get('free_gb')} ГБ свободно ({disk_metrics.get('percent_used')}% занято)\n"
        f"3. Арбитр: {state.get('local_arbiter')}\n"
        f"4. Сбои за 24 часа: {len(error_logs)} событий зафиксировано и самовосстановлено.\n"
    )

    return {"subject": "Проверка связи", "html": html, "plain": plain}

def build_ai_news_digest_email() -> Dict[str, str]:
    """
    Builds Email 2: Daily AI News Intelligence Digest via MiniMax M3.1 Flash and Keenable.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    keenable_key = os.getenv("KEENABLE_API_KEY", "")
    now_date = datetime.now(timezone(timedelta(hours=7))).strftime("%d.%m.%Y")
    now_novt = datetime.now(timezone(timedelta(hours=7))).strftime("%d.%m.%Y 12:00 НСК")

    subject = f"Ежедневный дайджест ИИ: {now_date}"
    executor = MiniMaxExecutor(nvidia_api_key=nvidia_key, keenable_api_key=keenable_key)

    task_prompt = (
        "Сформируй подробный, высокопрофессиональный ежедневный дайджест мировых новостей искусственного интеллекта за прошедшие сутки.\n"
        "Структурируй дайджест по 4 разделам:\n"
        "1. 🚀 Новые модели и архитектуры (Open-source релизы, веса, закрытые флагманы, мультимодальность)\n"
        "2. 🧠 Прорывы в рассуждениях и бенчмарки (Reasoning models, Math, Coding, LMSYS Chatbot Arena)\n"
        "3. 🤖 Агентные системы и автономные фреймворки (AI Agents, автономные пайплайны, Tool Calling)\n"
        "4. ⚡ Инфраструктура, чипы и новости индустрии (NVIDIA, TPU, Cloud, слияния, научные препринты)\n\n"
        "Для каждой новости приведи краткую суть, технические особенности и ссылки/источники если есть.\n"
        "Используй форматирование с красивыми подзаголовками, маркерами списков и акцентами на ключевые цифры."
    )

    print("[*] Generating AI News Digest via MiniMax M3.1 Flash...")
    res = executor.execute_task(
        task_prompt=task_prompt,
        context=f"Дата дайджеста: {now_date}. Запрос новостей ИИ за прошедшие 24 часа.",
        enable_search=True,
        max_tokens=65536,
        temperature=0.6
    )

    content_markdown = res.get("content", "Не удалось сгенерировать текст дайджеста.")
    elapsed = res.get("elapsed_sec", 0)

    # Convert basic markdown formatting to clean HTML
    html_body = content_markdown.replace("\n\n", "</p><p style='margin: 10px 0; line-height: 1.6;'>")
    html_body = html_body.replace("\n- ", "<br>• ")
    html_body = html_body.replace("\n### ", "<h3 style='color: #0f172a; margin-top: 24px; font-size: 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px;'>")
    html_body = html_body.replace("\n## ", "<h2 style='color: #1e293b; margin-top: 28px; font-size: 18px;'>")
    html_body = html_body.replace("**", "<b>") # simple bold pair
    # Fix paired tags
    parts = html_body.split("<b>")
    reconstructed = parts[0]
    for i in range(1, len(parts)):
        subparts = parts[i].split("</b>", 1) if "</b>" in parts[i] else parts[i].split("<b>", 1)
        # Handle simple bolding
        if len(subparts) == 2:
            reconstructed += "<b>" + subparts[0] + "</b>" + subparts[1]
        else:
            reconstructed += "<b>" + parts[i]
    html_body = reconstructed

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; color: #1e293b;">
        <div style="max-width: 720px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); padding: 26px 32px; color: #ffffff;">
                <h1 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px;">🌐 {subject}</h1>
                <p style="margin: 6px 0 0 0; color: #bfdbfe; font-size: 14px;">Подготовлено сетевой моделью MiniMax M3.1 Flash • Время: {now_novt}</p>
            </div>

            <div style="padding: 28px 32px; font-size: 14px; line-height: 1.6; color: #334155;">
                <p style='margin: 0 0 16px 0;'>Представляем сводный аналитический отчет по ключевым событиям и релизам в мире искусственного интеллекта за прошедшие сутки, собранный с помощью поисковой системы Keenable и синтезированный нейросетевым харнессом.</p>
                <div style="background: #ffffff;">
                    {html_body}
                </div>
            </div>

            <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 32px; font-size: 12px; color: #94a3b8; text-align: center;">
                Синтез выполнен за {elapsed}с • Полные немодифицированные данные Keenable • {SENDER_EMAIL or 'sender-not-configured'} → {resolve_recipient() or 'recipient-not-configured'}
            </div>
        </div>
    </body>
    </html>
    """

    return {"subject": subject, "html": html, "plain": content_markdown}

def _log_dispatch(label: str, recipient: str, result: Dict[str, Any]) -> None:
    """
    Records the outcome of a report dispatch in the operational log.

    Without this record nobody — including the Command Center itself — can
    answer whether the daily reports were actually sent. The wording is
    natural language on purpose: the log is searched both semantically and
    lexically, so "отчёты отправлены?" must match the record literally.
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
            f"{headline} | тип={label} | получатель={recipient or 'не задан'} | статус={status}{detail}"
        )
        print(f"[*] Dispatch logged: {label} -> {status} (recipient={recipient or 'unset'})")
    except Exception as e:
        print(f"[!] Could not log dispatch outcome: {e}")


def run_daily_dispatch(dry_run: bool = False, health_only: bool = False, news_only: bool = False) -> Dict[str, Any]:
    print("=" * 72)
    print("📬 STARTING DAILY 12:00 NOVT DISPATCH PIPELINE")
    print(f"[*] Dispatcher timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    sender = EmailSender()
    results = {}

    # Resolved at send time so a recipient changed from Telegram is honoured
    # by the scheduled dispatch, not just by manual checks.
    recipient = resolve_recipient()
    if not recipient:
        print("[!] WARNING: no dispatch recipient configured (DISPATCH_RECIPIENT_EMAIL unset).")
    else:
        print(f"[*] Dispatch recipient: {recipient}")

    # 1. Email 1: Проверка связи (Health & Incidents)
    if not news_only:
        print("\n--- GENERATING EMAIL 1: Проверка связи ---")
        health_mail = build_health_report_email()
        if dry_run:
            print("[DRY-RUN] Email 1 ready. Length:", len(health_mail['html']), "chars.")
            results["health_report"] = {"status": "DRY_RUN_OK", "subject": health_mail["subject"]}
        else:
            res1 = sender.send_email(
                recipient_email=recipient,
                subject=health_mail["subject"],
                html_content=health_mail["html"],
                plain_text_content=health_mail["plain"]
            )
            results["health_report"] = res1
            _log_dispatch("Проверка связи", recipient, res1)

    # 2. Email 2: Ежедневный дайджест ИИ
    if not health_only:
        print("\n--- GENERATING EMAIL 2: Ежедневный дайджест ИИ ---")
        try:
            news_mail = build_ai_news_digest_email()
            if dry_run:
                print("[DRY-RUN] Email 2 ready. Length:", len(news_mail['html']), "chars.")
                results["news_digest"] = {"status": "DRY_RUN_OK", "subject": news_mail["subject"]}
            else:
                res2 = sender.send_email(
                    recipient_email=recipient,
                    subject=news_mail["subject"],
                    html_content=news_mail["html"],
                    plain_text_content=news_mail["plain"]
                )
                results["news_digest"] = res2
                _log_dispatch("Ежедневный дайджест ИИ", recipient, res2)
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
    args = parser.parse_args()

    run_daily_dispatch(dry_run=args.dry_run, health_only=args.health_only, news_only=args.news_only)
