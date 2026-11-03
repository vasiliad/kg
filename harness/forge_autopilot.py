#!/usr/bin/env python3
"""
The complex tries to improve ITSELF — without an order.

The operator asked whether the complex can write programs «сама» (by
itself) as well as «по приказу заказчика». The order path exists. This is
the second half: an attempt nobody asked for.

The rules of the attempt are the same rules the operator's order obeys:

  1. a MODEL proposes the patch — the duty holder of the shift, or the
     reserve local model when the network tier is silent. The deterministic
     kernel cannot write programs and says so instead of inventing one;
  2. the patch is vetted (no secrets) and applied ONLY in a scratch clone
     of the COMMITTED tree;
  3. the FULL offline suite must be green in that clone — the constitution
     is the judge, not the model's opinion;
  4. only then a `forge/*` branch appears — never `main`, and never pushed
     outward (the operator's decision);
  5. the attempt is written down: the verdict and the patch itself land in
     `data/forge_autopilot/`, so the operator can read the proposal on his
     phone and decide. A proposal nobody can read is a proposal nobody
     accepted — and nobody rejected.

The attempt is grounded in MEASURED facts (the fleet's live scores, the
drift, the measured latencies), not in the model's imagination: an
improvement with no measured weakness behind it is noise.

One attempt per UTC day by default: an unbounded self-improvement loop is
a self-DoS, and the complex has a rule against exactly that (one retry per
ring, no infinite rings).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import code_forge
from harness import forge_writers

ARTEFACT = PROJECT_ROOT / "data" / "forge_autopilot.json"
PATCH_DIR = PROJECT_ROOT / "data" / "forge_autopilot"

# The reason the attempt is grounded in: the complex looks at what was
# MEASURED to be weak, and asks for one minimal patch against that.
DEFAULT_REASON = (
    "Улучшить работу комплекса по ИЗМЕРЕННЫМ слабым местам: "
    "исправь узкое место, которое замеры уже показали."
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def read_verdict() -> Dict[str, Any]:
    """The last verdict, as written down."""
    return _read_json(ARTEFACT)


def build_facts() -> str:
    """
    The measured facts the proposal must be grounded in.

    Only AGGREGATES go to the model: model ids, scores, latencies and
    counters. No chat ids, no addresses, no error text — the prompt goes
    to an external provider.
    """
    lines: list[str] = []

    report = _read_json(PROJECT_ROOT / "data" / "fleet_report.json")
    drift = report.get("drift") or {}
    if drift:
        worst = drift.get("worst") or {}
        lines.append(
            f"Дрейф флота: {drift.get('status')}, средняя дельта {drift.get('mean_score_delta')}, "
            f"худший: {worst.get('id')} (дельта {worst.get('delta')})"
        )
    for member in (report.get("fleet") or [])[:5]:
        if not isinstance(member, dict):
            continue
        measured = member.get("measured") or {}
        tasks = measured.get("task_scores") or {}
        weakest = ""
        if isinstance(tasks, dict) and tasks:
            weak_id = min(tasks, key=lambda k: tasks[k])
            weakest = f", слабейшая задача {weak_id}={tasks[weak_id]}"
        lines.append(
            f"- {member.get('id')}: балл {member.get('quality_score')}, "
            f"задержка {measured.get('latency_sec')}с, ok_rate {measured.get('ok_rate')}{weakest}"
        )

    shift = _read_json(PROJECT_ROOT / "data" / "duty_shift.json")
    if shift.get("model_id"):
        lines.append(
            f"Текущая смена: {shift.get('model_id')}, задержка {shift.get('last_latency_sec')}с, "
            f"выбрана: {shift.get('chosen_by')}"
        )

    sessions = _read_json(PROJECT_ROOT / "data" / ".bot_sessions.json")
    health = sessions.get("health") or {}
    if health:
        lines.append(
            f"Здоровье бота: запусков {health.get('runs')}, ошибок {health.get('errors')}, "
            f"ответов {health.get('messages_sent')}"
        )

    return "\n".join(lines) if lines else "Измерений пока нет — улучшай то, что замеряешь сам."


def _executor() -> Any:
    from harness.minimax_executor import MiniMaxExecutor

    return MiniMaxExecutor()


def _engine() -> Any:
    from harness.sovereign_cpu_engine import SovereignCPUEngine

    return SovereignCPUEngine()


def attempt(
    reason: str = DEFAULT_REASON,
    executor: Optional[Any] = None,
    engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    One attempt: propose → vet → apply in scratch → the whole offline suite.

    Returns the verdict. Never raises: a failed attempt is a RECORDED
    outcome, not a crash.
    """
    started = _now()
    verdict: Dict[str, Any] = {
        "attempted_at": started.isoformat(),
        "reason": reason,
        "ok": False,
        "stage": "started",
    }

    executor = executor if executor is not None else _executor()
    engine = engine if engine is not None else _engine()

    facts = build_facts()
    verdict["facts_digest"] = facts[:1200]
    ask = (
        code_forge.SELF_IMPROVEMENT_PROMPT
        + "\n\nЧто реально измерено в комплексе сейчас:\n"
        + facts
        + "\n\nЗадача автопилота:\n"
        + reason
    )

    # 1. Who writes: the duty holder, or the reserve when the network is gone.
    try:
        picked = forge_writers.pick_writer(ask, executor, engine)
    except Exception as e:
        verdict.update(stage="writer_error", error=f"{type(e).__name__}: {e}")
        return verdict

    writer = picked.get("writer") or {}
    verdict["writer"] = {
        "model": writer.get("model"),
        "role": writer.get("role"),
        "score": writer.get("score"),
        "verified": writer.get("verified"),
        "elapsed": writer.get("elapsed"),
        "status": writer.get("status"),
    }
    verdict["duty_status"] = picked.get("duty_status")
    verdict["local_status"] = (picked.get("local") or {}).get("status")

    if not picked.get("ok"):
        verdict.update(
            stage="no_writer",
            error=(
                "сетевая дежурная: "
                f"{picked.get('duty_status') or 'нет ответа'}; "
                "локальная резервная: "
                f"{(picked.get('local') or {}).get('status') or 'нет ответа'}"
            ),
            summary="Ни одна модель не предложила патч — попытка честно пустая.",
        )
        return verdict

    # 2. The proposal must BE a diff.
    content = str(writer.get("content") or "")
    diff = code_forge.extract_diff(content)
    if not diff:
        verdict.update(
            stage="no_patch",
            summary="Модель ответила без патча — улучшать нечего.",
        )
        return verdict

    # The proposal itself is kept: the operator must be able to read WHAT
    # was proposed, not only that something was.
    verdict["_diff"] = diff

    # 3. The gate: vet → scratch clone → the WHOLE offline suite → branch.
    try:
        outcome = code_forge.apply_self_improvement(diff)
    except Exception as e:
        verdict.update(stage="gate_error", error=f"{type(e).__name__}: {e}")
        return verdict

    verdict["gate"] = {
        "ok": outcome.get("ok"),
        "stage": outcome.get("stage"),
        "branch": outcome.get("branch"),
        "error": outcome.get("error"),
    }
    verdict["ok"] = bool(outcome.get("ok"))
    verdict["stage"] = str(outcome.get("stage") or "gate")
    verdict["branch"] = outcome.get("branch")
    verdict["patch_bytes"] = len(diff.encode("utf-8"))

    if outcome.get("ok"):
        verdict["summary"] = (
            f"Патч прошёл весь офлайн-suite; ветка {outcome.get('branch')} "
            "ждёт оператора (в main не вошло, наружу не отправлено)."
        )
    else:
        verdict["summary"] = (
            f"Патч отклонён конституцией на этапе «{outcome.get('stage')}»: "
            f"{outcome.get('error') or 'без причины'}"
        )
    return verdict


def run_daily(force: bool = False) -> Dict[str, Any]:
    """
    The daily attempt: at most ONE per UTC day, the verdict written down.

    The artefact is the deliverable: the bot reads it and tells the
    operator what the complex decided about itself.
    """
    now = _now()
    day = now.strftime("%Y-%m-%d")
    previous = _read_json(ARTEFACT)
    if not force and previous.get("attempt_day") == day:
        # The cycle still CHECKS every day — only the attempt is rationed.
        # The check stamps the artefact so the supervisor sees a live cycle
        # and does not dispatch a no-op every six hours.
        previous["checked_at"] = now.isoformat()
        previous["skipped_today"] = True
        ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTEFACT.write_text(json.dumps(previous, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return {
            "skipped": True,
            "attempt_day": day,
            "verdict_id": previous.get("verdict_id"),
            "checked_at": now.isoformat(),
            "summary": "Сегодня попытка уже была — одна в сутки, без бесконечного цикла.",
        }

    verdict = attempt()
    verdict["attempt_day"] = day
    verdict["checked_at"] = now.isoformat()
    verdict["verdict_id"] = now.strftime("%Y%m%dT%H%M%SZ")
    verdict.setdefault("summary", "Попытка без вывода.")

    # The patch itself is kept next to the verdict: the operator must be
    # able to read WHAT the complex proposed, not only that it proposed.
    diff_text = str(verdict.pop("_diff", "") or "")
    if diff_text:
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        patch_path = PATCH_DIR / f"{verdict['verdict_id']}.patch"
        patch_path.write_bytes(diff_text.replace("\r\n", "\n").encode("utf-8"))
        try:
            rel = str(patch_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(patch_path)
        verdict["patch_file"] = rel.replace("\\", "/")
        verdict["patch_bytes"] = len(diff_text.encode("utf-8"))

    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


def main() -> None:
    force = "--force" in sys.argv
    verdict = run_daily(force=force)
    print("=" * 72)
    print("🛠  АВТОПИЛОТ КУЗНИЦЫ: попытка улучшить себя")
    print("=" * 72)
    print(json.dumps(verdict, ensure_ascii=False, indent=2)[:4000])
    print("=" * 72)
    if verdict.get("skipped"):
        print("SKIPPED:", verdict.get("summary"))
        return
    if verdict.get("ok"):
        print("VERDICT: предложение прошло конституцию — ждёт оператора.")
    else:
        print("VERDICT: улучшения нет —", verdict.get("summary"))


if __name__ == "__main__":
    main()
