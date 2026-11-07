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


def _hours_since(text: Any) -> Optional[float]:
    """How many hours ago a stamp was (None when it cannot be read)."""
    from harness.state_clock import parse_ts

    stamp = parse_ts(text)
    if stamp is None:
        return None
    delta = (_now() - stamp).total_seconds() / 3600.0
    return round(max(0.0, delta), 2)


def attempt_freshness(verdict: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """
    When the ATTEMPT happened — in one honest Russian line.

    Measured 2026-11-06: the artefact carried a fresh `checked_at` next to
    an eight-hour-old `attempted_at`, and the whole record then read like
    the result of the noon cycle although nothing had been tried since
    04:09. A heartbeat stamp must never look like a result, so the attempt's
    own time and age are always named.
    """
    moment = now or _now()
    attempted = str(verdict.get("attempted_at") or "")
    from harness.state_clock import parse_ts

    stamp = parse_ts(attempted)
    if stamp is None:
        return "Последняя попытка: время неизвестно (отметка не читается)."
    age_h = round(max(0.0, (moment - stamp).total_seconds() / 3600.0), 1)
    when = stamp.strftime("%d.%m %H:%M UTC")
    if age_h < 1:
        return f"Последняя попытка: {when} ({round(age_h * 60)} мин назад)."
    return f"Последняя попытка: {when} ({age_h} ч назад)."


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

    # The duty shift is a FACT with a life of its own, and the facts digest
    # must say whether that life is still running. Measured 2026-11-06: the
    # digest called a shift that had expired four days earlier «Текущая
    # смена» — and the complex wrote its self-improvement from that dead
    # fact. The shift now speaks for itself: live, broken or expired.
    shift = _read_json(PROJECT_ROOT / "data" / "duty_shift.json")
    if shift.get("model_id"):
        try:
            from harness import duty_shift as _duty_shift
            lines.append(_duty_shift.shift_status_line(shift))
        except Exception:
            lines.append(
                f"Дежурная смена: {shift.get('model_id')}, "
                f"задержка {shift.get('last_latency_sec')}с, "
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

    # The GROUNDED proposal: the writer names a REAL file of this repository,
    # sees its real content, and writes the diff; the constitution judges.
    # Measured 2026-11-03 in the cloud: a proposal to a file that does not
    # exist, with a hunk header that did not match its own body, killed the
    # whole attempt — the complex asked for something it had never made
    # possible.
    proposal = forge_writers.propose_patch(
        task=reason,
        executor=executor,
        engine=engine,
        gate=code_forge.apply_self_improvement,
        facts=facts,
    )

    for key in (
        "ok", "stage", "branch", "error", "summary", "writer", "chooser",
        "duty_status", "local_status", "target", "normalised", "notes",
        "repair", "answer_preview",
    ):
        if key in proposal:
            verdict[key] = proposal[key]

    outcome = proposal.get("outcome") or {}
    verdict["gate"] = {
        "ok": outcome.get("ok"),
        "stage": outcome.get("stage"),
        "branch": outcome.get("branch"),
        "error": outcome.get("error"),
    }

    # The proposal itself is kept: the operator must be able to read WHAT
    # was proposed, not only that something was.
    verdict["_diff"] = str(proposal.get("diff") or "")
    verdict["patch_bytes"] = len(verdict["_diff"].encode("utf-8"))
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
        #
        # But the heartbeat stamp must NOT look like a result (measured
        # 2026-11-06: a fresh `checked_at` sat next to an eight-hour-old
        # `attempted_at`, and the record read like the noon cycle even
        # though nothing had been tried since 04:09). So the attempt's own
        # time and age are written next to it, and the verdict says plainly
        # that THIS pass tried nothing.
        previous["checked_at"] = now.isoformat()
        previous["checked_kind"] = "check_only"
        previous["skipped_today"] = True
        previous["last_attempt_at"] = str(previous.get("attempted_at") or "")
        previous["attempt_age_hours"] = _hours_since(previous.get("attempted_at"))
        previous["freshness"] = attempt_freshness(previous, now=now)
        ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTEFACT.write_text(json.dumps(previous, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return {
            "skipped": True,
            "attempt_day": day,
            "verdict_id": previous.get("verdict_id"),
            "checked_at": now.isoformat(),
            "checked_kind": "check_only",
            "last_attempt_at": previous.get("last_attempt_at"),
            "attempt_age_hours": previous.get("attempt_age_hours"),
            "summary": (
                "Сегодня попытка уже была — одна в сутки, без бесконечного "
                f"цикла. {previous['freshness']} Этот проход — только проверка: "
                "новой попытки не было."
            ),
        }

    verdict = attempt()
    verdict["attempt_day"] = day
    verdict["checked_at"] = now.isoformat()
    verdict["checked_kind"] = "attempt"
    verdict["verdict_id"] = now.strftime("%Y%m%dT%H%M%SZ")
    verdict.setdefault("summary", "Попытка без вывода.")
    # The attempt's own time and age are recorded from the first second of
    # the attempt — the artefact then always says WHEN the work happened,
    # not only when the file was last touched.
    verdict["last_attempt_at"] = str(verdict.get("attempted_at") or now.isoformat())
    verdict["attempt_age_hours"] = _hours_since(verdict.get("attempted_at"))
    verdict["freshness"] = attempt_freshness(verdict, now=now)

    # The patch itself is kept next to the verdict: the operator must be
    # able to read WHAT the complex proposed, not only that it proposed.
    # It is stored as a VALID patch: LF endings and a final newline. The
    # stored file is what the operator (and the next cycle) may apply, so
    # it must not carry the defect that killed the gate on 2026-11-06.
    diff_text = str(verdict.pop("_diff", "") or "")
    if diff_text:
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        patch_path = PATCH_DIR / f"{verdict['verdict_id']}.patch"
        stored = diff_text.replace("\r\n", "\n").rstrip("\n")
        patch_path.write_bytes((stored + "\n").encode("utf-8"))
        try:
            rel = str(patch_path.relative_to(PROJECT_ROOT))
        except Exception:
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
