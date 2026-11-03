#!/usr/bin/env python3
"""
Who writes the code — the duty holder, or the reserve when the network dies.

The logic lives HERE, not in the Telegram bot, because the forge has two
callers: the operator's order (through the bot) and the complex's OWN
attempt to improve itself (without an order). A second copy of the routing
would be a second truth — and the routing is exactly where the complex is
allowed to lie ("the model wrote it" when nobody wrote it).

The rules are unchanged and measured:
  - the duty holder of the shift writes first;
  - if the network tier is silent, the RESERVE LOCAL model writes;
  - the deterministic kernel CANNOT write programs and says so plainly.
    A program that nobody wrote is never presented as a program.

The PROPOSAL PATH is grounded, and the grounding was learned the hard way.
Measured 2026-11-03 in the cloud: the model proposed a patch to
`autopilot/model_selection_policy.py` — a directory and a file that do not
exist in this repository — and the hunk header did not match its own body
(`@@ -12,7 +12,7 @@` over six lines), so git called the patch «corrupt».
The gate was right to refuse it; the complex was wrong to ask for it. A
proposal that CANNOT be made possible is a proposal that never happens, so
the path now:
  1. shows the writer the REAL inventory and asks WHICH file to improve;
  2. shows the writer the EXACT content of that file — a unified diff whose
     context lines were guessed does not apply;
  3. repairs hunk headers whose counts do not match the body (mechanical —
     the gate still judges the result);
  4. allows ONE repair round when the gate says the patch does not apply
     (the project's rule: one retry per ring, no endless loops).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "duty_writer", "local_writer", "pick_writer",
    "choose_target", "propose_patch",
]


def duty_writer(ask: str, executor: Any) -> Dict[str, Any]:
    """The holder of the duty shift writes the code."""
    from harness.duty_team import DutyTeam
    from harness import duty_shift

    try:
        team = DutyTeam()
        roster = team.load_roster()
        shift, _ = duty_shift.current_duty(roster)
    except Exception as e:
        return {"ok": False, "status": f"ROSTER_ERROR:{type(e).__name__}", "content": ""}

    model_id = str(shift.get("model_id") or "")
    if not model_id:
        return {"ok": False, "status": "NO_DUTY_MODEL", "content": "", "shift": shift}
    entry = next(
        (m for m in (roster.get("duty") or []) if str(m.get("id")) == model_id), {}
    )
    try:
        res = executor.execute_task(
            task_prompt=ask,
            context="",
            enable_search=False,
            max_tokens=8192,
            temperature=0.2,
            top_p=0.9,
            model_id=model_id,
        )
    except Exception as e:
        return {"ok": False, "status": f"EXECUTOR_ERROR:{type(e).__name__}",
                "model": model_id, "content": "", "shift": shift}

    content = str(res.get("content") or "")
    elapsed = res.get("elapsed_sec")
    ok = res.get("status") == "SUCCESS" and bool(content.strip())
    duty_shift.note_outcome(
        model_id, elapsed, ok,
        measured_latency=(entry.get("measured") or {}).get("latency_sec")
        or entry.get("latency_sec"),
    )
    return {
        "ok": ok, "model": model_id, "role": "duty_primary",
        "score": entry.get("score"), "verified": entry.get("verified"),
        "elapsed": elapsed, "content": content,
        "status": res.get("status"), "shift": shift,
    }


def local_writer(ask: str, engine: Any) -> Dict[str, Any]:
    """
    The RESERVE LOCAL model writes the code when the network is gone.
    The deterministic kernel cannot write programs — and says so.
    """
    try:
        sov = engine.execute_sovereign_mission(
            task_description=ask,
            context={"mode": "forge_order"},
        )
    except Exception as e:
        return {"ok": False, "status": f"LOCAL_ENGINE_ERROR:{type(e).__name__}", "content": ""}

    engine_name = str(sov.get("engine") or "")
    text = str(sov.get("output") or "").strip()
    if engine_name == "deterministic_kernel":
        return {"ok": False, "status": "DETERMINISTIC_KERNEL_CANNOT_WRITE_CODE",
                "model": "локальное ядро (детерминированное)", "content": "",
                "elapsed": sov.get("elapsed_sec")}
    return {
        "ok": bool(text), "model": f"локальная модель ({engine_name})",
        "role": "sovereign_local", "score": None, "verified": False,
        "elapsed": sov.get("elapsed_sec"), "content": text,
        "status": sov.get("status"),
    }


def pick_writer(
    ask: str,
    executor: Any,
    engine: Any,
    duty: Optional[Callable[[str], Dict[str, Any]]] = None,
    local: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    The bench: the duty holder writes; the reserve replaces it only when the
    holder stays silent. One voice at a time, and the answer says which.

    The two writers can be passed IN (the Telegram bot has its own wrappers
    around them) — the routing stays HERE, so there is still one truth about
    who wrote what.
    """
    duty_fn = duty if duty is not None else (lambda a: duty_writer(a, executor))
    local_fn = local if local is not None else (lambda a: local_writer(a, engine))
    duty_result = duty_fn(ask)
    local_result: Dict[str, Any] = {}
    writer = duty_result
    if not duty_result.get("ok"):
        local_result = local_fn(ask)
        if local_result.get("ok"):
            writer = local_result
    return {
        "writer": writer,
        "local": local_result,
        "ok": bool(writer.get("ok")),
        "duty_status": duty_result.get("status"),
    }


def choose_target(answer: str, paths: List[str]) -> str:
    """
    Which REAL file did the writer name?

    The answer must match the inventory: a file outside the inventory does
    not exist, and a proposal to a file that does not exist is dead on
    arrival (measured 2026-11-03 in the cloud).
    """
    if not answer or not paths:
        return ""
    text = str(answer)
    for rel in sorted(paths, key=len, reverse=True):
        if rel in text:
            return rel
    by_name: Dict[str, List[str]] = {}
    for rel in paths:
        by_name.setdefault(Path(rel).name, []).append(rel)
    for name, rels in by_name.items():
        if name and name in text and len(rels) == 1:
            return rels[0]
    return ""


def _preview(text: str, limit: int = 800) -> str:
    """What the writer actually said — kept, masked, short."""
    out = str(text or "").strip()[:limit]
    try:
        from harness import redaction

        out = redaction.mask_emails_deep(out)
    except Exception:
        pass
    return out


def _run_gate(gate: Callable[[str], Dict[str, Any]], diff: str) -> Dict[str, Any]:
    """The gate is the judge — and a broken gate is a recorded failure."""
    try:
        return gate(diff) or {}
    except Exception as e:
        return {"ok": False, "stage": "gate_error", "error": f"{type(e).__name__}: {e}"}


def propose_patch(
    task: str,
    executor: Any,
    engine: Any,
    gate: Callable[[str], Dict[str, Any]],
    facts: str = "",
    duty: Optional[Callable[[str], Dict[str, Any]]] = None,
    local: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    The GROUNDED proposal: choose a real file, write a diff for it with the
    file's real content in hand, and let the constitution judge.

    Measured 2026-11-03 in the cloud: the model proposed a patch to
    `autopilot/model_selection_policy.py` — a file that does not exist — with
    a hunk header that did not match its own body. The gate refused it, and
    the complex lost the day. Grounding only makes the proposal POSSIBLE;
    the gate still decides whether it is TRUE.

    The path:
      1. the writer sees the REAL inventory and names ONE file to improve;
      2. the writer sees that file's EXACT content and writes the diff —
         a unified diff whose context lines were guessed does not apply;
      3. the hunk headers whose counts do not match the body are repaired
         (mechanical — git recomputes them anyway);
      4. ONE repair round if the gate says the patch does not apply
         (the project's rule: one retry per ring, no endless loops).
    """
    from harness import code_forge

    verdict: Dict[str, Any] = {
        "ok": False,
        "stage": "started",
        "diff": "",
        "target": "",
        "writer": {},
        "duty_status": None,
        "local_status": None,
        "repair": {"attempted": False, "ok": False},
        "normalised": False,
        "notes": [],
    }

    paths = code_forge.inventory_paths()
    inventory = code_forge.repo_inventory()
    if not paths:
        verdict.update(
            stage="no_inventory",
            error="в репозитории нет ни одного файла — улучшать нечего",
            summary="Репозиторий пуст — предлагать нечего.",
        )
        return verdict

    # --- 1. WHICH real file? ---------------------------------------------
    choose_ask = (
        code_forge.PATCH_TARGET_PROMPT
        + "\n\nРеальные файлы комплекса (других не существует):\n" + inventory
        + "\n\nЧто нужно улучшить:\n" + str(task)
        + (("\n\nЧто измерено в комплексе сейчас:\n" + facts) if facts else "")
    )
    picked = pick_writer(choose_ask, executor, engine, duty=duty, local=local)
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
            summary="Ни одна модель не выбрала файл — улучшать нечего.",
        )
        return verdict

    chooser = picked.get("writer") or {}
    answer = str(chooser.get("content") or "")
    verdict["chooser"] = {
        "model": chooser.get("model"),
        "role": chooser.get("role"),
        "elapsed": chooser.get("elapsed"),
        "status": chooser.get("status"),
    }
    if "NO_PATCH" in answer.upper():
        verdict.update(
            stage="no_patch",
            answer_preview=_preview(answer),
            summary="Модель честно отказалась: улучшать нечего.",
        )
        return verdict

    target = choose_target(answer, paths)
    if not target:
        verdict.update(
            stage="no_target",
            answer_preview=_preview(answer),
            summary="Модель не назвала ни одного РЕАЛЬНОГО файла комплекса.",
        )
        return verdict
    verdict["target"] = target

    # --- 2. the diff — with the file's REAL content in hand --------------
    body = code_forge.read_target_file(target)
    patch_ask = (
        code_forge.SELF_IMPROVEMENT_PROMPT
        + "\n\nТы улучшаешь ФАЙЛ: " + target
        + "\n\nЕго ТЕКСТ (строки контекста патча должны совпадать с ним):\n"
        + "```\n" + body + "\n```"
        + "\n\nРеальные файлы комплекса (только они существуют):\n" + inventory
        + "\n\nЧто нужно улучшить:\n" + str(task)
        + (("\n\nЧто измерено в комплексе сейчас:\n" + facts) if facts else "")
    )
    picked = pick_writer(patch_ask, executor, engine, duty=duty, local=local)
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
            summary="Ни одна модель не написала патч — улучшать нечего.",
        )
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
    content = str(writer.get("content") or "")
    diff = code_forge.extract_diff(content)
    if not diff:
        verdict.update(
            stage="no_patch",
            answer_preview=_preview(content),
            summary="Модель ответила без патча — улучшать нечего.",
        )
        return verdict

    diff, notes = code_forge.normalize_diff(diff)
    verdict["notes"] = list(notes)
    verdict["normalised"] = bool(notes)
    outcome = _run_gate(gate, diff)
    verdict["outcome"] = outcome

    # --- 3. ONE repair round when the patch does not apply ---------------
    if not outcome.get("ok") and str(outcome.get("stage")) in ("apply_check", "apply"):
        verdict["repair"] = {"attempted": True, "ok": False}
        repair_ask = (
            code_forge.PATCH_REPAIR_PROMPT.format(
                error=str(outcome.get("error") or "")[:500]
            )
            + "\n\nФайл: " + target
            + "\n\nЕго ТЕКСТ:\n```\n" + body + "\n```"
            + "\n\nТвой прежний патч:\n```diff\n" + diff + "\n```"
        )
        repicked = pick_writer(repair_ask, executor, engine, duty=duty, local=local)
        if repicked.get("ok"):
            repaired_content = str((repicked.get("writer") or {}).get("content") or "")
            repaired = code_forge.extract_diff(repaired_content)
            if not repaired:
                verdict["repair"]["status"] = "no_patch"
            else:
                repaired, repair_notes = code_forge.normalize_diff(repaired)
                if repair_notes:
                    verdict["notes"] = list(verdict.get("notes") or []) + list(repair_notes)
                if repaired == diff:
                    verdict["repair"]["status"] = "same_patch"
                else:
                    diff = repaired
                    outcome = _run_gate(gate, diff)
                    verdict["outcome"] = outcome
                    verdict["repair"]["ok"] = bool(outcome.get("ok"))
                    verdict["repair"]["status"] = outcome.get("stage")
                    verdict["repair"]["model"] = (repicked.get("writer") or {}).get("model")
        else:
            verdict["repair"]["status"] = repicked.get("duty_status") or "no_writer"

    verdict["diff"] = diff
    verdict["ok"] = bool(outcome.get("ok"))
    verdict["stage"] = str(outcome.get("stage") or "gate")
    verdict["branch"] = outcome.get("branch")
    verdict["error"] = outcome.get("error")
    if verdict["ok"]:
        verdict["summary"] = (
            f"Патч прошёл весь офлайн-suite; ветка {verdict.get('branch')} "
            "ждёт оператора (в main не вошло, наружу не отправлено)."
        )
    else:
        verdict["summary"] = (
            f"Патч отклонён конституцией на этапе «{verdict.get('stage')}»: "
            f"{verdict.get('error') or 'без причины'}"
        )
    return verdict
