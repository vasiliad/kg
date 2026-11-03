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
"""

from typing import Any, Dict, Optional

__all__ = ["duty_writer", "local_writer", "pick_writer"]


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


def pick_writer(ask: str, executor: Any, engine: Any) -> Dict[str, Any]:
    """
    The bench: the duty holder writes; the reserve replaces it only when the
    holder stays silent. One voice at a time, and the answer says which.
    """
    duty = duty_writer(ask, executor)
    local: Dict[str, Any] = {}
    writer = duty
    if not duty.get("ok"):
        local = local_writer(ask, engine)
        if local.get("ok"):
            writer = local
    return {
        "writer": writer,
        "local": local,
        "ok": bool(writer.get("ok")),
        "duty_status": duty.get("status"),
    }
