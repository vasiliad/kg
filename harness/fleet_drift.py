#!/usr/bin/env python3
"""
Fleet drift: what changed between two MEASURED cycles.

The tuner re-measures the duty roster every cycle, so the fleet already
adapts to a model that got worse. What it did not do was *remember*:
`data/fleet_report.json` was overwritten each time, so a model that quietly
degraded looked exactly like a model that was always fine. Drift is the
difference between those two states.

Every number here is the difference between two measured scores. Nothing is
inferred from a static seed and nothing is estimated — if there is no
previous cycle, the answer is "FIRST", not a guess.

Statuses:
  FIRST      no previous cycle to compare against
  STABLE     the roster moved by less than the noise floor
  IMPROVING  the roster measured better
  DEGRADED   the roster measured worse — this is the one that matters
  REPLACED   the primary changed without a score collapse
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import atomic_io

# A score is measured on a 0..100 scale. Five points is roughly the spread
# between two neighbouring duty models, so a smaller mean move is noise.
MEAN_DELTA_THRESHOLD = 5.0
# A single model losing this much is a real regression even if the rest of
# the roster holds it up.
MODEL_DELTA_THRESHOLD = 12.0
# How many cycles to keep. At one cycle every six hours this is six days.
MAX_HISTORY_CYCLES = 24

# Operator-facing names. The status codes stay in the data files; the card
# the operator reads in Telegram shows the plain Russian sentence.
STATUS_LABELS = {
    "FIRST": "нет данных для сравнения (первый замер)",
    "RESCALED": "набор проб изменился — баллы разных шкал не сравниваются",
    "STABLE": "ростер в пределах шума",
    "IMPROVING": "ростер измерен лучше",
    "DEGRADED": "ростер измерен хуже",
    "REPLACED": "основной дежурный заменён",
}


def _index(report: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{model_id: measurement} for every MEASURED model in a report."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(report, dict):
        return out
    for m in report.get("measurements") or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        if m.get("status") != "MEASURED":
            continue
        out[m["id"]] = m
    return out


def _quality(measurement: Dict[str, Any]) -> Optional[float]:
    """Mean of the per-task probe scores — the raw quality behind a score."""
    scores = (measurement.get("task_scores") or {}).values()
    values = [float(v) for v in scores if isinstance(v, (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _fleet_ids(report: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report, dict):
        return []
    ids = []
    for f in report.get("fleet") or []:
        if isinstance(f, dict) and f.get("id"):
            ids.append(f["id"])
    return ids


def _primary(report: Optional[Dict[str, Any]]) -> Optional[str]:
    for f in (report or {}).get("fleet") or []:
        if isinstance(f, dict) and "duty_primary" in (f.get("roles") or []):
            return f.get("id")
    return None


def _task_set(report: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The probe set a cycle was measured with, if the report records it."""
    policy = (report or {}).get("policy") or {}
    tasks = policy.get("quality_tasks")
    if isinstance(tasks, list) and tasks:
        return [str(t) for t in tasks]
    return None


def compare_cycles(previous: Optional[Dict[str, Any]], current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares two measured cycles. Never raises: a broken report yields a
    "FIRST" answer rather than taking the tuner down with it.
    """
    prev = _index(previous)
    curr = _index(current)
    prev_fleet = _fleet_ids(previous)
    curr_fleet = _fleet_ids(current)

    if not prev or not curr:
        return {
            "status": "FIRST",
            "mean_score_delta": None,
            "worst": None,
            "added": sorted(set(curr_fleet) - set(prev_fleet)),
            "removed": sorted(set(prev_fleet) - set(curr_fleet)),
            "per_model": {},
            "note": "нет предыдущего замера для сравнения",
        }

    per_model: Dict[str, Dict[str, Any]] = {}
    deltas: List[float] = []
    worst: Optional[Dict[str, Any]] = None
    for mid, m in curr.items():
        old = prev.get(mid)
        if not old:
            continue
        old_score = old.get("final_score")
        new_score = m.get("final_score")
        if not isinstance(old_score, (int, float)) or not isinstance(new_score, (int, float)):
            continue
        delta = round(float(new_score) - float(old_score), 2)
        old_q = _quality(old)
        new_q = _quality(m)
        per_model[mid] = {
            "score": new_score,
            "previous_score": old_score,
            "delta": delta,
            "quality": new_q,
            "quality_delta": (round(new_q - old_q, 4)
                              if (old_q is not None and new_q is not None) else None),
            "roles": m.get("roles") or [],
        }
        deltas.append(delta)
        if worst is None or delta < worst["delta"]:
            worst = {"id": mid, "delta": delta}

    added = sorted(set(curr_fleet) - set(prev_fleet))
    removed = sorted(set(prev_fleet) - set(curr_fleet))
    mean_delta = round(sum(deltas) / len(deltas), 2) if deltas else None

    # A different probe set means a different scale. The composite is a
    # weighted mean over the probes, so adding one moves every score — the
    # delta would look like an improvement that never happened. Say so
    # instead of measuring the scale change.
    prev_tasks = _task_set(previous)
    curr_tasks = _task_set(current)
    rescaled = bool(prev_tasks and curr_tasks and prev_tasks != curr_tasks)

    if rescaled:
        status = "RESCALED"
    elif mean_delta is not None and mean_delta <= -MEAN_DELTA_THRESHOLD:
        status = "DEGRADED"
    elif worst is not None and worst["delta"] <= -MODEL_DELTA_THRESHOLD:
        status = "DEGRADED"
    elif mean_delta is not None and mean_delta >= MEAN_DELTA_THRESHOLD:
        status = "IMPROVING"
    elif _primary(previous) != _primary(current):
        status = "REPLACED"
    else:
        status = "STABLE"

    return {
        "status": status,
        "mean_score_delta": mean_delta,
        "worst": worst,
        "added": added,
        "removed": removed,
        "per_model": per_model,
        "compared_models": len(per_model),
        "previous_tasks": prev_tasks,
        "current_tasks": curr_tasks,
        "note": {
            "FIRST": "нет предыдущего замера для сравнения",
            "RESCALED": "набор проб изменён — баллы разных шкал не сравниваются",
            "STABLE": "ростер в пределах шума",
            "IMPROVING": "ростер измерен лучше",
            "DEGRADED": "ростер измерен хуже",
            "REPLACED": "основной дежурный заменён",
        }[status],
    }


def load_history(path: str) -> List[Dict[str, Any]]:
    """The recorded cycles. A broken file yields an empty history, never a crash."""
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_cycle(path: str, report: Dict[str, Any], max_cycles: int = MAX_HISTORY_CYCLES) -> List[Dict[str, Any]]:
    """
    Records one cycle and keeps the history bounded. The cycle is compact on
    purpose: the full report is a separate file, this is only the trend line.
    """
    cycle = {
        "tuned_at": report.get("tuned_at"),
        "fleet": _fleet_ids(report),
        "primary": _primary(report),
        "tasks": _task_set(report),
        "scores": {
            mid: m.get("final_score")
            for mid, m in _index(report).items()
            if isinstance(m.get("final_score"), (int, float))
        },
    }
    history = load_history(path)
    history.append(cycle)
    if len(history) > max_cycles:
        history = history[-max_cycles:]
    # Atomic: the drift history is the memory the next comparison reads.
    atomic_io.write_json_atomic(path, history)
    return history


def _details(drift: Dict[str, Any]) -> List[str]:
    """The measured numbers behind a drift, as plain parts."""
    parts: List[str] = []
    mean = drift.get("mean_score_delta")
    if isinstance(mean, (int, float)):
        parts.append(f"средний сдвиг {mean:+.2f}")
    worst = drift.get("worst")
    if isinstance(worst, dict) and worst.get("id"):
        parts.append(f"худший: {worst['id']} ({worst.get('delta', 0):+.2f})")
    added = drift.get("added") or []
    removed = drift.get("removed") or []
    if added:
        parts.append(f"добавлено: {', '.join(added[:3])}")
    if removed:
        parts.append(f"выбыло: {', '.join(removed[:3])}")
    return parts


def summarize(drift: Optional[Dict[str, Any]]) -> str:
    """One plain-Russian line for the operator. Never raises."""
    if not isinstance(drift, dict):
        return "Дрейф флота: нет данных."
    status = drift.get("status", "FIRST")
    parts = [f"Дрейф флота: {status}"] + _details(drift)
    return "; ".join(parts) + "."


def describe(drift: Optional[Dict[str, Any]]) -> str:
    """
    The same measurement, said in words instead of status codes — for the
    operator's Telegram card. Never raises.
    """
    if not isinstance(drift, dict):
        return "нет данных."
    status = str(drift.get("status") or "FIRST")
    label = STATUS_LABELS.get(status, str(drift.get("note") or status))
    parts = [label] + _details(drift)
    return "; ".join(parts) + "."
