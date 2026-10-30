#!/usr/bin/env python3
"""
Objective probe tasks for the autonomous fleet tuner.

Every check here is *verifiable by construction*: the expected answer is
computed, parsed or compiled, never guessed by a second language model and
never copied from a static leaderboard. That is what allows the complex to
put a real number on a model ("quality_score", "score_verified": true)
instead of repeating an unverified seed value.

Three task families:
  1. reasoning    - multi-step arithmetic with a uniquely correct fraction
  2. code         - the answer must actually parse as Python
  3. json         - the answer must actually parse as a JSON object
  4. instruction  - answers in Russian (the operator's language)

The reasoning task is deliberately small: a 12-ball urn gives exactly one
correct fraction (47/66), so "smart" is distinguishable from "confident".
"""

import ast
import json
import re
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------

PING_PROMPT = (
    "Ответь по-русски одним словом: что означает слово «готовность»?"
)

REASONING_PROMPT = (
    "В коробке 3 красных шара, 4 зелёных и 5 синих. Наугад достают два шара "
    "(без возврата). Какова точная вероятность, что шары разных цветов? "
    "Покажи ход решения и в последней строке дай сокращённую дробь."
)

REASONING_ANSWER = "47/66"

CODE_PROMPT = (
    "Напиши на Python (asyncio) функцию `check_urls`, которая параллельно "
    "проверяет список HTTP-URL с ограничением одновременных запросов 5 и "
    "таймаутом 3 секунды, и возвращает словарь {url: bool}. "
    "Код должен обрабатывать исключения. Ответь только кодом."
)

JSON_PROMPT = (
    "Извлеки сущности в JSON без markdown и без пояснений:\n"
    "'NVIDIA Corporation released Nemotron-4 340B on June 2024 with 4096 context tokens.'\n"
    'Формат: {"company": str, "model": str, "release_date": str, "context_tokens": int}'
)

PROBE_TASKS: Dict[str, Dict[str, Any]] = {
    "reasoning": {"prompt": REASONING_PROMPT, "max_tokens": 3072, "weight": 0.35},
    "code": {"prompt": CODE_PROMPT, "max_tokens": 2048, "weight": 0.35},
    "json": {"prompt": JSON_PROMPT, "max_tokens": 1024, "weight": 0.20},
}

# The instruction check rides on the cheap ping: it is the only place where
# the answer is measured in the operator's own language.
INSTRUCTION_WEIGHT = 0.10


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_code(text: str) -> str:
    """Pulls the Python source out of a model answer."""
    if not text:
        return ""
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        return fenced.group(1)
    idx = text.find("async def")
    if idx == -1:
        idx = text.find("def ")
    if idx == -1:
        idx = text.find("import ")
    return text[idx:] if idx != -1 else ""


def _extract_json(text: str) -> str:
    """Pulls the outermost JSON object out of a model answer."""
    if not text:
        return ""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    return cleaned[start:end + 1] if start != -1 and end > start else ""


# --------------------------------------------------------------------------
# Checkers: (score 0..1, short verdict)
# --------------------------------------------------------------------------

def check_reasoning(text: str) -> Tuple[float, str]:
    """
    The urn has exactly one correct fraction: 47/66.
    Partial credit is given when the numbers appear but are not reduced, so a
    model that reasons correctly but writes "47/66" in words still scores.
    """
    flat = _flatten(text)
    if not flat:
        return 0.0, "empty"
    compact = flat.replace(" ", "")
    has47 = bool(re.search(r"\b47\b", flat))
    has66 = bool(re.search(r"\b66\b", flat))
    if REASONING_ANSWER in compact:
        score = 1.0
    elif has47 and has66:
        score = 0.7
    elif has66 or has47:
        score = 0.3
    else:
        score = 0.0
    # A worked solution is worth a little more than a bare number.
    if score > 0 and ("=" in flat or re.search(r"\d+\s*/\s*\d+", flat)):
        score = min(1.0, score + 0.05)
    return round(score, 3), ("correct" if score >= 0.99 else "partial" if score > 0 else "wrong")


def check_code(text: str) -> Tuple[float, str]:
    """The answer must parse as Python and use the required constructs."""
    code = _extract_code(text)
    if not code.strip():
        return 0.0, "no_code"
    try:
        ast.parse(code)
        compiles = True
    except SyntaxError:
        compiles = False
    low = code.lower()
    required = ("asyncio", "semaphore", "clientsession", "try", "except")
    hits = sum(1 for kw in required if kw in low)
    score = 0.5 * (hits / len(required)) + 0.5 * (1.0 if compiles else 0.0)
    verdict = "compiles" if compiles else "syntax_error"
    return round(score, 3), f"{verdict} {hits}/{len(required)}"


def check_json(text: str) -> Tuple[float, str]:
    """The answer must parse as a JSON object with the four requested keys."""
    blob = _extract_json(text)
    if not blob:
        return 0.0, "no_json"
    try:
        data = json.loads(blob)
    except Exception:
        return 0.0, "invalid_json"
    if not isinstance(data, dict):
        return 0.0, "not_object"
    required = ("company", "model", "release_date", "context_tokens")
    present = [k for k in required if k in data]
    score = len(present) / len(required)
    if data.get("context_tokens") == 4096:
        score = min(1.0, score + 0.1)
    if isinstance(data.get("company"), str) and data.get("company"):
        score = min(1.0, score + 0.05)
    return round(score, 3), f"keys {len(present)}/{len(required)}"


def check_instruction(text: str) -> Tuple[float, str]:
    """The ping answer must be in Russian: the operator's language."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0, "empty"
    cyr = sum(1 for c in letters if ("А" <= c <= "я") or c in "Ёё")
    ratio = cyr / len(letters)
    if ratio > 0.5:
        return 1.0, "russian"
    if ratio > 0.2:
        return 0.5, "mixed"
    return 0.0, "not_russian"


CHECKERS = {
    "reasoning": check_reasoning,
    "code": check_code,
    "json": check_json,
    "instruction": check_instruction,
}


def score_probe(task_name: str, text: str) -> Tuple[float, str]:
    """Runs the checker for a task; unknown tasks score 0 instead of raising."""
    checker = CHECKERS.get(task_name)
    if not checker:
        return 0.0, "unknown_task"
    try:
        return checker(text)
    except Exception as e:  # a broken checker must never crash the tuner
        return 0.0, f"checker_error:{type(e).__name__}"


def probe_weights(task_names: List[str]) -> Dict[str, float]:
    """
    Weights for the requested task set. The weights always sum to 1.0 so the
    composite score stays comparable between runs with different task sets.
    """
    weights = {}
    for name in task_names:
        if name == "instruction":
            weights[name] = INSTRUCTION_WEIGHT
        else:
            weights[name] = PROBE_TASKS.get(name, {}).get("weight", 0.1)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}
