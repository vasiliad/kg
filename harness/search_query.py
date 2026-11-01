#!/usr/bin/env python3
"""
Search query formulation.

The executor used to search with the first 120 characters of whatever the
operator typed. A question like «что нового в NVIDIA NIM за последние дни и
какие модели добавили в каталог?» therefore reached the search engine as one
long truncated blob: the intent words ("что нового", "какие") crowd out the
terms that actually carry the topic, and the cut can land mid-word.

This module turns the operator's question into the query the search stack
should actually send, and decides whether a question needs the live web at
all. It is deterministic — no model call, no guesswork — and every rule is
covered by the offline suite.

The operator's own explicit query (`/search <query>`) is never rewritten:
a query the human typed is already a query. Only the automatic pre-fetch for
a free-form question is formulated.
"""

import re
from typing import Any, Dict, List, Optional

# The command the operator states to the bot ("найди в интернете: ...") is an
# instruction, not a topic — it must not travel to the search engine.
INTENT_PREFIXES: tuple = (
    "найди в интернете",
    "поищи в интернете",
    "поиск в интернете",
    "найди в сети",
    "поищи",
    "загугли",
    "lookup",
    "search for",
    "search:",
    "google",
)

# The question frame: "что нового в X" carries the topic in X, not in the
# frame. The frame is stripped only from the front — inside the sentence it
# may be the very word that matters.
QUESTION_FRAMES: tuple = (
    "что нового в",
    "что нового про",
    "что нового по",
    "новости про",
    "новости о",
    "новости по",
    "что изменилось в",
    "что происходит с",
    "что известно о",
    "что известно про",
    "информация о",
    "информация про",
    "что такое",
    "кто такой",
    "кто такая",
    "чем отличается",
    "чем отличаются",
    "как работает",
    "как устроен",
    "как устроена",
    "расскажи про",
    "расскажи о",
)

# Politeness and meta-words: they carry no topic, only noise.
FILLER_WORDS: tuple = (
    "пожалуйста",
    "please",
    "спасибо",
    "спасибо большое",
    "мне",
    "нам",
    "у меня",
    "у нас",
    "хочу знать",
    "хочу узнать",
    "нужно",
    "надо",
    "дай",
    "дайте",
    "расскажи",
    "расскажите",
    "скажи",
    "скажите",
    "объясни",
    "объясните",
    "и ещё",
    "и еще",
    "а также",
    "кстати",
    "pls",
)

# A question that only makes sense against the live web.
RECENCY_MARKERS: tuple = (
    "нов",           # новости, нового, новый, новая
    "последн",
    "свеж",
    "сегодня",
    "вчера",
    "на этой неделе",
    "в этом месяце",
    "в этом году",
    "текущ",
    "актуальн",
    "цена",
    "курс",
    "вышел",
    "вышла",
    "вышли",
    "релиз",
    "по состоянию на",
    "latest",
    "current",
    "today",
    "news",
    "price",
    "release",
)

# Questions about our own complex are answered by our own state — the live
# web cannot know who is on duty or what the last report said.
LOCAL_MARKERS: tuple = (
    "дежурн",
    "ростер",
    "флот",
    "статус системы",
    "отчёт",
    "отчет",
    "почта",
    "email",
    "хранитель",
    "комплекс",
    "бот",
)

MAX_QUERY_LEN = 120

# A leftover preposition is grammar, not topic: "про is_palindrome" carries
# no more meaning than "is_palindrome".
LEADING_PREPOSITIONS: tuple = ("про", "о", "об", "от", "по", "для")


def _strip_prefix(text: str, prefixes: tuple) -> str:
    """Removes the first matching prefix (case-insensitive) from the front."""
    low = text.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            return text[len(prefix):].lstrip(" :,.—–-")
    return text


def _drop_fillers(text: str) -> str:
    """Removes politeness and meta-words wherever they sit."""
    for word in FILLER_WORDS:
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,;:.—–-")


def _strip_leading_preposition(text: str) -> str:
    """Removes the prepositions left behind by the frame/filler rules."""
    changed = True
    while changed:
        changed = False
        for prep in LEADING_PREPOSITIONS:
            if text.lower().startswith(prep + " "):
                text = text[len(prep) + 1:].lstrip()
                changed = True
                break
    return text


def formulate(prompt: str, max_len: int = MAX_QUERY_LEN) -> str:
    """
    The topical core of the operator's question — what the search engine
    should actually receive. Never raises and never returns an empty query
    for a non-empty question.
    """
    raw = (prompt or "").strip()
    if not raw:
        return ""

    text = _strip_prefix(raw, INTENT_PREFIXES)
    text = _drop_fillers(text)
    text = _strip_prefix(text, QUESTION_FRAMES)
    text = _strip_leading_preposition(text)
    text = text.strip(" \t\n?!.«»\"'")
    text = re.sub(r"\s+", " ", text).strip(" ,;:.—–-")

    if not text:
        # Nothing topical survived — better a clipped original than a query
        # that says nothing.
        text = re.sub(r"\s+", " ", raw).strip(" ,;:.—–-")

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].strip(" ,;:.—–-")
    return text


def has_intent(prompt: str) -> bool:
    """The operator explicitly asked for the live web."""
    low = (prompt or "").strip().lower()
    return any(low.startswith(prefix) for prefix in INTENT_PREFIXES)


def needs_search(prompt: str) -> bool:
    """
    Whether this question has to be answered against the live web. A question
    about our own complex is answered by our own state, so it never costs a
    search.
    """
    text = (prompt or "").strip()
    if not text:
        return False
    if has_intent(text):
        return True
    low = text.lower()
    if any(marker in low for marker in LOCAL_MARKERS):
        return False
    return any(marker in low for marker in RECENCY_MARKERS)


def plan(prompt: str) -> Dict[str, Any]:
    """
    What the search layer should do with this question: the query to send,
    whether the live web is needed at all, and why (for the log).
    """
    query = formulate(prompt)
    search = needs_search(prompt)
    if search:
        reason = "оператор запросил интернет" if has_intent(prompt) else "вопрос о свежих данных"
    else:
        reason = "вопрос отвечается собственным состоянием комплекса"
    return {"query": query, "search": search, "reason": reason, "raw": (prompt or "").strip()}
