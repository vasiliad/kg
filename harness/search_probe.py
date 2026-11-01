#!/usr/bin/env python3
"""
Proof that the live web is really reachable — and that the planner asks it.

Measured in the cloud on 2026-11-01: the operator asked for a company and
its price («Есть компания Атлас… Можешь найти её в интернете и узнать
сколько стоит тест?»), and the answer came from memory. The capability was
there — TinyFish and LangSearch were healthy — but the planner never asked,
because the request sat in the middle of the sentence and the old rule only
looked at the front of it.

Three things can break that promise independently, so this probe measures
all three and fails loudly when one of them does:

  1. the planner decides "search" for the operator's own sentence;
  2. every tier is asked on its own, so a tier that answers in another
     language is not mistaken for a dead tier (and the other way round);
  3. the production chain really returns results for the formulated query.

It sends nothing to Telegram, writes nothing, and prints no key.
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The operator's own sentence, verbatim: the request sits in the middle.
QUESTION = ("Есть компания Атлас,которая делает генетические тесты."
            "Можешь найти её в интернете и узнать сколько стоит тест?")

# Words that carry no topic, and must not be counted as a match.
_NOISE = {
    "есть", "можешь", "найти", "интернете", "узнать", "сколько", "стоит",
    "тест", "тесты", "компания", "которая", "делает", "какая", "какой",
    "которое", "который", "please", "the", "and", "for",
}


def topic_words(query: str) -> list:
    """The content words of the query — what a relevant answer must mention."""
    words = re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", query or "")
    return [w.lower() for w in words if w.lower() not in _NOISE]


def relevance(results: list, words: list) -> int:
    """How many results mention a topic word of the query."""
    if not words:
        return 0
    hits = 0
    for item in results:
        blob = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        if any(w in blob for w in words):
            hits += 1
    return hits


def probe(quest: str = QUESTION, num_results: int = 3) -> dict:
    """Runs the planner, every tier and the production chain."""
    from harness import search_query
    from harness.minimax_executor import MiniMaxExecutor

    plan = search_query.plan(quest)
    result = {
        "question": quest,
        "search": plan["search"],
        "reason": plan["reason"],
        "query": plan["query"],
        "tiers": [],
        "tier": "",
        "results": [],
        "relevant": 0,
        "ok": False,
    }
    if not plan["search"]:
        return result

    executor = MiniMaxExecutor()
    words = topic_words(plan["query"])

    # 2. Every tier on its own.
    for outcome in executor.search_each_tier(plan["query"], num_results=num_results):
        hits = relevance(outcome["results"], words)
        result["tiers"].append({
            "tier": outcome["tier"],
            "count": len(outcome["results"]),
            "relevant": hits,
            "error": outcome["error"],
            "sample": [str(i.get("title", ""))[:70] for i in outcome["results"][:2]],
        })

    # 3. The production chain.
    results = executor.search_web(plan["query"], num_results=num_results)
    result["tier"] = getattr(executor, "last_search_tier", "") or "none"
    result["results"] = results
    result["relevant"] = relevance(results, words)
    result["ok"] = bool(results)
    return result


def main() -> int:
    result = probe()
    print(f"[probe] question : {result['question']}")
    print(f"[probe] planner  : search={result['search']} reason={result['reason']}")
    print(f"[probe] query    : {result['query']}")
    if not result["search"]:
        print("[probe] FAIL: the planner did not ask the live web.")
        return 1

    print("[probe] tiers:")
    for tier in result["tiers"]:
        detail = f"{tier['count']} results, {tier['relevant']} mentioning the topic"
        if tier["error"]:
            detail = f"error {tier['error']}"
        print(f"[probe]   - {tier['tier']}: {detail}")
        for sample in tier["sample"]:
            print(f"[probe]       · {sample}")

    print(f"[probe] chain    : tier={result['tier']} "
          f"({len(result['results'])} results, {result['relevant']} on topic)")
    for item in result["results"][:3]:
        title = str(item.get("title", ""))[:90]
        url = str(item.get("url", ""))[:140]
        print(f"[probe]   - {title} :: {url}")

    if not result["results"]:
        print("[probe] FAIL: every live-search tier returned nothing.")
        return 1
    if result["relevant"] == 0:
        print("[probe] WARNING: the web answered, but nothing mentions the topic — "
              "the query language or the tier index is the problem.")
    print("[probe] OK: the operator's request reaches the live web.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
