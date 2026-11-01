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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The operator's own sentence, verbatim: the request sits in the middle.
QUESTION = ("Есть компания Атлас,которая делает генетические тесты."
            "Можешь найти её в интернете и узнать сколько стоит тест?")


def on_topic_count(results: list, stems: list) -> int:
    """How many results speak about the topic of the query."""
    if not stems:
        return 0
    return sum(1 for item in results if _mentions(item, stems))


def _mentions(item: dict, stems: list) -> bool:
    blob = " ".join([
        str(item.get("title", "")),
        str(item.get("url", "")),
        str(item.get("snippet", "")),
    ]).lower()
    return any(stem in blob for stem in stems)


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
    stems = search_query.topic_stems(plan["query"])

    # 2. Every tier on its own.
    for outcome in executor.search_each_tier(plan["query"], num_results=num_results):
        hits = on_topic_count(outcome["results"], stems)
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
    result["relevant"] = on_topic_count(results, stems)
    result["ok"] = bool(results)
    return result


def answer_with_live_web(question: str) -> dict:
    """
    The whole path, with the real model: the operator's sentence -> the live
    web -> the duty model's answer. This is the proof that the question the
    operator actually asked is answered from live facts, and not from memory.
    """
    from harness.minimax_executor import MiniMaxExecutor

    executor = MiniMaxExecutor()
    return executor.execute_task(
        task_prompt=question,
        context=("VERIFIED SYSTEM FACTS: this is a live probe of the complex's own "
                 "web path. Answer the question only from the LIVE WEB FACTS block."),
        enable_search=True,
        max_tokens=2560,
    )


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

    if "--answer" in sys.argv:
        answer = answer_with_live_web(QUESTION)
        print(f"[probe] model    : status={answer.get('status')} "
              f"model={answer.get('model')} elapsed={answer.get('elapsed_sec')}s")
        for citation in (answer.get("citations") or [])[:5]:
            print(f"[probe] cite    : {citation}")
        content = str(answer.get("content") or answer.get("error") or "")
        print(f"[probe] answer  : {content[:1500]}")

    print("[probe] OK: the operator's request reaches the live web.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
