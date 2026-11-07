"""
MiniMax M3.1 Flash Dedicated Worker Harness (Streaming Protocol).
Orchestrated by Supreme Local Systems Arbiter Qwen/Qwen2.5-1.5B on CPU.

Ground-Truth Verified Rules for Minimax M3.1 Flash:
1. Minimax generates extensive internal Chain-of-Thought (CoT) in reasoning_content (1200-1800 tokens).
2. max_tokens budget MUST be >= 2560 (otherwise token limit is hit during thinking, leaving content empty).
3. Streaming (stream=True) is MANDATORY to prevent HTTP socket ReadTimeouts during long reasoning traces.
"""
import os
import sys
import time
import json
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

from harness.key_pool import NVIDIAKeyPool
from harness import search_query
from harness import tool_call

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _streamed_tool_call(chunks: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Builds a tool call out of the streamed `delta.tool_calls` deltas.

    NIM sends the tool name and the arguments in pieces, one delta per
    fragment. These deltas used to be dropped on the floor: the model asked
    for a tool and the harness answered with an empty message.
    """
    name = ""
    arguments = ""
    for chunk in chunks or []:
        parts = chunk if isinstance(chunk, list) else [chunk]
        for part in parts:
            if not isinstance(part, dict):
                continue
            function = part.get("function") or {}
            if function.get("name"):
                name += str(function["name"])
            if function.get("arguments"):
                arguments += str(function["arguments"])
    if not name:
        return None
    return {"tool": name, "arguments": arguments}


class MiniMaxExecutor:
    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        keenable_api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1"
    ):
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY", "")
        if not self.nvidia_api_key:
            env_path = PROJECT_ROOT / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("NVIDIA_API_KEY="):
                                self.nvidia_api_key = line.strip().split("=", 1)[1].strip()
                except Exception:
                    pass

        self.keenable_api_key = keenable_api_key or os.getenv("KEENABLE_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model_id = "minimaxai/minimax-m3.1-flash"
        self.key_pool = NVIDIAKeyPool(self.nvidia_api_key)
        self.nvidia_api_key = self.key_pool.current()
        self._last_search_time = 0.0
        # Which tier actually served the last search. Measured, never assumed:
        # the block of live facts must name its own source.
        self.last_search_tier = ""

    # ---------------------------------------------------------------
    # The search tiers, one method each.
    #
    # Every tier is a separate method so that it can be MEASURED on its own:
    # measured on 2026-11-01, the public Keenable tier answered an English
    # query perfectly and a Russian query with pure noise (LAX APM for
    # «стоимость генетического теста Атлас»). A chain that hides its tiers
    # cannot be diagnosed; a chain of named tiers can.
    # ---------------------------------------------------------------

    def _search_tinyfish(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """TinyFish Search — an 8-key rotating pool (keys live only in the secret)."""
        from harness.tinyfish import get_tinyfish_pool
        return get_tinyfish_pool().search(query, num_results=num_results) or []

    def _search_langsearch(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """LangSearch — the operator's key, ~1M tokens/day."""
        from harness.langsearch import get_langsearch_client
        return get_langsearch_client().search(query, num_results=num_results) or []

    def _search_keenable(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Keenable Search. If the configured key is rejected, the public tier
        is tried too, so that one stale key never costs the complex its
        live-search capability.
        """
        keenable_attempts = []
        if self.keenable_api_key:
            keenable_attempts.append((
                "https://api.keenable.ai/v1/search",
                {
                    "X-API-Key": self.keenable_api_key,
                    "Content-Type": "application/json",
                    "X-Keenable-Title": "DeepSeek-TPU-Harness"
                }
            ))
        keenable_attempts.append((
            "https://api.keenable.ai/v1/search/public",
            {
                "Content-Type": "application/json",
                "X-Keenable-Title": "DeepSeek-TPU-Harness"
            }
        ))

        for k_url, k_headers in keenable_attempts:
            try:
                res = requests.post(
                    k_url,
                    headers=k_headers,
                    json={"query": query, "mode": "realtime"},
                    timeout=12
                )
                if res.status_code == 200:
                    data = res.json()
                    results = []
                    for item in data.get("results", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", "") or item.get("description", "")
                        })
                    if results:
                        return results
                elif res.status_code in (401, 403):
                    print(f"[!] Keenable rejected the request ({res.status_code}) on {k_url.split('/')[-2]}; trying next tier.")
                    continue
            except Exception:
                continue
        return []

    def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Tavily Search API (key via TAVILY_API_KEY env)."""
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key:
            return []
        res = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": tavily_key, "query": query, "max_results": num_results},
            timeout=12
        )
        if res.status_code != 200:
            return []
        results = []
        for item in res.json().get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")
            })
        return results

    def _search_firecrawl(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Firecrawl Search (key via FIRECRAWL_API_KEY env).

        A RESERVE tier by design: the key's free quota is a one-off credit
        grant, so this tier sits AFTER the quota-cheap tiers and BEFORE the
        anonymous last resort — it only burns credits when everything above
        it is silent. Measured 2026-11-07: v2 search answers
        {"success": true, "data": {"web": [{title, url, description}]}}.
        """
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "")
        if not firecrawl_key:
            return []
        res = requests.post(
            "https://api.firecrawl.dev/v2/search",
            headers={
                "Authorization": f"Bearer {firecrawl_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "limit": num_results},
            timeout=15
        )
        if res.status_code != 200:
            print(f"[!] Firecrawl answered HTTP {res.status_code}; trying next tier.")
            return []
        data = res.json().get("data")
        if isinstance(data, dict):
            items = data.get("web") or data.get("results") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        results = []
        for item in items[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", "") or item.get("snippet", "")
            })
        return results

    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo HTML — the last resort, with no key and no quota."""
        import re
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code != 200:
            return []
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL)
        links = re.findall(r'<a class="result__url[^>]*href="([^"]+)"', res.text)
        results = []
        for l, s in zip(links[:num_results], snippets[:num_results]):
            clean_s = re.sub(r'<[^>]+>', '', s).strip()
            results.append({"title": query, "url": l.strip(), "snippet": clean_s})
        return results

    # The order of the chain. The first tier that answers wins.
    # Firecrawl sits AFTER the quota-cheap tiers: its free quota is a
    # one-off grant, so it is a reserve, not a daily driver — and BEFORE
    # the anonymous last resort, which is flakier than a measured API.
    SEARCH_TIERS: tuple = (
        ("TinyFish", "_search_tinyfish"),
        ("LangSearch", "_search_langsearch"),
        ("Keenable", "_search_keenable"),
        ("Tavily", "_search_tavily"),
        ("Firecrawl", "_search_firecrawl"),
        ("DuckDuckGo", "_search_duckduckgo"),
    )

    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Runs the search chain and returns the results of the first tier that
        answered. The tier that served them is remembered in
        `last_search_tier` — the live facts must name their own source.
        """
        now = time.time()
        elapsed = now - self._last_search_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_search_time = time.time()
        self.last_search_tier = ""

        for tier, method_name in self.SEARCH_TIERS:
            try:
                results = getattr(self, method_name)(query, num_results) or []
            except Exception:
                continue
            if not results:
                continue
            # A tier that answers with noise is worse than a tier that is
            # silent: the duty model would cite the noise as a fact.
            stems = search_query.topic_stems(query)
            if stems and not search_query.is_on_topic(results, stems):
                print(f"[Search] {tier} answered off-topic ({len(results)} results); trying next tier.")
                continue
            self.last_search_tier = tier
            print(f"[Search] served by {tier} ({len(results)} results)")
            return results
        return []

    def search_each_tier(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Asks every tier on its own and reports what each one answered. Used
        by the probe to tell a dead tier from a tier that answers in another
        language — the two look identical from the outside.
        """
        outcomes: List[Dict[str, Any]] = []
        for tier, method_name in self.SEARCH_TIERS:
            try:
                results = getattr(self, method_name)(query, num_results) or []
                outcomes.append({"tier": tier, "results": results, "error": ""})
            except Exception as exc:
                outcomes.append({
                    "tier": tier,
                    "results": [],
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return outcomes

    def _search_query_for(self, task_prompt: str, override: str = "") -> str:
        """
        The query the search stack receives. It is formulated from the
        operator's question — the topical core, not a truncated blob of the
        whole prompt.

        An override wins: the bot may have inherited the topic of the
        previous turn («Найди на озон» after a question about a genetic
        test), and re-formulating the prompt here would lose that topic
        again.
        """
        if override and override.strip():
            return override.strip()
        return search_query.formulate(task_prompt)

    def _call_model(
        self,
        target_model: str,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        One streaming call to the duty model.

        Returns {"ok", "content", "reasoning", "error"} and never raises: a
        broken transport is a verdict, not a crash.

        `tools` is the SAME catalogue for every model — the harness offers
        `live_search` and `fetch_webpage` to whoever is on duty, so the
        ability to search is a property of the harness, not of the model.
        A model that rejects the parameter (HTTP 400/422) is retried once
        without it: an old endpoint must not cost the model its answer.
        """
        # Respect the caller's requested budget, capped by the 256k hard ceiling.
        # Keep a floor of 2560 so reasoning models never exhaust the budget mid-CoT.
        total_chars = len(system_prompt) + len(user_content)
        estimated_prompt_tokens = int(total_chars / 3.2) + 256
        hard_ceiling = 262144 - estimated_prompt_tokens - 256

        effective_max_tokens = max(min(int(max_tokens), hard_ceiling), 2560)

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": effective_max_tokens,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"

        from harness.nim_pacer import nim_pacer
        pacer_sleep = nim_pacer.wait_for_slot()
        if pacer_sleep > 0:
            print(f"[NIM Pacer] Paced request (+{pacer_sleep:.2f}s) to maintain <=30 RPM...")

        print(f"[MiniMax Harness] Streaming execution task (max_tokens={effective_max_tokens} / 256k limit)...")
        try:
            # Try each pool key once: on 401/429 rotate to the next key and retry.
            resp = None
            for _ in range(max(len(self.key_pool), 1)):
                headers = {
                    "Authorization": f"Bearer {self.key_pool.current()}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=90)
                if resp.status_code in (401, 403, 429) and len(self.key_pool) > 1:
                    next_key = self.key_pool.mark_bad()
                    print(f"[MiniMax Harness] Key rejected (HTTP {resp.status_code}). Rotating to ...{next_key[-6:]}.")
                    continue
                break

            # The endpoint may not know the `tools` parameter. Losing the
            # tool catalogue is acceptable; losing the answer is not.
            if resp is not None and resp.status_code in (400, 422) and tools:
                print(f"[MiniMax Harness] Endpoint rejected the tool schema (HTTP {resp.status_code}); retrying without it.")
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                headers = {
                    "Authorization": f"Bearer {self.key_pool.current()}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=90)

            if resp is None or resp.status_code != 200:
                err = resp.text[:300] if resp is not None else "no response"
                code = resp.status_code if resp is not None else 0
                return {
                    "ok": False,
                    "error": f"HTTP {code}: {err}",
                    "content": "",
                    "reasoning": "",
                }

            full_content = []
            full_reasoning = []
            streamed_calls: List[Any] = []

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data_body = line_str[6:]
                    if data_body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_body)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content")
                        r = delta.get("reasoning_content")
                        t = delta.get("tool_calls")
                        if c:
                            full_content.append(c)
                        if r:
                            full_reasoning.append(r)
                        if t:
                            streamed_calls.append(t)
                    except Exception:
                        pass

            content_str = "".join(full_content).strip()
            reasoning_str = "".join(full_reasoning).strip()

            # A native tool call (the OpenAI-shaped `delta.tool_calls`) used to
            # be dropped on the floor: the model asked for a tool and the
            # harness answered with an empty message.
            native = _streamed_tool_call(streamed_calls)
            if native and not content_str:
                content_str = json.dumps(native, ensure_ascii=False)

            return {
                "ok": True,
                "content": content_str,
                "reasoning": reasoning_str,
                "error": "",
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "content": "", "reasoning": ""}

    def _fetch_page_text(self, url: str, max_chars: int = 4000) -> str:
        """
        Reads a page as TEXT.

        The page is untrusted data: it is never executed, never imported,
        never written to disk — only stripped down to text and handed to the
        model as something to quote.
        """
        try:
            from harness.tools import HTMLTextExtractor
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AutonomousComplex/1.0)"}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return f"The page could not be read: HTTP {resp.status_code}."
            extractor = HTMLTextExtractor()
            extractor.feed(resp.text)
            text = extractor.get_text().strip()
            return text[:max_chars] if text else "The page returned no readable text."
        except Exception as e:
            return f"The page could not be read: {type(e).__name__}."

    def read_source_text(self, url: str, max_chars: int = 8000) -> str:
        """
        The text of a source page, for the digest.

        Same contract as `_fetch_page_text`: untrusted data, read as text,
        never executed. It is public because the dispatcher needs it — the
        operator asked for the news to be written out IN FULL, and a full
        account can only come from the page itself, not from a search
        snippet.
        """
        return self._fetch_page_text(url, max_chars=max_chars)

    def _run_tool(self, call: Dict[str, Any], num_results: int = 5) -> Dict[str, Any]:
        """
        Runs the tool the duty model asked for.

        The model says WHAT to look for; the harness decides WHERE. The
        provider named by the model («source»: «tinyfish») is ignored — the
        tier chain is ours, and it is measured.
        """
        name = str(call.get("tool") or "").lower()
        try:
            if name in tool_call.SEARCH_TOOLS:
                query = tool_call.tool_query(call)
                if not query:
                    return {"ok": False, "label": name,
                            "text": "The call carried no query.", "citations": []}
                results = self.search_web(query, num_results=num_results)
                tier = getattr(self, "last_search_tier", "") or "unknown"
                if not results:
                    return {
                        "ok": True,
                        "label": f"{name} (source: {tier})",
                        "text": f"No result for the query «{query}». Every search tier was asked.",
                        "citations": [],
                    }
                items = []
                citations = []
                for item in results:
                    title = item.get("title", "")
                    url = item.get("url", "")
                    snippet = str(item.get("snippet", "")).strip()
                    if url:
                        citations.append(url)
                    items.append(f"Source: {title} ({url})\nFull Content:\n{snippet}")
                return {
                    "ok": True,
                    "label": f"{name} (source: {tier})",
                    "text": "\n\n".join(items),
                    "citations": citations,
                }

            if name in tool_call.FETCH_TOOLS:
                url = tool_call.tool_url(call)
                if not url:
                    return {"ok": False, "label": name,
                            "text": "The call carried no URL.", "citations": []}
                return {
                    "ok": True,
                    "label": f"{name} ({url})",
                    "text": self._fetch_page_text(url),
                    "citations": [url],
                }
        except Exception as e:
            return {"ok": False, "label": name,
                    "text": f"The tool failed: {type(e).__name__}: {e}", "citations": []}

        return {
            "ok": False,
            "label": name,
            "text": f"Unknown tool «{name}». The harness can search the web and read a page.",
            "citations": [],
        }

    def execute_task(
        self,
        task_prompt: str,
        context: str = "",
        enable_search: bool = True,
        max_tokens: int = 131072,
        temperature: float = 0.6,
        top_p: float = 0.95,
        model_id: Optional[str] = None,
        search_query: str = "",
        num_results: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes delegated task via streaming to prevent timeouts.
        Captures dual-stream: internal reasoning_content and final content.
        Supports full long-context budget from 128k (131,072) up to 256k (262,144).

        `model_id` routes the call to any duty-team member measured by the
        fleet tuner; the default stays the tactical executor MiniMax M3.1
        Flash. A non-default model gets a neutral duty role instead of the
        MiniMax identity block.

        `search_query` overrides the formulated query — the bot may have
        inherited the topic of the previous turn, and this executor must not
        re-formulate the prompt and lose it again.

        A duty model may answer with a TOOL CALL instead of an answer.
        Measured on 2026-11-01: «Найди на озон» came back as a raw JSON call
        and the operator received "only curly braces". The call is executed
        here, the result goes back into the context, and the model is asked
        again for an answer in prose.
        """
        target_model = (model_id or self.model_id).strip()
        start_time = time.time()
        citations = []

        # 1. Enrich with the live web if requested
        enriched_context = context
        if enable_search:
            query_text = self._search_query_for(task_prompt, search_query)
            print(f"[MiniMax Harness] Pre-fetching live web intel: '{query_text}'...")
            search_results = self.search_web(query_text, num_results=num_results)
            tier = getattr(self, "last_search_tier", "") or "unknown"
            if search_results:
                citations = [r.get("url") for r in search_results if r.get("url")]
                full_items = []
                for r in search_results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snip = r.get("snippet", "").strip()
                    full_items.append(f"Source: {title} ({url})\nFull Content:\n{snip}")

                search_text = "\n\n".join(full_items)
                # The block names the tier that actually served it. A hardcoded
                # label ("KEENABLE") made a TinyFish answer look like a
                # Keenable one — and a dead Keenable made the duty model
                # declare the whole web unavailable.
                enriched_context += f"\n\n--- LIVE WEB FACTS (source: {tier}) ---\n{search_text}"
            else:
                # The web WAS asked and answered nothing. Reporting that as
                # "the live search is unavailable" is a lie the model must not
                # be able to tell.
                enriched_context += (
                    "\n\n--- LIVE WEB FACTS (source: none) ---\n"
                    f"The live web WAS queried for this question (query: '{query_text}') and every "
                    "search tier returned no result. Do NOT describe the live web as unavailable; "
                    "say that the query returned nothing."
                )

        identity_block = (
            f"You are {target_model}, a duty executor model of the Autonomous Complex "
            "called by the keeper (the local CPU chief model).\n"
            if target_model != self.model_id
            else "You are MiniMax M3.1 Flash, the high-speed tactical execution agent in the Autonomous Complex.\n"
        )
        system_prompt = (
            identity_block +
            "\nOPERATIONAL RUNTIME SPECIFICATIONS & SYSTEM MANIFEST:\n"
            "- Architecture: Sovereign Dual-Tier Complex (100% independent, zero reliance on external compute queues).\n"
            "- Total Context Window: 256,000 tokens (Hard empirical ceiling: 262,144 tokens).\n"
            "- Generation Output Budget: Up to 128,000 tokens (131,072 tokens).\n"
            "- Request Rate Pacing: <= 30 RPM (enforced with >= 2.0s intervals via NIM Pacer).\n"
            "- Supreme Arbiter: Qwen/Qwen2.5-1.5B-Instruct on local CPU (SovereignCPUEngine).\n"
            "- Persistent Long-Term Memory: ACTIVE via NVIDIA Nemotron 3.5 Embed and data/telegram_history.txt.\n"
            "- Inter-Model Channel: Direct and reverse links with the Lead Arbiter and Telegram Command Center are ACTIVE.\n\n"
            "IDENTITY AND TRUTHFULNESS RULES:\n"
            "- You ARE the Telegram Command Center itself. There is no separate human operator, no duty "
            "operator, no on-call shift and no escalation chain. Never invent one, never tell the user to "
            "check something themselves, and never ask them to consult a log or a supervisor.\n"
            "- The context contains a 'VERIFIED SYSTEM FACTS' snapshot: operational log lines, the active "
            "fleet and the last diagnostics. Answer questions about what the complex did from that "
            "snapshot, quoting timestamps. If a fact is absent, say so plainly and offer the concrete "
            "action you can take right now.\n"
            "- Never claim an action was performed unless the snapshot or your own output shows it.\n"
            "- Tools: the harness offers you `live_search` (search the live web) and `fetch_webpage` "
            "(read one page). Call them when the answer needs the live web — the harness executes the "
            "call and puts the facts into your context. If you do not need the live web, answer "
            "directly. Your FINAL message must be prose for the operator: never leave a raw tool "
            "call as the answer.\n\n"
            "When asked about your context window, environment limits, or supervisor/arbiter, cite these exact technical specifications without hesitation. "
            "Execute assigned tasks with rigorous logical reasoning and deliver complete, actionable responses."
        )

        # The duty model may answer with a TOOL CALL instead of an answer.
        # Measured on 2026-11-01: «Найди на озон» came back as a raw JSON
        # call and the operator received "only curly braces". The call is
        # executed here, the result goes back into the context, and the model
        # is asked again — for an answer in prose this time.
        tool_rounds = 0
        max_tool_rounds = 2
        executed_tools: List[Dict[str, Any]] = []
        content_str = ""
        reasoning_str = ""

        while True:
            user_content = f"Directive from Autonomous Command / Arbiter:\n{task_prompt}"
            if enriched_context.strip():
                user_content += f"\n\nContext & References:\n{enriched_context}"

            call_result = self._call_model(
                target_model=target_model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                # The same catalogue for every model on duty.
                tools=tool_call.TOOL_SCHEMAS,
            )
            if not call_result.get("ok"):
                return {
                    "status": "FAILED",
                    "model": target_model,
                    "task": task_prompt,
                    "error": call_result.get("error", "unknown error"),
                    "elapsed_sec": round(time.time() - start_time, 2),
                }

            content_str = call_result.get("content", "")
            reasoning_str = call_result.get("reasoning", "")

            requested = tool_call.parse_tool_call(content_str)
            if not requested or tool_rounds >= max_tool_rounds:
                break

            tool_rounds += 1
            outcome = self._run_tool(requested, num_results=num_results)
            executed_tools.append({
                "tool": requested.get("tool"),
                "query": tool_call.tool_query(requested),
                "ok": bool(outcome.get("ok")),
            })
            for url in outcome.get("citations") or []:
                if url and url not in citations:
                    citations.append(url)
            enriched_context += (
                f"\n\n--- TOOL RESULT ({outcome.get('label', requested.get('tool'))}) ---\n"
                f"{outcome.get('text', '')}"
            )
            if not outcome.get("ok"):
                enriched_context += (
                    "\n(The tool could not be executed. Answer without it and say plainly "
                    "what is missing.)"
                )

        # The rounds are spent and the model still wants a tool: the operator
        # gets a sentence, never the JSON.
        unresolved = tool_call.parse_tool_call(content_str)
        if unresolved:
            wanted = tool_call.tool_query(unresolved) or tool_call.tool_url(unresolved) or "—"
            content_str = (
                f"Модель {target_model} вернула вызов инструмента "
                f"({tool_call.describe_call(unresolved)}), но ответ получить не удалось.\n"
                f"Запрос, который она хотела выполнить: {wanted}"
            )

        duration = round(time.time() - start_time, 2)
        print(f"[MiniMax Harness] Stream completed successfully in {duration}s! (Content length: {len(content_str)} chars)")

        return {
            "status": "SUCCESS",
            "model": target_model,
            "task": task_prompt,
            "reasoning_trace": reasoning_str,
            "content": content_str,
            "elapsed_sec": duration,
            "citations": citations,
            "tool_rounds": tool_rounds,
            "tools": executed_tools,
        }
