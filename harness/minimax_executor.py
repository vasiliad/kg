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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes web search via Keenable API with rate limiting (<=10 req/s)
        and fallback to DuckDuckGo HTML.
        """
        now = time.time()
        elapsed = now - self._last_search_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_search_time = time.time()

        # 1. Primary: Keenable Search API
        try:
            if self.keenable_api_key:
                k_url = "https://api.keenable.ai/v1/search"
                k_headers = {
                    "X-API-Key": self.keenable_api_key,
                    "Content-Type": "application/json",
                    "X-Keenable-Title": "DeepSeek-TPU-Harness"
                }
            else:
                k_url = "https://api.keenable.ai/v1/search/public"
                k_headers = {
                    "Content-Type": "application/json",
                    "X-Keenable-Title": "DeepSeek-TPU-Harness"
                }

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
        except Exception:
            pass

        # 2. Fallback: Tavily Search API (key via TAVILY_API_KEY env)
        try:
            tavily_key = os.getenv("TAVILY_API_KEY", "")
            if tavily_key:
                res = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": tavily_key, "query": query, "max_results": num_results},
                    timeout=12
                )
                if res.status_code == 200:
                    results = []
                    for item in res.json().get("results", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", "")
                        })
                    if results:
                        return results
        except Exception:
            pass

        # 3. Fallback: DuckDuckGo HTML
        try:
            import re
            encoded = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code == 200:
                snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL)
                links = re.findall(r'<a class="result__url[^>]*href="([^"]+)"', res.text)
                results = []
                for l, s in zip(links[:num_results], snippets[:num_results]):
                    clean_s = re.sub(r'<[^>]+>', '', s).strip()
                    results.append({"title": query, "url": l.strip(), "snippet": clean_s})
                return results
        except Exception:
            pass

        return []

    def execute_task(
        self,
        task_prompt: str,
        context: str = "",
        enable_search: bool = True,
        max_tokens: int = 131072,
        temperature: float = 0.6,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Executes delegated task via streaming to prevent timeouts.
        Captures dual-stream: internal reasoning_content and final content.
        Supports full long-context budget from 128k (131,072) up to 256k (262,144).
        """
        start_time = time.time()
        citations = []

        # 1. Enrich with Keenable Search if requested
        enriched_context = context
        if enable_search:
            search_query = task_prompt[:120]
            print(f"[MiniMax Harness] Pre-fetching live web intel via Keenable: '{search_query}'...")
            search_results = self.search_web(search_query, num_results=3)
            if search_results:
                citations = [r.get("url") for r in search_results if r.get("url")]
                full_items = []
                for r in search_results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snip = r.get("snippet", "").strip()
                    full_items.append(f"Source: {title} ({url})\nFull Content:\n{snip}")

                search_text = "\n\n".join(full_items)
                enriched_context += f"\n\n--- LIVE WEB FACTS (KEENABLE) ---\n{search_text}"

        system_prompt = (
            "You are MiniMax M3.1 Flash, the high-speed tactical execution agent in the Autonomous Complex.\n\n"
            "OPERATIONAL RUNTIME SPECIFICATIONS & SYSTEM MANIFEST:\n"
            "- Architecture: Sovereign Dual-Tier Complex (100% independent, zero reliance on external compute queues).\n"
            "- Total Context Window: 256,000 tokens (Hard empirical ceiling: 262,144 tokens).\n"
            "- Generation Output Budget: Up to 128,000 tokens (131,072 tokens).\n"
            "- Request Rate Pacing: <= 30 RPM (enforced with >= 2.0s intervals via NIM Pacer).\n"
            "- Supreme Arbiter: Qwen/Qwen2.5-1.5B-Instruct on local CPU (SovereignCPUEngine).\n"
            "- Persistent Long-Term Memory: ACTIVE via NVIDIA Nemotron 3.5 Embed and data/telegram_history.txt.\n"
            "- Inter-Model Channel: Direct and reverse links with the Lead Arbiter and Telegram Command Center are ACTIVE.\n\n"
            "When asked about your context window, environment limits, or supervisor/arbiter, cite these exact technical specifications without hesitation. "
            "Execute assigned tasks with rigorous logical reasoning and deliver complete, actionable responses."
        )

        user_content = f"Directive from Autonomous Command / Arbiter:\n{task_prompt}"
        if enriched_context.strip():
            user_content += f"\n\nContext & References:\n{enriched_context}"

        # Respect the caller's requested budget, capped by the 256k hard ceiling.
        # Keep a floor of 2560 so reasoning models never exhaust the budget mid-CoT.
        total_chars = len(system_prompt) + len(user_content)
        estimated_prompt_tokens = int(total_chars / 3.2) + 256
        hard_ceiling = 262144 - estimated_prompt_tokens - 256

        effective_max_tokens = max(min(int(max_tokens), hard_ceiling), 2560)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": effective_max_tokens,
            "stream": True
        }

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

            if resp is None or resp.status_code != 200:
                err = resp.text[:300] if resp is not None else "no response"
                code = resp.status_code if resp is not None else 0
                return {
                    "status": "FAILED",
                    "model": self.model_id,
                    "error": f"HTTP {code}: {err}",
                    "elapsed_sec": round(time.time() - start_time, 2)
                }

            full_content = []
            full_reasoning = []

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
                        if c:
                            full_content.append(c)
                        if r:
                            full_reasoning.append(r)
                    except Exception:
                        pass

            duration = round(time.time() - start_time, 2)
            content_str = "".join(full_content).strip()
            reasoning_str = "".join(full_reasoning).strip()

            print(f"[MiniMax Harness] Stream completed successfully in {duration}s! (Content length: {len(content_str)} chars)")

            return {
                "status": "SUCCESS",
                "model": self.model_id,
                "task": task_prompt,
                "reasoning_trace": reasoning_str,
                "content": content_str,
                "elapsed_sec": duration,
                "citations": citations
            }
        except Exception as e:
            return {
                "status": "FAILED",
                "model": self.model_id,
                "task": task_prompt,
                "error": str(e),
                "elapsed_sec": round(time.time() - start_time, 2)
            }
