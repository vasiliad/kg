"""
MiniMax M3.1 Flash Dedicated Worker Harness (Streaming Protocol).
Orchestrated by Lead Systems Architect DeepSeek-R1-32B on Kaggle TPU.

Ground-Truth Verified Rules for Minimax M3.1 Flash:
1. Minimax generates extensive internal Chain-of-Thought (CoT) in reasoning_content (1200-1800 tokens).
2. max_tokens budget MUST be >= 2560 (otherwise token limit is hit during thinking, leaving content empty).
3. Streaming (stream=True) is MANDATORY to prevent HTTP socket ReadTimeouts during long reasoning traces.
"""
import os
import time
import json
import urllib.parse
from typing import Dict, Any, Optional, List
import requests

class MiniMaxExecutor:
    def __init__(
        self,
        nvidia_api_key: str,
        keenable_api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1"
    ):
        self.nvidia_api_key = nvidia_api_key
        self.keenable_api_key = keenable_api_key or os.getenv("KEENABLE_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model_id = "minimaxai/minimax-m3.1-flash"
        self.headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }
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

        # 2. Fallback: DuckDuckGo HTML
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
        max_tokens: int = 3072,
        temperature: float = 0.6,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Executes delegated task via streaming to prevent timeouts.
        Captures dual-stream: internal reasoning_content and final content.
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
            "You are MiniMax M3.1 Flash, a high-speed execution agent working under the direction "
            "of Lead Systems Architect DeepSeek-R1. Execute the assigned task with rigorous logical analysis. "
            "Use the provided live web context when applicable. Deliver a complete, clear, and actionable final response."
        )

        user_content = f"Task Assignment from DeepSeek-R1:\n{task_prompt}"
        if enriched_context.strip():
            user_content += f"\n\nContext & References:\n{enriched_context}"

        effective_max_tokens = max(int(max_tokens), 2560)
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

        print(f"[MiniMax Harness] Streaming execution task (max_tokens={effective_max_tokens})...")
        try:
            resp = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=90)
            if resp.status_code != 200:
                return {
                    "status": "FAILED",
                    "model": self.model_id,
                    "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
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
