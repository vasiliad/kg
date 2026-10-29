"""
Agent Tools Environment for DeepSeek-R1-32B on TPU.
Provides the local model with real actions:
- web_search(query)
- fetch_webpage(url)
- query_nim_catalog()
- test_nim_model(model_id, temperature, top_p, max_tokens, system_prompt, test_prompt)
"""
import requests
import json
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Dict, Any, List

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ["script", "style", "nav", "footer", "header"]:
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ["script", "style", "nav", "footer", "header"]:
            self.skip = False

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.result.append(data.strip())

    def get_text(self):
        return " ".join(self.result)

class AgentToolbox:
    def __init__(self, nvidia_api_key: str):
        self.nvidia_api_key = nvidia_api_key
        self.headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }

    def web_search(self, query: str, num_results: int = 5) -> str:
        """
        Searches the live web via DuckDuckGo HTML without external paid keys.
        """
        print(f"[TOOL: web_search] Searching web for: '{query}'...")
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                return f"Search error HTTP {res.status_code}"
            
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL)
            links = re.findall(r'<a class="result__url[^>]*href="([^"]+)"', res.text)
            
            results = []
            for l, s in zip(links[:num_results], snippets[:num_results]):
                clean_s = re.sub(r'<[^>]+>', '', s).strip()
                results.append({"link": l.strip(), "snippet": clean_s})
            
            if not results:
                return "No search results found. Try rephrasing query."
            return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Search exception: {e}"

    def fetch_webpage(self, url: str, max_chars: int = 4000) -> str:
        """
        Fetches official documentation or benchmark page content and extracts clean text.
        """
        print(f"[TOOL: fetch_webpage] Reading documentation: {url}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                return f"Failed to fetch page. HTTP status: {res.status_code}"
            
            parser = HTMLTextExtractor()
            parser.feed(res.text)
            text = parser.get_text()
            return text[:max_chars]
        except Exception as e:
            return f"Fetch exception: {e}"

    def query_nim_catalog(self) -> str:
        """
        Queries live NVIDIA NIM /v1/models endpoint to get the raw list of all currently active models.
        """
        print("[TOOL: query_nim_catalog] Querying live NVIDIA NIM endpoints...")
        url = "https://integrate.api.nvidia.com/v1/models"
        try:
            res = requests.get(url, headers=self.headers, timeout=15)
            if res.status_code == 401:
                return "ERROR 401: Unauthorized. The NVIDIA API key is expired or revoked (needs rotation)."
            if res.status_code != 200:
                return f"ERROR HTTP {res.status_code}: {res.text}"
            data = res.json()
            model_ids = [m.get("id") for m in data.get("data", [])]
            return json.dumps({"active_model_count": len(model_ids), "model_ids": model_ids}, indent=2)
        except Exception as e:
            return f"Catalog exception: {e}"

    def test_nim_model(
        self,
        model_id: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: str,
        test_prompt: str,
        reasoning_effort: int = 100
    ) -> str:
        """
        Sends a test request with custom parameters configured by the local agent.
        Supports DeepSeek-specific reasoning_effort percentage (0-100).
        """
        print(f"[TOOL: test_nim_model] Executing live test on '{model_id}' (T={temperature}, P={top_p}, effort={reasoning_effort}%)...")
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_prompt}
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens)
        }
        # If testing DeepSeek or reasoning models, pass reasoning_effort in percentage (0 to 100)
        if "deepseek" in model_id.lower() or "r1" in model_id.lower() or reasoning_effort is not None:
            payload["reasoning_effort"] = int(reasoning_effort)
            # Some NIM deployments accept chat_template_kwargs or extra_body
            payload["extra_body"] = {"reasoning_effort": int(reasoning_effort)}
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if res.status_code != 200:
                return json.dumps({"status": f"FAILED_HTTP_{res.status_code}", "error": res.text})
            
            data = res.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return json.dumps({
                "status": "SUCCESS",
                "response_preview": content[:400],
                "tokens_used": usage.get("total_tokens", 0)
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "EXCEPTION", "error": str(e)})
    def probe_nim_context_window(self, model_id: str) -> str:
        """
        Actively probes the exact server-side context window limit of the model on NVIDIA NIM.
        """
        print(f"[TOOL: probe_nim_context_window] Probing exact NIM context limit for '{model_id}'...")
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1000000
        }
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=20)
            data = res.json()
            details = data.get("details", {})
            if "limit" in details:
                limit_tokens = details["limit"]
                return json.dumps({
                    "model_id": model_id,
                    "exact_nim_context_limit": limit_tokens,
                    "approx_k": f"{round(limit_tokens / 1024)}k",
                    "source": "nvidia_nim_server_probe"
                })
            elif "message" in data:
                return json.dumps({
                    "model_id": model_id,
                    "server_message": data["message"],
                    "source": "nvidia_nim_response"
                })
            return json.dumps({"model_id": model_id, "raw_response": str(data)[:300]})
        except Exception as e:
            return json.dumps({"model_id": model_id, "probe_error": str(e)})

    def probe_reasoning_mode(self, model_id: str) -> str:
        """
        Empirically probes how the model handles thinking/reasoning mode:
        1. Is thinking ON by default? (presence of <think> tokens or reasoning_content field).
        2. Can thinking be disabled or minimized? (testing reasoning_effort=0 or chat_template_kwargs).
        3. Is thinking forced on max by default?
        """
        print(f"[TOOL: probe_reasoning_mode] Probing reasoning behavior for '{model_id}'...")
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        test_q = "How many 'r's are in strawberry? Answer in one word."
        
        try:
            res_default = requests.post(url, headers=self.headers, json={
                "model": model_id,
                "messages": [{"role": "user", "content": test_q}],
                "max_tokens": 512
            }, timeout=25)
            
            if res_default.status_code != 200:
                return json.dumps({"model_id": model_id, "status": f"HTTP_{res_default.status_code}", "error": res_default.text})
            
            body_default = res_default.json()
            choice = body_default.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            reasoning_content = msg.get("reasoning_content", "")
            has_think_tags = "<think>" in content or bool(reasoning_content)
            
            # Test B: Attempt to toggle / adjust reasoning effort (0% vs 100%)
            res_zero = requests.post(url, headers=self.headers, json={
                "model": model_id,
                "messages": [{"role": "user", "content": test_q}],
                "max_tokens": 512,
                "reasoning_effort": 0,
                "chat_template_kwargs": {"thinking": False}
            }, timeout=25)
            
            can_disable = False
            if res_zero.status_code == 200:
                msg_zero = res_zero.json().get("choices", [{}])[0].get("message", {})
                content_zero = msg_zero.get("content", "")
                reasoning_zero = msg_zero.get("reasoning_content", "")
                if "<think>" not in content_zero and not reasoning_zero:
                    can_disable = True
            
            return json.dumps({
                "model_id": model_id,
                "thinking_on_by_default": has_think_tags,
                "can_disable_thinking": can_disable,
                "default_behavior": "MAX_THINKING_FORCED" if has_think_tags and not can_disable else "CONFIGURABLE",
                "sample_content_preview": content[:150],
                "has_separate_reasoning_field": bool(reasoning_content)
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"model_id": model_id, "probe_error": str(e)})
