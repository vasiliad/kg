#!/usr/bin/env python3
"""
LangSearch web-search client — the second tier of the complex's search stack.

The operator supplied an API key (Free plan: ~1M tokens/day, 6 RPS). The key
lives ONLY in the LANGSEARCH_API_KEY GitHub secret — this repository is
public, so no key may ever appear in the source.

API (docs.langsearch.com): POST https://api.langsearch.com/v1/web-search
  headers: Authorization: Bearer <key>
  body:    {"query": str, "count": 1..50}
  answer:  data.webPages.value[] items with name / url / snippet (or text).
"""

import os
from typing import Any, Dict, List, Optional

import requests

SEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"


class LangSearchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.getenv("LANGSEARCH_API_KEY", "")).strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _parse(data: Dict[str, Any], num_results: int) -> List[Dict[str, Any]]:
        pages = ((data or {}).get("data") or {}).get("webPages") or {}
        out = []
        for item in (pages.get("value") or [])[:num_results]:
            if not isinstance(item, dict):
                continue
            out.append({
                "title": item.get("name", "") or "",
                "url": item.get("url", "") or "",
                "snippet": item.get("snippet", "") or item.get("text", "") or "",
            })
        return out

    def search(self, query: str, num_results: int = 5, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """One web search; falls back silently (empty list) on any failure so
        the caller's next search tier takes over."""
        if not self.is_configured or not (query or "").strip():
            return []
        try:
            res = requests.post(
                SEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "count": max(1, min(int(num_results), 50))},
                timeout=timeout,
            )
        except Exception:
            return []
        if res.status_code != 200:
            return []
        try:
            return self._parse(res.json(), num_results)
        except Exception:
            return []


_CLIENT: Optional[LangSearchClient] = None


def get_langsearch_client() -> LangSearchClient:
    """Process-wide client singleton (the key is read once from the env)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LangSearchClient()
    return _CLIENT
