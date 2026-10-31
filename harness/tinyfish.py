#!/usr/bin/env python3
"""
TinyFish Search — the complex's primary web-search tier.

The operator supplied a POOL of TinyFish API keys (generous limits: ~30
req/min and 12k req/day per key), so the pool rotates through them
round-robin: one bad or rate-limited key never costs the complex its search
capability, and the combined budget scales with the pool size.

The keys live ONLY in the TINYFISH_API_KEYS GitHub secret (comma-separated)
— this repository is public, so no key may ever appear in the source.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"


class TinyFishSearchPool:
    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            raw = os.getenv("TINYFISH_API_KEYS", "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._keys = [k for k in keys if k]
        self._idx = 0
        self._bad: set = set()
        self._lock = threading.Lock()
        self._last_call = 0.0

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def is_configured(self) -> bool:
        return bool(self._keys)

    def _next_key(self) -> Optional[str]:
        with self._lock:
            if not self._keys:
                return None
            for _ in range(len(self._keys)):
                key = self._keys[self._idx % len(self._keys)]
                self._idx += 1
                if key not in self._bad:
                    return key
            # every key is marked bad: give them one more chance (the outage
            # may have been momentary) instead of losing search entirely
            self._bad.clear()
            return self._keys[self._idx % len(self._keys)]

    def mark_bad(self, key: str) -> None:
        with self._lock:
            self._bad.add(key)

    def _get(self, query: str, key: str, timeout: float) -> Optional[requests.Response]:
        try:
            return requests.get(
                SEARCH_ENDPOINT,
                headers={"X-API-Key": key},
                params={"query": query},
                timeout=timeout,
            )
        except Exception:
            return None

    @staticmethod
    def _parse(res: requests.Response, num_results: int) -> List[Dict[str, Any]]:
        try:
            data = res.json()
        except Exception:
            return []
        out = []
        for item in (data.get("results") or [])[:num_results]:
            if not isinstance(item, dict):
                continue
            out.append({
                "title": item.get("title", "") or item.get("site_name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", "") or item.get("description", ""),
            })
        return out

    def search(self, query: str, num_results: int = 5, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """
        One web search, rotated across the key pool. On a rejected key
        (401/403/429) the next key takes over within the same call.
        Returns the standard {title, url, snippet} shape.
        """
        if not self.is_configured or not (query or "").strip():
            return []

        # polite spacing between pool calls (per-key budget is what matters;
        # the rotation spreads load across keys)
        elapsed = time.time() - self._last_call
        if elapsed < 0.15:
            time.sleep(0.15 - elapsed)
        self._last_call = time.time()

        first_key = self._next_key()
        if not first_key:
            return []
        res = self._get(query, first_key, timeout)
        if res is None or res.status_code in (401, 403, 429):
            if res is not None:
                self.mark_bad(first_key)
            second_key = self._next_key()
            if not second_key or second_key == first_key:
                return []
            res = self._get(query, second_key, timeout)
            if res is None:
                return []
        if res.status_code != 200:
            return []
        return self._parse(res, num_results)


_POOL: Optional[TinyFishSearchPool] = None
_POOL_LOCK = threading.Lock()


def get_tinyfish_pool() -> TinyFishSearchPool:
    """Process-wide pool singleton (keys are read once from the env)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = TinyFishSearchPool()
        return _POOL
