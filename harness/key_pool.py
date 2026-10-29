"""
NVIDIA NIM API Key Pool with automatic rotation.
Loads NVIDIA_API_KEY plus NVIDIA_API_KEY_2..NVIDIA_API_KEY_9 from the environment
and advances to the next key on 401 (revoked/rotated) or 429 (rate limit), so a
single dead key can never take the autonomous complex down.
"""
import os
from typing import List, Optional


class NVIDIAKeyPool:
    def __init__(self, primary: Optional[str] = None):
        candidates = [primary or "", os.getenv("NVIDIA_API_KEY", "")]
        for i in range(2, 10):
            candidates.append(os.getenv(f"NVIDIA_API_KEY_{i}", ""))
        seen, pool = set(), []
        for key in candidates:
            key = (key or "").strip()
            if key and key not in seen:
                seen.add(key)
                pool.append(key)
        self._pool: List[str] = pool
        self._idx = 0

    def current(self) -> str:
        return self._pool[self._idx] if self._pool else ""

    def __len__(self) -> int:
        return len(self._pool)

    def mark_bad(self) -> str:
        """Advances to the next key after a 401/429. Returns the new current key."""
        if len(self._pool) > 1:
            self._idx = (self._idx + 1) % len(self._pool)
        return self.current()
