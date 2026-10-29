"""
NVIDIA NIM Rate Limiter & Request Pacer.
Enforces the critical rate limit rule:
- Maximum 30 requests per minute (strictly <= 40 RPM limit).
- Smooth pacing: minimum 2.0 seconds between consecutive requests.
- Burst-free: prevents concurrency spikes that trigger 429 Too Many Requests.
"""
import time
import threading

class NIMPacer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._last_request_time = 0.0
                cls._instance._min_interval = 2.0  # 60s / 30 RPM = 2.0 seconds
                cls._instance._pacer_lock = threading.Lock()
            return cls._instance

    def wait_for_slot(self) -> float:
        """
        Enforces a clean 2.0 second spacing between any NVIDIA NIM API requests.
        Returns the actual time slept (if any).
        """
        with self._pacer_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            slept = 0.0
            if elapsed < self._min_interval:
                slept = self._min_interval - elapsed
                time.sleep(slept)
            self._last_request_time = time.time()
            return slept

# Global singleton instance
nim_pacer = NIMPacer()
