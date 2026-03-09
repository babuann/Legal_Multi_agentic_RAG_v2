import logging
import threading
import time
from collections import deque
from functools import wraps
from typing import Any, Callable

from google.api_core.exceptions import ResourceExhausted
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int = 14, window_seconds: int = 60) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def acquire(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._evict_expired(now)

            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return 0.0

            wait = self.window_seconds - (now - self._calls[0])
            return max(wait, 0.0)

    def wait_and_acquire(self) -> None:
        while True:
            wait = self.acquire()
            if wait == 0.0:
                return
            logger.info("Rate limit reached. Sleeping %.2fs …", wait + 0.1)
            time.sleep(wait + 0.1)


_limiter = SlidingWindowRateLimiter()


def get_limiter() -> SlidingWindowRateLimiter:
    return _limiter


def llm_call_with_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _limiter.wait_and_acquire()
        return func(*args, **kwargs)

    return wrapper
