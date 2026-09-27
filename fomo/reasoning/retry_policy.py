"""Bounded retry helper for transient operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class PermanentFailure(RuntimeError):
    """An operation failure that must not be retried."""


class RetryPolicy:
    def __init__(
        self,
        *,
        max_attempts: int = 3,
        initial_delay: float = 0.2,
        max_delay: float = 2.0,
        retry_if: Callable[[Exception], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= max_attempts <= 8:
            raise ValueError("max_attempts must be between 1 and 8")
        if initial_delay < 0 or max_delay < 0 or initial_delay > max_delay:
            raise ValueError("retry delays must satisfy 0 <= initial_delay <= max_delay")
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_if = retry_if or (lambda error: not isinstance(error, PermanentFailure))
        self.sleep = sleep

    def run(self, operation: Callable[[], T]) -> T:
        delay = self.initial_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as error:
                if attempt == self.max_attempts or not self.retry_if(error):
                    raise
                if delay:
                    self.sleep(delay)
                delay = min(self.max_delay, delay * 2)
        raise AssertionError("unreachable")