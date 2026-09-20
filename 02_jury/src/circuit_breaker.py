from __future__ import annotations


class FailureCircuitBreaker:
    """Trips after too many consecutive failures — a strong signal the vLLM
    server itself has died, not just an occasional bad generation. Once
    open, callers should fast-fail remaining work instead of burning the
    full retry/backoff/timeout cycle (minutes each, with --max-retries=5)
    on every request against a server that is no longer there.
    """

    def __init__(self, threshold: int = 15) -> None:
        self._threshold = threshold
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        return self._consecutive_failures >= self._threshold

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
