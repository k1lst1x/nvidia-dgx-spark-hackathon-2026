"""Small bounded in-process sliding-window limits for the demo API.

These limits protect one API process. A multi-worker deployment should use a
shared limiter at the gateway, but callers can never select the peer/global
keys enforced here.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


class LimitExceeded(RuntimeError):
    def __init__(self, detail: str, retry_after: int = 1) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


class SlidingWindowLimiter:
    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float,
        minimum_interval: float = 0,
        max_keys: int = 4096,
        detail: str = "Rate limit exceeded",
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.minimum_interval = minimum_interval
        self.max_keys = max_keys
        self.detail = detail
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._prune(current)
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    raise LimitExceeded("Rate limiter capacity exceeded", 5)
                bucket = deque()
                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)
            if bucket and current - bucket[-1] < self.minimum_interval:
                raise LimitExceeded("Frames are arriving too quickly", max(1, int(self.minimum_interval)))
            if len(bucket) >= self.limit:
                retry = max(1, int(self.window_seconds - (current - bucket[0])))
                raise LimitExceeded(self.detail, retry)
            bucket.append(current)

    def _prune(self, now: float) -> None:
        for key, bucket in list(self._buckets.items()):
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if not bucket:
                del self._buckets[key]

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._buckets)


@dataclass(frozen=True)
class LimitRule:
    limit: int
    window_seconds: float
    minimum_interval: float = 0
    max_keys: int = 4096
    detail: str = "Rate limit exceeded"


class TieredSlidingWindowLimiter:
    """Atomically checks and records several independent rate-limit tiers."""

    def __init__(self, **rules: LimitRule) -> None:
        self.rules = rules
        self._buckets = {name: OrderedDict() for name in rules}
        self._lock = threading.Lock()

    def check(self, *checks: tuple[str, str], now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            for name, _key in checks:
                self._prune(name, current)
            # Validate every tier before appending to any tier. A rejection can
            # therefore never leave partial charges in another bucket.
            for name, key in checks:
                rule = self.rules[name]
                store = self._buckets[name]
                bucket = store.get(key)
                if bucket is None:
                    if len(store) >= rule.max_keys:
                        raise LimitExceeded("Rate limiter capacity exceeded", 5)
                    continue
                if bucket and current - bucket[-1] < rule.minimum_interval:
                    raise LimitExceeded("Frames are arriving too quickly", max(1, int(rule.minimum_interval)))
                if len(bucket) >= rule.limit:
                    retry = max(1, int(rule.window_seconds - (current - bucket[0])))
                    raise LimitExceeded(rule.detail, retry)
            for name, key in checks:
                store = self._buckets[name]
                bucket = store.get(key)
                if bucket is None:
                    bucket = deque()
                    store[key] = bucket
                else:
                    store.move_to_end(key)
                bucket.append(current)

    def _prune(self, name: str, now: float) -> None:
        rule = self.rules[name]
        store = self._buckets[name]
        for key, bucket in list(store.items()):
            while bucket and now - bucket[0] >= rule.window_seconds:
                bucket.popleft()
            if not bucket:
                del store[key]

    def clear(self) -> None:
        with self._lock:
            for store in self._buckets.values():
                store.clear()

    def key_count(self, tier: str) -> int:
        with self._lock:
            return len(self._buckets[tier])
