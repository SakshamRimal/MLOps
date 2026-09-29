import time
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any
from app.config import settings


class ResponseCache:
    """LRU cache with TTL for prompt/response pairs."""
    """Thread-safe LRU cache with TTL for prompt/response pairs."""

    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages: list[dict], model: str, **kwargs) -> str:
        content = json.dumps({"messages": messages, "model": model, **kwargs}, sort_keys=True)
        serialized_kwargs = {k: v for k, v in sorted(kwargs.items()) if v is not None}
        content = json.dumps(
            {"messages": messages, "model": model, "kwargs": serialized_kwargs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list[dict], model: str, **kwargs) -> Any | None:
        key = self._make_key(messages, model, **kwargs)
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, messages: list[dict], model: str, value: Any, **kwargs) -> None:
        key = self._make_key(messages, model, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), value)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate_all(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


response_cache = ResponseCache()

