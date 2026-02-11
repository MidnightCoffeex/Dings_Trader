"""
Shared Feature Cache for dings-trader.
Ensures candles and core features are only fetched and computed once per interval.
"""
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger("FeatureCache")

class FeatureCache:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FeatureCache, cls).__new__(cls)
                cls._instance._cache = {}
                cls._instance._ttl = 60  # Default 60 seconds TTL for candles/features
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            return None
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expire_in = ttl if ttl is not None else self._ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + expire_in
        }

def get_feature_cache() -> FeatureCache:
    return FeatureCache()
