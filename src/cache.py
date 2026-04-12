"""Cache abstraction with SQLite fallback and optional Redis backend."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DB_PATH = "data/response_cache.db"
DEFAULT_TTL_HOURS = 24


def _make_key(question: str, n_chunks: int) -> str:
    payload = f"{question.strip().lower()}::{n_chunks}"
    return hashlib.sha256(payload.encode()).hexdigest()


class BaseCacheBackend:
    def get(self, question: str, n_chunks: int) -> dict | None: ...
    def set(self, question: str, n_chunks: int, response: dict, ttl_hours: int): ...
    def invalidate_all(self) -> int: ...
    def stats(self) -> dict: ...


class SQLiteCacheBackend(BaseCacheBackend):
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        os.makedirs("data", exist_ok=True)

    def init(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key   TEXT PRIMARY KEY,
                question    TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                hit_count   INTEGER DEFAULT 0
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON response_cache(expires_at)")
        conn.commit()
        conn.close()

    def get(self, question: str, n_chunks: int) -> dict | None:
        if not os.path.exists(self.db_path):
            return None
        key = _make_key(question, n_chunks)
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT response_json, expires_at FROM response_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            conn.close()
            return None

        response_json, expires_at = row
        if expires_at < now:
            conn.execute("DELETE FROM response_cache WHERE cache_key = ?", (key,))
            conn.commit()
            conn.close()
            return None

        conn.execute("UPDATE response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (key,))
        conn.commit()
        conn.close()
        return json.loads(response_json)

    def set(self, question: str, n_chunks: int, response: dict, ttl_hours: int):
        key = _make_key(question, n_chunks)
        now = datetime.utcnow()
        expires_at = (now + timedelta(hours=ttl_hours)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO response_cache
                (cache_key, question, response_json, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (key, question, json.dumps(response), now.isoformat(), expires_at),
        )
        conn.commit()
        conn.close()

    def invalidate_all(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
        conn.execute("DELETE FROM response_cache")
        conn.commit()
        conn.close()
        return n

    def stats(self) -> dict:
        if not os.path.exists(self.db_path):
            return {"total_entries": 0, "total_hits": 0, "backend": "sqlite"}
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM response_cache").fetchone()
        conn.close()
        return {"total_entries": row[0], "total_hits": row[1], "backend": "sqlite"}


class RedisCacheBackend(BaseCacheBackend):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis

            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def init(self):
        self._get_client().ping()

    def get(self, question: str, n_chunks: int) -> dict | None:
        key = f"rag:cache:{_make_key(question, n_chunks)}"
        client = self._get_client()
        data = client.get(key)
        if not data:
            return None
        client.incr("rag:cache:hits")
        return json.loads(data)

    def set(self, question: str, n_chunks: int, response: dict, ttl_hours: int):
        key = f"rag:cache:{_make_key(question, n_chunks)}"
        self._get_client().setex(key, int(ttl_hours * 3600), json.dumps(response))

    def invalidate_all(self) -> int:
        client = self._get_client()
        keys = list(client.scan_iter("rag:cache:*") )
        if keys:
            client.delete(*keys)
        return len(keys)

    def stats(self) -> dict:
        client = self._get_client()
        count = sum(1 for _ in client.scan_iter("rag:cache:*") )
        hits = int(client.get("rag:cache:hits") or 0)
        return {"total_entries": count, "total_hits": hits, "backend": "redis"}


_backend: BaseCacheBackend | None = None


def _get_backend() -> BaseCacheBackend:
    global _backend
    if _backend is not None:
        return _backend

    backend = os.getenv("CACHE_BACKEND", "sqlite").lower().strip()
    if backend == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            b = RedisCacheBackend(redis_url)
            b.init()
            _backend = b
            logger.info("Using Redis cache backend")
            return _backend
        except Exception as exc:
            logger.warning("Redis cache unavailable (%s); falling back to SQLite", exc)

    b = SQLiteCacheBackend(CACHE_DB_PATH)
    b.init()
    _backend = b
    logger.info("Using SQLite cache backend")
    return _backend


def init_cache(db_path: str = CACHE_DB_PATH) -> None:
    _ = db_path
    _get_backend()


def get_cached(question: str, n_chunks: int, db_path: str = CACHE_DB_PATH) -> dict | None:
    _ = db_path
    try:
        return _get_backend().get(question, n_chunks)
    except Exception as e:
        logger.error("Cache read failed: %s", e)
        return None


def set_cached(
    question: str,
    n_chunks: int,
    response: dict,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    db_path: str = CACHE_DB_PATH,
) -> None:
    _ = db_path
    try:
        _get_backend().set(question, n_chunks, response, ttl_hours)
    except Exception as e:
        logger.error("Cache write failed: %s", e)


def invalidate_all(db_path: str = CACHE_DB_PATH) -> int:
    _ = db_path
    try:
        return _get_backend().invalidate_all()
    except Exception as e:
        logger.error("Cache invalidation failed: %s", e)
        return 0


def get_cache_stats(db_path: str = CACHE_DB_PATH) -> dict:
    _ = db_path
    try:
        return _get_backend().stats()
    except Exception:
        return {"total_entries": 0, "total_hits": 0, "backend": "unknown"}
