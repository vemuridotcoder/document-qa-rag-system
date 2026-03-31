"""
cache.py — Document Q&A System
================================
SQLite-backed response cache for repeated questions.

Why cache:
- Groq free tier has rate limits (~30 req/min, ~14,400/day).
  Repeated questions consume quota unnecessarily.
- Response latency: a cached response returns in < 1ms vs 1-3s for a full RAG pipeline.
- Identical questions from different users get the same answer deterministically.

Cache design decisions:
- SQLite (not Redis): zero-config, runs locally, no separate process.
  Production upgrade path: swap SQLite for Redis with minimal code change.
- Cache key: SHA-256 of (question + n_chunks). Different n_chunks = different context
  window = potentially different answer. Must be included in the key.
- TTL: 24 hours. Documents may be re-indexed with updated content.
  Stale cache after re-indexing would serve outdated answers.
- Cache invalidation on /store reset: deletes all cached entries when
  the vector store is cleared (documents changed).
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DB_PATH = "data/response_cache.db"
DEFAULT_TTL_HOURS = 24


def init_cache(db_path: str = CACHE_DB_PATH) -> None:
    """Create cache table. Called on application startup."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key   TEXT PRIMARY KEY,
            question    TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            hit_count   INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON response_cache(expires_at)")
    conn.commit()
    conn.close()


def _make_key(question: str, n_chunks: int) -> str:
    """
    Deterministic cache key from question + retrieval parameters.
    SHA-256 ensures fixed-length key regardless of question length.
    """
    payload = f"{question.strip().lower()}::{n_chunks}"
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(
    question: str,
    n_chunks: int,
    db_path: str = CACHE_DB_PATH,
) -> dict | None:
    """
    Return cached response if it exists and has not expired.
    Returns None on cache miss or expiry.
    Increments hit_count for analytics.
    """
    if not os.path.exists(db_path):
        return None

    key = _make_key(question, n_chunks)
    now = datetime.utcnow().isoformat()

    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT response_json, expires_at FROM response_cache WHERE cache_key = ?",
            (key,)
        ).fetchone()

        if row is None:
            conn.close()
            return None

        response_json, expires_at = row
        if expires_at < now:
            # Expired — delete and return miss
            conn.execute("DELETE FROM response_cache WHERE cache_key = ?", (key,))
            conn.commit()
            conn.close()
            logger.debug(f"Cache expired for: {question[:50]}")
            return None

        # Cache hit — increment counter
        conn.execute(
            "UPDATE response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,)
        )
        conn.commit()
        conn.close()
        logger.info(f"Cache HIT: {question[:60]}")
        return json.loads(response_json)

    except Exception as e:
        logger.error(f"Cache read failed: {e}")
        return None


def set_cached(
    question: str,
    n_chunks: int,
    response: dict,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    db_path: str = CACHE_DB_PATH,
) -> None:
    """Store a response in the cache with TTL expiry."""
    key = _make_key(question, n_chunks)
    now = datetime.utcnow()
    expires_at = (now + timedelta(hours=ttl_hours)).isoformat()

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO response_cache
                (cache_key, question, response_json, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (key, question, json.dumps(response), now.isoformat(), expires_at))
        conn.commit()
        conn.close()
        logger.debug(f"Cached response for: {question[:60]}")
    except Exception as e:
        logger.error(f"Cache write failed: {e}")


def invalidate_all(db_path: str = CACHE_DB_PATH) -> int:
    """
    Delete all cached responses.
    Called when /store is reset — cached answers based on old documents
    are no longer valid after re-indexing.
    """
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
    conn.execute("DELETE FROM response_cache")
    conn.commit()
    conn.close()
    logger.info(f"Cache invalidated: {n} entries cleared")
    return n


def get_cache_stats(db_path: str = CACHE_DB_PATH) -> dict:
    """Return cache statistics for the /health endpoint."""
    if not os.path.exists(db_path):
        return {"total_entries": 0, "total_hits": 0}
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM response_cache"
    ).fetchone()
    conn.close()
    return {"total_entries": row[0], "total_hits": row[1]}
