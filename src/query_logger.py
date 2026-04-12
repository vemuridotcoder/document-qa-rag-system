"""Query logging with SQLite/Postgres support and analytics helpers."""

from __future__ import annotations

import os
import sqlite3
import time
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "data/query_logs.db"


def _db_url() -> str:
    return os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")


class QueryLogStore:
    def __init__(self):
        self.url = _db_url()

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql://") or self.url.startswith("postgres://")

    def init(self):
        if self.is_postgres:
            import psycopg

            with psycopg.connect(self.url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_logs (
                            id                  BIGSERIAL PRIMARY KEY,
                            timestamp           TEXT NOT NULL,
                            question            TEXT NOT NULL,
                            answer_preview      TEXT,
                            retrieval_confidence TEXT,
                            generation_skipped  INTEGER,
                            response_time_ms    REAL,
                            top_source_file     TEXT,
                            top_source_distance REAL,
                            tokens_used         INTEGER DEFAULT 0
                        )
                        """
                    )
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON query_logs(retrieval_confidence)")
                conn.commit()
            return

        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT NOT NULL,
                question            TEXT NOT NULL,
                answer_preview      TEXT,
                retrieval_confidence TEXT,
                generation_skipped  INTEGER,
                response_time_ms    REAL,
                top_source_file     TEXT,
                top_source_distance REAL,
                tokens_used         INTEGER DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON query_logs(retrieval_confidence)")
        conn.commit()
        conn.close()

    def insert(self, payload: tuple):
        if self.is_postgres:
            import psycopg

            with psycopg.connect(self.url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO query_logs
                            (timestamp, question, answer_preview, retrieval_confidence,
                            generation_skipped, response_time_ms, top_source_file, top_source_distance)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        payload,
                    )
                conn.commit()
            return

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO query_logs
                (timestamp, question, answer_preview, retrieval_confidence,
                 generation_skipped, response_time_ms, top_source_file, top_source_distance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
        conn.close()

    def analytics(self) -> dict:
        import pandas as pd

        if self.is_postgres:
            import psycopg

            with psycopg.connect(self.url) as conn:
                summary = pd.read_sql_query(
                    """
                    SELECT COUNT(*) AS total_queries,
                           ROUND(100.0 * SUM(generation_skipped)::numeric / NULLIF(COUNT(*), 0), 2) AS skip_rate_pct,
                           ROUND(AVG(response_time_ms)::numeric, 1) AS avg_response_ms,
                           ROUND(MIN(response_time_ms)::numeric, 1) AS min_response_ms,
                           ROUND(MAX(response_time_ms)::numeric, 1) AS max_response_ms
                    FROM query_logs
                    """,
                    conn,
                )
                conf = pd.read_sql_query(
                    """
                    SELECT retrieval_confidence, COUNT(*) AS count,
                           ROUND(100.0 * COUNT(*)::numeric / NULLIF((SELECT COUNT(*) FROM query_logs),0), 2) AS pct
                    FROM query_logs GROUP BY retrieval_confidence ORDER BY count DESC
                    """,
                    conn,
                )
                low = pd.read_sql_query(
                    """
                    SELECT timestamp, question, top_source_distance, response_time_ms
                    FROM query_logs
                    WHERE retrieval_confidence = 'low'
                    ORDER BY timestamp DESC
                    LIMIT 20
                    """,
                    conn,
                )
            return {"summary": summary, "confidence_distribution": conf, "low_confidence_queries": low}

        if not os.path.exists(DB_PATH):
            return {}

        conn = sqlite3.connect(DB_PATH)
        summary = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total_queries,
                   ROUND(100.0 * SUM(generation_skipped) / COUNT(*), 2) AS skip_rate_pct,
                   ROUND(AVG(response_time_ms), 1) AS avg_response_ms,
                   ROUND(MIN(response_time_ms), 1) AS min_response_ms,
                   ROUND(MAX(response_time_ms), 1) AS max_response_ms
            FROM query_logs
            """,
            conn,
        )
        conf = pd.read_sql_query(
            """
            SELECT retrieval_confidence, COUNT(*) AS count,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM query_logs), 2) AS pct
            FROM query_logs
            GROUP BY retrieval_confidence
            ORDER BY count DESC
            """,
            conn,
        )
        low = pd.read_sql_query(
            """
            SELECT timestamp, question, top_source_distance, response_time_ms
            FROM query_logs
            WHERE retrieval_confidence = 'low'
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            conn,
        )
        hourly = pd.read_sql_query(
            """
            SELECT SUBSTR(timestamp, 1, 13) AS hour,
                   COUNT(*) AS queries,
                   SUM(COUNT(*)) OVER (ORDER BY SUBSTR(timestamp, 1, 13)) AS cumulative_queries
            FROM query_logs
            GROUP BY hour
            ORDER BY hour
            """,
            conn,
        )
        conn.close()
        return {
            "summary": summary,
            "confidence_distribution": conf,
            "low_confidence_queries": low,
            "hourly_volume": hourly,
        }


_STORE = QueryLogStore()


def init_db(db_path: str = DB_PATH) -> None:
    _ = db_path
    _STORE.init()


@contextmanager
def log_query(db_path: str = DB_PATH):
    _ = db_path
    start_time = time.perf_counter()
    result_container = {}

    def record(response_obj):
        result_container["response"] = response_obj

    yield record

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response = result_container.get("response")
    if response is None:
        return

    top_source_file = None
    top_source_distance = None
    if response.sources:
        top_source_file = response.sources[0].source_file
        top_source_distance = response.sources[0].relevance_distance

    payload = (
        datetime.utcnow().isoformat(),
        response.question,
        response.answer[:300] if response.answer else None,
        response.retrieval_confidence.value if hasattr(response.retrieval_confidence, "value") else str(response.retrieval_confidence),
        int(response.generation_skipped),
        round(elapsed_ms, 2),
        top_source_file,
        top_source_distance,
    )
    try:
        _STORE.insert(payload)
    except Exception as exc:
        logger.error("Failed to log query: %s", exc)


def get_analytics(db_path: str = DB_PATH) -> dict:
    _ = db_path
    return _STORE.analytics()



def log_feedback(question: str, rating: str, comment: str | None = None) -> None:
    """Stores explicit user feedback for product quality loops."""
    timestamp = datetime.utcnow().isoformat()
    if _STORE.is_postgres:
        import psycopg
        with psycopg.connect(_STORE.url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_logs (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        question TEXT NOT NULL,
                        rating TEXT NOT NULL,
                        comment TEXT
                    )
                """)
                cur.execute(
                    "INSERT INTO feedback_logs (timestamp, question, rating, comment) VALUES (%s, %s, %s, %s)",
                    (timestamp, question, rating, comment),
                )
            conn.commit()
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            rating TEXT NOT NULL,
            comment TEXT
        )
    """)
    conn.execute(
        "INSERT INTO feedback_logs (timestamp, question, rating, comment) VALUES (?, ?, ?, ?)",
        (timestamp, question, rating, comment),
    )
    conn.commit()
    conn.close()
