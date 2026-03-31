"""
query_logger.py — Document Q&A System
=======================================
SQLite-backed logging of all API queries and responses.

Logs every /ask request with:
- Question text
- Answer generated
- Retrieval confidence level
- Response time (milliseconds)
- Whether generation was skipped (low confidence)
- Timestamp

Why logging every query:
- Analytics: which questions are users asking most?
- Quality monitoring: which questions return low confidence?
- Debugging: reproduce any past interaction exactly.
- Drift signal: if low-confidence responses increase over time,
  the indexed documents may be stale.

SQL queries provided for analytics use cases:
- Most common questions
- Low confidence rate over time
- Average response time by confidence level
"""

import os
import sqlite3
import time
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "data/query_logs.db"


def init_db(db_path: str = DB_PATH) -> None:
    """
    Create the query log table if it does not exist.
    Schema is append-only — no updates, no deletes.
    Immutable log is the correct pattern for audit trails.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            question            TEXT    NOT NULL,
            answer_preview      TEXT,
            retrieval_confidence TEXT,
            generation_skipped  INTEGER,
            response_time_ms    REAL,
            top_source_file     TEXT,
            top_source_distance REAL,
            tokens_used         INTEGER DEFAULT 0
        )
    """)

    # Index on timestamp for time-range queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp)
    """)
    # Index on confidence for filtering
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_confidence ON query_logs(retrieval_confidence)
    """)
    conn.commit()
    conn.close()
    logger.info(f"Query log DB initialised: {db_path}")


@contextmanager
def log_query(db_path: str = DB_PATH):
    """
    Context manager that times a query and logs the result.

    Usage:
        with log_query() as log:
            response = build_ask_response(question)
            log(response)  # call the logger with the response
    """
    start_time = time.perf_counter()
    result_container = {}

    def record(response_obj):
        result_container["response"] = response_obj

    yield record

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    response = result_container.get("response")
    if response is None:
        return

    try:
        top_source_file = None
        top_source_distance = None
        if response.sources:
            top_source_file = response.sources[0].source_file
            top_source_distance = response.sources[0].relevance_distance

        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO query_logs
                (timestamp, question, answer_preview, retrieval_confidence,
                 generation_skipped, response_time_ms, top_source_file, top_source_distance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            response.question,
            response.answer[:300] if response.answer else None,
            response.retrieval_confidence.value
                if hasattr(response.retrieval_confidence, "value")
                else str(response.retrieval_confidence),
            int(response.generation_skipped),
            round(elapsed_ms, 2),
            top_source_file,
            top_source_distance,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


def get_analytics(db_path: str = DB_PATH) -> dict:
    """
    SQL-based analytics on the query log.
    Returns a dict of DataFrames — useful for a monitoring dashboard.
    """
    import pandas as pd

    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)

    analytics = {}

    # Total queries and confidence breakdown
    analytics["summary"] = pd.read_sql_query("""
        SELECT
            COUNT(*)                                        AS total_queries,
            ROUND(100.0 * SUM(generation_skipped)
                  / COUNT(*), 2)                           AS skip_rate_pct,
            ROUND(AVG(response_time_ms), 1)                AS avg_response_ms,
            ROUND(MIN(response_time_ms), 1)                AS min_response_ms,
            ROUND(MAX(response_time_ms), 1)                AS max_response_ms
        FROM query_logs
    """, conn)

    # Confidence distribution
    analytics["confidence_distribution"] = pd.read_sql_query("""
        SELECT
            retrieval_confidence,
            COUNT(*)                                        AS count,
            ROUND(100.0 * COUNT(*) /
                  (SELECT COUNT(*) FROM query_logs), 2)    AS pct
        FROM query_logs
        GROUP BY retrieval_confidence
        ORDER BY count DESC
    """, conn)

    # Low confidence queries — candidates for document re-indexing
    analytics["low_confidence_queries"] = pd.read_sql_query("""
        SELECT
            timestamp,
            question,
            top_source_distance,
            response_time_ms
        FROM query_logs
        WHERE retrieval_confidence = 'low'
        ORDER BY timestamp DESC
        LIMIT 20
    """, conn)

    # Hourly query volume (window function for running total)
    analytics["hourly_volume"] = pd.read_sql_query("""
        SELECT
            SUBSTR(timestamp, 1, 13)                       AS hour,
            COUNT(*)                                       AS queries,
            SUM(COUNT(*)) OVER (ORDER BY SUBSTR(timestamp, 1, 13))
                                                           AS cumulative_queries
        FROM query_logs
        GROUP BY hour
        ORDER BY hour
    """, conn)

    conn.close()
    return analytics


def print_analytics_report(db_path: str = DB_PATH) -> None:
    """Print a formatted analytics report to stdout."""
    analytics = get_analytics(db_path)
    if not analytics:
        print("No query logs found.")
        return

    print("\n" + "="*60)
    print("  QUERY LOG ANALYTICS")
    print("="*60)
    for section, df in analytics.items():
        print(f"\n  {section.replace('_', ' ').title()}:")
        print(df.to_string(index=False))
