"""Simple async job queue with optional Redis-backed state."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self.redis_url = os.getenv("REDIS_URL")
        self._redis = None

    def _get_redis(self):
        if not self.redis_url:
            return None
        if self._redis is None:
            import redis

            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _persist(self, rec: JobRecord):
        client = self._get_redis()
        if client is None:
            return
        client.hset(
            f"rag:job:{rec.job_id}", mapping={k: str(v) for k, v in asdict(rec).items()}
        )
        client.expire(f"rag:job:{rec.job_id}", 24 * 3600)

    def submit(self, fn, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        rec = JobRecord(job_id=job_id, status="queued", created_at=now, updated_at=now)
        with self._lock:
            self._jobs[job_id] = rec
        self._persist(rec)

        def runner():
            rec.status = "running"
            rec.updated_at = datetime.utcnow().isoformat()
            self._persist(rec)
            try:
                result = fn(*args, **kwargs)
                rec.status = "completed"
                rec.result = result
            except Exception as exc:  # pragma: no cover
                rec.status = "failed"
                rec.error = str(exc)
            finally:
                rec.updated_at = datetime.utcnow().isoformat()
                self._persist(rec)

        self._executor.submit(runner)
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            local = self._jobs.get(job_id)
        if local is not None:
            return local

        client = self._get_redis()
        if client is None:
            return None
        data = client.hgetall(f"rag:job:{job_id}")
        if not data:
            return None
        return JobRecord(
            job_id=data["job_id"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            result=None,
            error=data.get("error") if data.get("error") != "None" else None,
        )
