import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from jobs import JobManager


def test_job_lifecycle():
    jm = JobManager(max_workers=1)

    def work(x):
        time.sleep(0.05)
        return {"value": x + 1}

    job_id = jm.submit(work, 1)
    status = jm.get(job_id)
    assert status is not None
    assert status.status in {"queued", "running", "completed"}

    time.sleep(0.1)
    status = jm.get(job_id)
    assert status is not None
    assert status.status == "completed"
    assert status.result == {"value": 2}
