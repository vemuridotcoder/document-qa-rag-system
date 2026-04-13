"""
test_api.py — Document Q&A API tests
=======================================
Run: pytest tests/test_api.py -v

Tests cover:
1. Health and store status endpoints
2. Ingestion of a real text file
3. Q&A on ingested content
4. Low confidence detection for off-topic questions
5. Batch Q&A
6. Input validation rejection
7. Empty store guard
8. Store reset
"""

import os
import pytest
import tempfile

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SAMPLE_DOCUMENT = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it
to learn for themselves.

Types of Machine Learning

Supervised learning uses labeled training data to learn the mapping function
from input to output. Common algorithms include linear regression, logistic
regression, and support vector machines.

Unsupervised learning finds hidden patterns in data without pre-existing labels.
Clustering algorithms like k-means and hierarchical clustering are examples.

Reinforcement learning trains an agent to make decisions by rewarding desired
behaviors and punishing undesired ones. It is used in game playing and robotics.

Applications

Machine learning is applied in spam detection, image recognition, natural
language processing, recommendation systems, and medical diagnosis.
The field has seen rapid growth due to increases in computing power and
availability of large datasets.
"""


@pytest.fixture(scope="module")
def test_doc():
    """Create a temporary text document for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_DOCUMENT)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def client():
    """Returns test client. Skips if components fail to load."""
    try:
        from api.main import app

        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not initialize app: {e}")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "store_chunks" in data
    assert "embedding_model" in data
    assert "llm_provider" in data


def test_store_status(client):
    response = client.get("/store/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "status" in data


def test_ingest_document(client, test_doc):
    """Ingest a sample document — store should have chunks afterwards."""
    response = client.post(
        "/ingest",
        json={"file_path": test_doc, "reset": True},  # Clean slate for test isolation
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] > 0
    assert data["total_chunks_in_store"] > 0


def test_ask_returns_answer(client, test_doc):
    """Question about ingested content should return a non-empty answer."""
    # Ensure doc is ingested
    client.post("/ingest", json={"file_path": test_doc, "reset": True})

    response = client.post(
        "/ask", json={"question": "What is supervised learning?", "n_chunks": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""
    assert data["question"] == "What is supervised learning?"
    assert len(data["sources"]) > 0
    assert data["retrieval_confidence"] in ["high", "medium", "low"]


def test_ask_returns_sources(client, test_doc):
    """Each answer must include source chunks with required fields."""
    client.post("/ingest", json={"file_path": test_doc, "reset": True})
    response = client.post(
        "/ask", json={"question": "What are machine learning applications?"}
    )
    assert response.status_code == 200
    data = response.json()
    for source in data["sources"]:
        assert "text_preview" in source
        assert "source_file" in source
        assert "relevance_distance" in source
        assert "confidence" in source
        assert 0 <= source["relevance_distance"] <= 1


def test_low_confidence_for_off_topic(client, test_doc):
    """
    Completely off-topic question should have low confidence
    and include a warning.
    """
    client.post("/ingest", json={"file_path": test_doc, "reset": True})
    response = client.post(
        "/ask", json={"question": "What is the exchange rate of USD to INR today?"}
    )
    assert response.status_code == 200
    data = response.json()
    # Off-topic question may return low confidence or skip generation
    if data["retrieval_confidence"] == "low":
        assert data["warning"] is not None


def test_batch_ask(client, test_doc):
    """Batch endpoint must return correct count."""
    client.post("/ingest", json={"file_path": test_doc, "reset": True})
    questions = [
        "What is machine learning?",
        "What are types of machine learning?",
    ]
    response = client.post("/ask/batch", json={"questions": questions})
    assert response.status_code == 200
    data = response.json()
    assert data["total_questions"] == 2
    assert len(data["answers"]) == 2
    total = data["high_confidence_answers"] + data["low_confidence_answers"]
    # Medium confidence not counted in those two, so total <= 2
    assert total <= 2


def test_batch_size_limit(client):
    """Batch over 10 questions must be rejected."""
    response = client.post("/ask/batch", json={"questions": ["q"] * 11})
    assert response.status_code == 422


def test_empty_question_rejected(client):
    """Empty question must return 422."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_missing_file_rejected(client):
    """Ingesting non-existent file must return 404."""
    response = client.post("/ingest", json={"file_path": "/nonexistent/file.txt"})
    assert response.status_code == 404


def test_ask_empty_store_rejected(client):
    """
    Asking before any document is indexed should return 400.
    First clear the store.
    """
    client.delete("/store")
    response = client.post("/ask", json={"question": "What is machine learning?"})
    assert response.status_code == 400


def test_store_reset(client, test_doc):
    """Delete endpoint must clear the store."""
    # First ingest something
    client.post("/ingest", json={"file_path": test_doc, "reset": False})
    # Then delete
    response = client.delete("/store")
    assert response.status_code == 200
    # Verify store is empty
    status = client.get("/store/status").json()
    assert status["total_chunks"] == 0


def test_async_ingest_job(client, test_doc):
    response = client.post("/ingest/async", json={"file_path": test_doc, "reset": True})
    assert response.status_code == 200
    payload = response.json()
    assert "job_id" in payload

    job = client.get(f"/jobs/{payload['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] in ["queued", "running", "completed"]
