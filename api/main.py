"""Production-oriented Document Q&A API."""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from collections import OrderedDict
from threading import Lock

import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except Exception:  # pragma: no cover
    Counter = Histogram = generate_latest = CONTENT_TYPE_LATEST = None

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion import IngestionPipeline
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from generator import AnswerGenerator
from retrieval import HybridRetriever
from cache import init_cache, get_cached, set_cached, invalidate_all, get_cache_stats
from query_logger import init_db, log_query, log_feedback
from jobs import JobManager
from api.schemas import (
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    SourceChunk,
    ConfidenceLevel,
    BatchAskRequest,
    BatchAskResponse,
    StoreStatus,
    HealthResponse,
    AsyncJobResponse,
    JobStatusResponse,
    FeedbackRequest,
    FeedbackResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
)
logger = logging.getLogger(__name__)


class _RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"].setdefault("request_id", kwargs["extra"].get("request_id", "-"))
        return msg, kwargs


log = _RequestLoggerAdapter(logger, {})

# Global state
_config = None
_embedder = None
_vectorstore = None
_generator = None
_ingestion_pipeline = None
_retriever = None
_jobs = None


class EmbeddingCache:
    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)


_embedding_cache = EmbeddingCache(max_size=1024)


def init_tracing(app: FastAPI) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({"service.name": "document-qa-rag-api"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        log.info("OpenTelemetry tracing enabled")
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to initialize OpenTelemetry: %s", exc)


def load_components():
    global _config, _embedder, _vectorstore, _generator, _ingestion_pipeline, _retriever, _jobs

    with open("configs/config.yaml", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    _embedder = EmbeddingModel(_config)
    _vectorstore = VectorStore(_config)
    _generator = AnswerGenerator(_config)
    _ingestion_pipeline = IngestionPipeline()
    _retriever = HybridRetriever(
        lexical_weight=float(_config.get("retrieval", {}).get("lexical_weight", 0.25)),
        mmr_lambda=float(_config.get("retrieval", {}).get("mmr_lambda", 0.7)),
    )
    _jobs = JobManager(max_workers=int(os.getenv("INGEST_JOB_WORKERS", "2")))

    init_cache()
    init_db()

    log.info(
        "Components loaded. chunks=%s llm=%s",
        _vectorstore.count(),
        "Ollama" if os.environ.get("USE_OLLAMA") == "true" else "Groq",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_components()
    init_tracing(app)
    yield
    log.info("API shutting down")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        if REQUEST_LATENCY_MS:
            REQUEST_LATENCY_MS.labels(request.method, request.url.path).observe(elapsed)
        if REQUEST_TOTAL:
            REQUEST_TOTAL.labels(
                request.method, request.url.path, str(response.status_code)
            ).inc()
        log.info(
            "request complete method=%s path=%s status=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            extra={"request_id": request_id},
        )
        return response


limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
REQUEST_TOTAL = (
    Counter(
        "rag_api_requests_total", "Total API requests", ["method", "path", "status"]
    )
    if Counter
    else None
)
REQUEST_LATENCY_MS = (
    Histogram(
        "rag_api_request_latency_ms",
        "Request latency in milliseconds",
        ["method", "path"],
    )
    if Histogram
    else None
)
RAG_CACHE_HITS = (
    Counter("rag_cache_hits_total", "Response cache hits") if Counter else None
)

app = FastAPI(
    title="Document Q&A API",
    description="Production RAG API with retrieval reranking, caching, auth, observability, and async ingestion jobs.",
    version="1.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "-")
    log.exception("unhandled error: %s", exc, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


async def verify_api_key(request: Request):
    configured = os.getenv("API_KEYS", "").strip()
    if not configured:
        return

    allowed_keys = {k.strip() for k in configured.split(",") if k.strip()}
    incoming = request.headers.get("X-API-Key")
    if incoming not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _embed_question(question: str):
    cache_key = question.strip().lower()
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    emb = _embedder.embed_single(question)
    _embedding_cache.set(cache_key, emb)
    return emb


def _ingest_sync(file_path: str, reset: bool) -> dict:
    if reset:
        _vectorstore.delete_collection()
    chunks = _ingestion_pipeline.ingest(file_path)
    if not chunks:
        raise ValueError("Document produced no chunks. Check file content.")
    embedding_result = _embedder.embed([c.text for c in chunks], show_progress=True)
    _vectorstore.add_chunks(chunks, embedding_result.embeddings)
    return {
        "status": "success",
        "file_path": file_path,
        "chunks_created": len(chunks),
        "total_chunks_in_store": _vectorstore.count(),
        "message": f"Indexed {len(chunks)} chunks from {file_path}.",
    }


def build_ask_response(question: str, n_chunks: int = 3) -> AskResponse:
    query_embedding = _embed_question(question)
    candidate_k = min(max(n_chunks * 4, n_chunks), 20)
    candidates = _vectorstore.query(query_embedding, n_results=candidate_k)
    retrieved = _retriever.rerank(question, candidates, top_k=n_chunks)
    gen_result = _generator.generate(question, retrieved)

    sources = [
        SourceChunk(
            text_preview=chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
            source_file=chunk.metadata.get("filename", "unknown"),
            chunk_index=chunk.chunk_index,
            relevance_distance=chunk.distance,
            confidence=ConfidenceLevel(chunk.confidence),
        )
        for chunk in retrieved
    ]

    warning = None
    if gen_result.retrieval_confidence == "low" or gen_result.generation_skipped:
        warning = "Retrieval confidence is low or fallback mode was used. Verify sources before acting."

    return AskResponse(
        question=question,
        answer=gen_result.answer,
        retrieval_confidence=ConfidenceLevel(gen_result.retrieval_confidence),
        sources=sources,
        generation_skipped=gen_result.generation_skipped,
        warning=warning,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    cache_stats = get_cache_stats()
    hit_rate = round(
        cache_stats["total_hits"] / max(cache_stats["total_entries"], 1), 4
    )
    return HealthResponse(
        status="healthy",
        store_chunks=_vectorstore.count(),
        embedding_model=_config["embedding"]["model_name"],
        llm_provider="ollama" if os.environ.get("USE_OLLAMA") == "true" else "groq",
        llm_model=_config["llm"]["model"],
        cache_entries=cache_stats["total_entries"],
        cache_hits=cache_stats["total_hits"],
        cache_backend=cache_stats.get("backend", "unknown"),
        cache_hit_rate=hit_rate,
    )


@app.get("/store/status", response_model=StoreStatus, tags=["Store"])
async def store_status():
    count = _vectorstore.count()
    return StoreStatus(
        total_chunks=count,
        status="ready" if count > 0 else "empty — ingest a document first",
        embedding_model=_config["embedding"]["model_name"],
        collection_name=_config["vectorstore"]["collection_name"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(request: IngestRequest, raw_request: Request):
    await verify_api_key(raw_request)
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.file_path}"
        )
    try:
        result = _ingest_sync(request.file_path, request.reset)
        return IngestResponse(**result)
    except Exception as exc:
        log.exception(
            "ingestion failed: %s",
            exc,
            extra={"request_id": raw_request.state.request_id},
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/ingest/async", response_model=AsyncJobResponse, tags=["Ingestion"])
async def ingest_document_async(request: IngestRequest, raw_request: Request):
    await verify_api_key(raw_request)
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.file_path}"
        )
    job_id = _jobs.submit(_ingest_sync, request.file_path, request.reset)
    return AsyncJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Ingestion"])
async def get_job_status(job_id: str, request: Request):
    await verify_api_key(request)
    rec = _jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=rec.job_id, status=rec.status, error=rec.error)


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest):
    await verify_api_key(request)
    if _vectorstore.count() == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. POST to /ingest first."
        )

    cached = get_cached(body.question, body.n_chunks)
    if cached is not None:
        if RAG_CACHE_HITS:
            RAG_CACHE_HITS.inc()
        return AskResponse(**cached)

    with log_query() as record:
        response = build_ask_response(body.question, body.n_chunks)
        record(response)

    if response.retrieval_confidence in (ConfidenceLevel.high, ConfidenceLevel.medium):
        set_cached(body.question, body.n_chunks, response.dict())

    return response


@app.post("/ask/batch", response_model=BatchAskResponse, tags=["Q&A"])
async def ask_batch(request: BatchAskRequest, raw_request: Request):
    await verify_api_key(raw_request)
    if _vectorstore.count() == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. POST to /ingest first."
        )

    answers = [build_ask_response(q) for q in request.questions]
    high_conf = sum(
        1 for a in answers if a.retrieval_confidence == ConfidenceLevel.high
    )
    low_conf = sum(1 for a in answers if a.retrieval_confidence == ConfidenceLevel.low)

    return BatchAskResponse(
        answers=answers,
        total_questions=len(answers),
        high_confidence_answers=high_conf,
        low_confidence_answers=low_conf,
    )


@app.delete("/store", tags=["Store"])
async def reset_store(request: Request):
    await verify_api_key(request)
    _vectorstore.delete_collection()
    n_cleared = invalidate_all()
    return {
        "status": "success",
        "message": f"Vector store cleared. {n_cleared} cached responses invalidated.",
    }


@app.get("/analytics", tags=["System"])
async def analytics(request: Request):
    await verify_api_key(request)
    from query_logger import get_analytics

    data = get_analytics()
    return {k: v.to_dict(orient="records") for k, v in data.items()}


@app.post("/feedback", response_model=FeedbackResponse, tags=["System"])
async def feedback(body: FeedbackRequest, request: Request):
    await verify_api_key(request)
    log_feedback(body.question, body.rating, body.comment)
    return FeedbackResponse(status="recorded")


@app.get("/metrics", tags=["System"])
async def metrics():
    if generate_latest is None:
        return PlainTextResponse("prometheus-client is not installed", status_code=503)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    uvicorn.run(
        "main:app", host=cfg["api"]["host"], port=cfg["api"]["port"], reload=False
    )
