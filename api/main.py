"""
main.py — Document Q&A API
============================
FastAPI application exposing the full RAG pipeline.

Endpoints:
  GET  /health               — system status
  GET  /store/status         — vector store stats
  POST /ingest               — index a document
  POST /ask                  — single question
  POST /ask/batch            — multiple questions
  DELETE /store              — reset the vector store

Pipeline per /ask request:
  1. Embed the question → 384-dim vector
  2. Query ChromaDB → top 3 semantically similar chunks
  3. Check retrieval confidence
  4. Generate answer with LLM (Groq or Ollama)
  5. Return structured response with sources
"""

import os
import sys
import logging
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion import IngestionPipeline
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from generator import AnswerGenerator
from cache import init_cache, get_cached, set_cached, invalidate_all
from query_logger import init_db, log_query
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ────────────────────────────────────────────────────────────
_config = None
_embedder = None
_vectorstore = None
_generator = None
_ingestion_pipeline = None


def load_components():
    global _config, _embedder, _vectorstore, _generator, _ingestion_pipeline

    with open("configs/config.yaml") as f:
        _config = yaml.safe_load(f)

    _embedder = EmbeddingModel(_config)
    _vectorstore = VectorStore(_config)
    _generator = AnswerGenerator(_config)
    _ingestion_pipeline = IngestionPipeline()

    # Initialise SQLite stores
    init_cache()
    init_db()

    logger.info(
        f"Components loaded. "
        f"Vector store: {_vectorstore.count()} chunks indexed. "
        f"LLM: {'Ollama' if os.environ.get('USE_OLLAMA') == 'true' else 'Groq'}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_components()
    yield
    logger.info("API shutting down")


# ── Rate limiter ────────────────────────────────────────────────────────────
# 30 requests/minute per IP — matches Groq free tier limit
# Prevents a single user from exhausting API quota
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


app = FastAPI(
    title="Document Q&A API",
    description=(
        "Semantic document Q&A using Sentence-BERT embeddings, "
        "ChromaDB vector store, and Llama 3.1 via Groq. "
        "Answers grounded in indexed documents only — hallucination constrained."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ───────────────────────────────────────────────────────────────────


def build_ask_response(question: str, n_chunks: int = 3) -> AskResponse:
    """Core Q&A logic. Used by both /ask and /ask/batch."""
    # Step 1: Embed question
    query_embedding = _embedder.embed_single(question)

    # Step 2: Retrieve relevant chunks
    retrieved = _vectorstore.query(query_embedding, n_results=n_chunks)

    # Step 3: Generate answer
    gen_result = _generator.generate(question, retrieved)

    # Step 4: Build source references for response
    sources = []
    for chunk in retrieved:
        sources.append(
            SourceChunk(
                text_preview=chunk.text[:300]
                + ("..." if len(chunk.text) > 300 else ""),
                source_file=chunk.metadata.get("filename", "unknown"),
                chunk_index=chunk.chunk_index,
                relevance_distance=chunk.distance,
                confidence=ConfidenceLevel(chunk.confidence),
            )
        )

    # Step 5: Build warning for low-confidence responses
    warning = None
    if gen_result.retrieval_confidence == "low" or gen_result.generation_skipped:
        warning = (
            "Retrieval confidence is low. The answer may be inaccurate. "
            "Try rephrasing your question or check that the relevant document is indexed."
        )

    return AskResponse(
        question=question,
        answer=gen_result.answer,
        retrieval_confidence=ConfidenceLevel(gen_result.retrieval_confidence),
        sources=sources,
        generation_skipped=gen_result.generation_skipped,
        warning=warning,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check. Includes vector store stats, cache stats, and LLM config."""
    
    return HealthResponse(
        status="healthy",
        store_chunks=_vectorstore.count(),
        embedding_model=_config["embedding"]["model_name"],
        llm_provider="ollama" if os.environ.get("USE_OLLAMA") == "true" else "groq",
        llm_model=_config["llm"]["model"],
    )


@app.get("/store/status", response_model=StoreStatus, tags=["Store"])
async def store_status():
    """Returns current vector store statistics."""
    count = _vectorstore.count()
    return StoreStatus(
        total_chunks=count,
        status="ready" if count > 0 else "empty — ingest a document first",
        embedding_model=_config["embedding"]["model_name"],
        collection_name=_config["vectorstore"]["collection_name"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the vector store.

    Pipeline: Load file → Split into chunks → Embed → Index in ChromaDB

    reset=True: deletes existing index before ingesting.
    reset=False (default): adds to existing index (multiple documents supported).

    Supported file types: .txt, .md, .pdf
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.file_path}"
        )

    if request.reset:
        _vectorstore.delete_collection()
        logger.info("Vector store reset before ingestion")

    try:
        # Load and chunk document
        chunks = _ingestion_pipeline.ingest(request.file_path)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Document produced no chunks. Check file content.",
            )

        # Embed chunks
        texts = [c.text for c in chunks]
        embedding_result = _embedder.embed(texts, show_progress=True)

        # Index in ChromaDB
        _vectorstore.add_chunks(chunks, embedding_result.embeddings)

        return IngestResponse(
            status="success",
            file_path=request.file_path,
            chunks_created=len(chunks),
            total_chunks_in_store=_vectorstore.count(),
            message=(
                f"Indexed {len(chunks)} chunks from {request.file_path}. "
                f"Store now contains {_vectorstore.count()} total chunks."
            ),
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest):
    """
    Ask a question about indexed documents.

    - Cache: repeated questions return instantly without hitting the LLM.
    - Rate limit: 30 requests/minute per IP (matches Groq free tier).
    - Logging: every query and response latency is logged to SQLite.
    """
    if _vectorstore.count() == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. POST to /ingest first."
        )

    # Cache check
    cached = get_cached(body.question, body.n_chunks)
    if cached is not None:
        return AskResponse(**cached)

    # Full pipeline
    with log_query() as record:
        response = build_ask_response(body.question, body.n_chunks)
        record(response)

    # Cache successful high/medium confidence responses
    if response.retrieval_confidence in (ConfidenceLevel.high, ConfidenceLevel.medium):
        set_cached(body.question, body.n_chunks, response.dict())

    return response


@app.post("/ask/batch", response_model=BatchAskResponse, tags=["Q&A"])
async def ask_batch(request: BatchAskRequest):
    """
    Ask up to 10 questions in a single request.
    More efficient than 10 sequential /ask calls.
    """
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
async def reset_store():
    """
    Delete all indexed documents and invalidate the response cache.
    Cache must be cleared — cached answers based on old documents are invalid.
    """
    _vectorstore.delete_collection()
    n_cleared = invalidate_all()
    return {
        "status": "success",
        "message": f"Vector store cleared. {n_cleared} cached responses invalidated.",
    }


@app.get("/analytics", tags=["System"])
async def analytics():
    """Query log analytics: confidence distribution, response times, low-confidence questions."""
    from query_logger import get_analytics

    data = get_analytics()
    return {k: v.to_dict(orient="records") for k, v in data.items()}


if __name__ == "__main__":
    import uvicorn

    cfg = yaml.safe_load(open("configs/config.yaml"))
    uvicorn.run(
        "main:app", host=cfg["api"]["host"], port=cfg["api"]["port"], reload=False
    )
