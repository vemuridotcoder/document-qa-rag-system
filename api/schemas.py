"""Pydantic schemas for Document Q&A API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    none = "none"


class IngestRequest(BaseModel):
    file_path: str = Field(..., example="data/raw/budget_2024.pdf")
    reset: bool = Field(False, description="If True, deletes existing index before ingesting.")


class IngestResponse(BaseModel):
    status: str
    file_path: str
    chunks_created: int
    total_chunks_in_store: int
    message: str


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None


class SourceChunk(BaseModel):
    text_preview: str
    source_file: str
    chunk_index: int
    relevance_distance: float
    confidence: ConfidenceLevel


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="What is the education budget?")
    n_chunks: Optional[int] = Field(3, ge=1, le=10, description="Number of chunks to retrieve")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieval_confidence: ConfidenceLevel
    sources: list[SourceChunk]
    generation_skipped: bool
    warning: Optional[str] = None


class BatchAskRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=10)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: list[str]) -> list[str]:
        if len(v) > 10:
            raise ValueError("Maximum 10 questions per batch")
        return [q.strip() for q in v if q.strip()]


class BatchAskResponse(BaseModel):
    answers: list[AskResponse]
    total_questions: int
    high_confidence_answers: int
    low_confidence_answers: int


class StoreStatus(BaseModel):
    total_chunks: int
    status: str
    embedding_model: str
    collection_name: str


class HealthResponse(BaseModel):
    status: str
    store_chunks: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    cache_entries: int = 0
    cache_hits: int = 0
    cache_backend: str = "sqlite"
    cache_hit_rate: float = 0.0
    version: str = "1.2.0"


class FeedbackRequest(BaseModel):
    question: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
