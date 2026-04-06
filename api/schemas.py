"""
schemas.py — Document Q&A API schemas
"""
from typing import Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    none = "none"


class IngestRequest(BaseModel):
    file_path: str = Field(..., example="data/raw/budget_2024.pdf")
    reset: bool = Field(
        False,
        description="If True, deletes existing index before ingesting. "
        "Use when replacing a document, not adding to collection.",
    )


class IngestResponse(BaseModel):
    status: str
    file_path: str
    chunks_created: int
    total_chunks_in_store: int
    message: str


class SourceChunk(BaseModel):
    """A retrieved chunk shown to the user for transparency."""

    text_preview: str  # First 300 chars
    source_file: str
    chunk_index: int
    relevance_distance: float
    confidence: ConfidenceLevel


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, example="What is the education budget?"
    )
    n_chunks: Optional[int] = Field(
        3, ge=1, le=10, description="Number of chunks to retrieve"
    )

    @validator("question")
    def question_not_empty(cls, v):
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
    questions: list[str] = Field(..., min_items=1, max_items=10)

    @validator("questions")
    def validate_questions(cls, v):
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
    version: str = "1.0.0"
