"""Pydantic v2 schemas for the scoring API.

Re-exports ingestion schemas plus API-specific response models.
"""
from ingestion.schema import (
    DecisionCode,
    MerchantCategory,
    RiskTier,
    ScoreResponse,
    TransactionInput,
)

__all__ = [
    "TransactionInput",
    "ScoreResponse",
    "RiskTier",
    "DecisionCode",
    "MerchantCategory",
    "BatchScoreRequest",
    "BatchScoreResponse",
    "HealthResponse",
    "MetricsResponse",
]

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=500)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    total: int
    fraud_count: int
    avg_fraud_probability: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    total_scored: int
    fraud_flagged: int
    auto_approved: int
    manual_review: int
    auto_declined: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    since: datetime
