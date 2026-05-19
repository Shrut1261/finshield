"""Pydantic schemas for transaction ingestion and API validation."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MerchantCategory(str, Enum):
    GROCERY = "grocery"
    GAS_STATION = "gas_station"
    RESTAURANT = "restaurant"
    ONLINE_RETAIL = "online_retail"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    ATM_WITHDRAWAL = "atm_withdrawal"
    WIRE_TRANSFER = "wire_transfer"
    CRYPTO = "crypto"
    OTHER = "other"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionCode(str, Enum):
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    AUTO_DECLINE = "auto_decline"


class TransactionInput(BaseModel):
    """Incoming transaction payload for fraud scoring."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    card_id: str = Field(..., description="Card identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(default="other")
    amount: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in USD")
    currency: str = Field(default="USD", max_length=3)
    country: str = Field(default="US", max_length=2)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool = Field(default=False)
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.upper()

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        return v.upper()


class ScoreResponse(BaseModel):
    """Fraud scoring response with SHAP explanation."""

    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier = RiskTier.LOW
    decision: DecisionCode = DecisionCode.AUTO_APPROVE
    shap_top_features: list[dict[str, str | float]] = Field(default_factory=list)
    model_version: str = "unknown"
    latency_ms: float = 0.0
    scored_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def assign_risk_and_decision(self) -> "ScoreResponse":
        p = self.fraud_probability
        if p < 0.30:
            self.risk_tier = RiskTier.LOW
            self.decision = DecisionCode.AUTO_APPROVE
        elif p < 0.60:
            self.risk_tier = RiskTier.MEDIUM
            self.decision = DecisionCode.MANUAL_REVIEW
        elif p < 0.85:
            self.risk_tier = RiskTier.HIGH
            self.decision = DecisionCode.MANUAL_REVIEW
        else:
            self.risk_tier = RiskTier.CRITICAL
            self.decision = DecisionCode.AUTO_DECLINE
        return self
