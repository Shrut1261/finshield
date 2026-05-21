"""POST /score — real-time fraud scoring endpoint."""
from __future__ import annotations

import asyncio
import time
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware import record_scoring_latency
from api.schemas import BatchScoreRequest, BatchScoreResponse, ScoreResponse, TransactionInput

router = APIRouter(prefix="/score", tags=["scoring"])


def get_model(request: Request) -> object:
    return request.app.state.model


def get_model_version(request: Request) -> str:
    return getattr(request.app.state, "model_version", "unknown")


@router.post(
    "",
    response_model=ScoreResponse,
    summary="Score a single transaction for fraud risk",
    description=(
        "Accepts a transaction payload and returns a fraud probability, "
        "risk tier (low/medium/high/critical), decision, and top-5 SHAP "
        "feature contributions for regulatory explainability."
    ),
)
async def score_transaction(
    transaction: TransactionInput,
    model: Annotated[object, Depends(get_model)],
    model_version: Annotated[str, Depends(get_model_version)],
) -> ScoreResponse:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline first.",
        )

    start = time.perf_counter()
    loop = asyncio.get_event_loop()

    try:
        row = pd.DataFrame([transaction.model_dump()])

        # Run synchronous sklearn/XGBoost inference in a thread pool so the
        # event loop is not blocked during CPU-bound scoring.
        fraud_probability = await loop.run_in_executor(
            None, lambda: float(model.predict_proba(row)[0])  # type: ignore[union-attr]
        )

        shap_features: list[dict] = []
        if hasattr(model, "explain"):
            explanations: list[list[dict]] = await loop.run_in_executor(
                None, lambda: model.explain(row, top_n=5)  # type: ignore[union-attr]
            )
            if explanations:
                shap_features = explanations[0]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring error: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    response = ScoreResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(fraud_probability, 6),
        shap_top_features=shap_features,
        model_version=model_version,
        latency_ms=round(latency_ms, 2),
    )

    record_scoring_latency(latency_ms, response.risk_tier.value, response.decision.value)
    return response


@router.post(
    "/batch",
    response_model=BatchScoreResponse,
    summary="Score up to 500 transactions in a single request",
)
async def score_batch(
    request: BatchScoreRequest,
    model: Annotated[object, Depends(get_model)],
    model_version: Annotated[str, Depends(get_model_version)],
) -> BatchScoreResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.perf_counter()
    loop = asyncio.get_event_loop()

    rows = pd.DataFrame([t.model_dump() for t in request.transactions])

    probas: list[float] = await loop.run_in_executor(
        None, lambda: model.predict_proba(rows).tolist()  # type: ignore[union-attr]
    )

    explanations: list[list[dict]] = []
    if hasattr(model, "explain"):
        explanations = await loop.run_in_executor(
            None, lambda: model.explain(rows, top_n=5)  # type: ignore[union-attr]
        )

    results: list[ScoreResponse] = []
    for i, (tx, prob) in enumerate(zip(request.transactions, probas)):
        resp = ScoreResponse(
            transaction_id=tx.transaction_id,
            fraud_probability=round(float(prob), 6),
            shap_top_features=explanations[i] if explanations else [],
            model_version=model_version,
            latency_ms=0.0,
        )
        results.append(resp)
        record_scoring_latency(0.0, resp.risk_tier.value, resp.decision.value)

    total_ms = (time.perf_counter() - start) * 1000
    fraud_count = sum(1 for r in results if r.risk_tier.value in ("high", "critical"))

    return BatchScoreResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        avg_fraud_probability=round(sum(r.fraud_probability for r in results) / len(results), 6),
        processing_time_ms=round(total_ms, 2),
    )
