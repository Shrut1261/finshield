"""GET /metrics — operational and Prometheus metrics."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.middleware import get_latency_percentiles
from api.schemas import MetricsResponse

router = APIRouter(tags=["ops"])

_counters: dict[str, int] = {
    "total_scored": 0,
    "fraud_flagged": 0,
    "auto_approved": 0,
    "manual_review": 0,
    "auto_declined": 0,
}
_metrics_since = datetime.utcnow()


def increment(key: str) -> None:
    _counters[key] = _counters.get(key, 0) + 1


@router.get("/metrics/json", response_model=MetricsResponse, summary="JSON operational metrics")
async def metrics_json() -> MetricsResponse:
    p50, p95, p99 = get_latency_percentiles()
    return MetricsResponse(
        total_scored=_counters["total_scored"],
        fraud_flagged=_counters["fraud_flagged"],
        auto_approved=_counters["auto_approved"],
        manual_review=_counters["manual_review"],
        auto_declined=_counters["auto_declined"],
        avg_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        since=_metrics_since,
    )


@router.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics_prometheus() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
