"""Request middleware: latency tracking and structured request logging."""
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Deque

import structlog
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "finshield_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint", "status_code"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0, 5.0],
)

REQUEST_COUNT = Counter(
    "finshield_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

FRAUD_SCORED = Counter(
    "finshield_fraud_scored_total",
    "Total transactions scored",
    ["risk_tier", "decision"],
)

# ── In-memory latency ring buffer for /metrics endpoint ───────────────────────
_latency_buffer: Deque[float] = deque(maxlen=10_000)


def record_scoring_latency(latency_ms: float, risk_tier: str, decision: str) -> None:
    _latency_buffer.append(latency_ms)
    FRAUD_SCORED.labels(risk_tier=risk_tier, decision=decision).inc()


def get_latency_percentiles() -> tuple[float, float, float]:
    """Return (p50, p95, p99) latency in ms from the ring buffer."""
    import numpy as np
    if not _latency_buffer:
        return 0.0, 0.0, 0.0
    arr = list(_latency_buffer)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Log request metadata and emit Prometheus metrics for every request."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        bound_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        bound_logger.info("request_started")

        try:
            response: Response = await call_next(request)  # type: ignore[arg-type]
        except Exception as exc:
            bound_logger.error("request_failed", error=str(exc))
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        endpoint = request.url.path.split("?")[0]

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).observe(elapsed_ms / 1000)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()

        bound_logger.info(
            "request_completed",
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(round(elapsed_ms, 2))
        return response
