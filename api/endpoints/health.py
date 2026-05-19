"""GET /health — liveness and readiness check."""
from __future__ import annotations

import time

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["ops"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    from api.main import app

    model = getattr(app.state, "model", None)
    db_engine = getattr(app.state, "db_engine", None)

    db_connected = False
    if db_engine is not None:
        try:
            with db_engine.connect():
                db_connected = True
        except Exception:
            db_connected = False

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_version=getattr(app.state, "model_version", "unknown"),
        model_loaded=model is not None,
        db_connected=db_connected,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
