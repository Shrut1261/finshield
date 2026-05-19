"""FinShield FastAPI application entry point.

Startup sequence:
  1. Configure structlog for JSON output
  2. Load model artifact from disk (or MLflow registry)
  3. Connect to PostgreSQL
  4. Mount routers
  5. Start Prometheus metrics server
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints.health import router as health_router
from api.endpoints.metrics import router as metrics_router
from api.endpoints.score import router as score_router
from api.middleware import LatencyLoggingMiddleware

# ── Logging ────────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/artifacts/ensemble.pkl"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and connect to DB on startup; clean up on shutdown."""
    logger.info("startup", model_path=str(MODEL_PATH))

    # Load model
    app.state.model = None
    app.state.model_version = MODEL_VERSION

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            app.state.model = pickle.load(f)
        logger.info("model_loaded", path=str(MODEL_PATH), version=MODEL_VERSION)
    else:
        logger.warning("model_not_found", path=str(MODEL_PATH))

    # Connect to database (optional — API degrades gracefully without it)
    app.state.db_engine = None
    if DATABASE_URL:
        try:
            from sqlalchemy import create_engine
            app.state.db_engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
            # Verify connectivity
            with app.state.db_engine.connect():
                pass
            logger.info("db_connected", url=DATABASE_URL.split("@")[-1])
        except Exception as exc:
            logger.warning("db_connection_failed", error=str(exc))

    yield  # Application runs here

    logger.info("shutdown")
    if app.state.db_engine:
        app.state.db_engine.dispose()


app = FastAPI(
    title="FinShield Fraud Detection API",
    description=(
        "Real-time fraud scoring API. Each request returns a fraud probability, "
        "risk tier, routing decision, and top-5 SHAP feature contributions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(LatencyLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(score_router)
app.include_router(health_router)
app.include_router(metrics_router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "FinShield Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "1")),
        reload=os.getenv("ENV", "prod") == "dev",
    )
