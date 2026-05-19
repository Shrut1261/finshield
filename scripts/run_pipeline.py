"""FinShield pipeline orchestrator.

Runs the end-to-end pipeline in phases.  Each phase is idempotent.

Usage:
    python scripts/run_pipeline.py --phase all
    python scripts/run_pipeline.py --phase ingest
    python scripts/run_pipeline.py --phase features
    python scripts/run_pipeline.py --phase train
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_enable=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"


def _get_engine():
    from sqlalchemy import create_engine
    db_url = os.environ.get("DATABASE_URL", "postgresql://finshield_user:finshield_dev_password@localhost:5432/finshield")
    return create_engine(db_url)


@app.command()
def main(
    phase: str = typer.Option("all", help="Phase to run: setup_db | ingest | dbt | features | train | all"),
    datasets: str = typer.Option("creditcard,paysim,ieee", help="Comma-separated dataset names"),
    mlflow_uri: str = typer.Option("http://localhost:5000", help="MLflow tracking URI"),
) -> None:
    phases = ["setup_db", "ingest", "dbt", "features", "train"] if phase == "all" else [phase]
    dataset_list = datasets.split(",")

    for p in phases:
        logger.info("═══ Phase: %s ═══", p.upper())

        if p == "setup_db":
            from scripts.setup_db import main as setup_db
            setup_db()

        elif p == "ingest":
            engine = _get_engine()
            from ingestion.loaders import DatasetLoader
            loader = DatasetLoader(data_dir=DATA_DIR, engine=engine)
            df, stats = loader.load_all(datasets=dataset_list)
            for s in stats:
                logger.info("  %s: %d rows | fraud_rate=%.4f%%", s.dataset, s.rows_loaded, s.fraud_rate * 100)
            loader.ingest_to_staging(df)

        elif p == "dbt":
            import subprocess
            dbt_dir = Path("warehouse/dbt_finshield")
            if not dbt_dir.exists():
                logger.warning("dbt project not found at %s — skipping", dbt_dir)
            else:
                result = subprocess.run(
                    ["dbt", "run", "--project-dir", str(dbt_dir)],
                    capture_output=True, text=True,
                )
                logger.info(result.stdout)
                if result.returncode != 0:
                    logger.error(result.stderr)
                    sys.exit(1)

        elif p == "features":
            engine = _get_engine()
            from ingestion.loaders import DatasetLoader
            from features.pipeline import FeaturePipeline
            loader = DatasetLoader(data_dir=DATA_DIR, engine=engine)
            df, _ = loader.load_all(datasets=dataset_list)
            pipeline = FeaturePipeline(engine=engine, persist_to_store=True)
            df_features = pipeline.run(df)
            FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
            cols = pipeline.feature_columns + ["is_fraud", "transaction_id"]
            df_features[[c for c in cols if c in df_features.columns]].to_parquet(FEATURES_PATH, index=False)
            logger.info("Feature matrix saved: %s (%d rows × %d cols)", FEATURES_PATH, len(df_features), len(pipeline.feature_columns))

        elif p == "train":
            if not FEATURES_PATH.exists():
                logger.error("Feature matrix not found at %s — run --phase features first", FEATURES_PATH)
                sys.exit(1)
            from models.train import train_all
            metrics = train_all(FEATURES_PATH, mlflow_uri=mlflow_uri)
            best = max(metrics.items(), key=lambda x: x[1]["auc_pr"])
            logger.info("Best model: %s | AUC-PR=%.4f", best[0], best[1]["auc_pr"])

        else:
            logger.error("Unknown phase: %s", p)
            sys.exit(1)

        logger.info("Phase %s complete ✓", p)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    app()
