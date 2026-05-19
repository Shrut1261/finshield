"""Database setup script: creates schemas, tables, and indexes.

Run once after starting the PostgreSQL container:
    python scripts/setup_db.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DDL_DIR = Path(__file__).parent.parent / "warehouse" / "ddl"
DDL_FILES = [
    "01_create_schemas.sql",
    "02_create_dimensions.sql",
    "03_create_facts.sql",
]


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=30))
def get_engine():
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://finshield_user:finshield_dev_password@localhost:5432/finshield",
    )
    engine = create_engine(db_url, echo=False)
    with engine.connect():
        pass
    return engine


def run_ddl(engine, path: Path) -> None:
    sql = path.read_text()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("Executed: %s", path.name)


def main() -> None:
    logger.info("Connecting to database ...")
    engine = get_engine()
    logger.info("Connected. Running DDL scripts ...")

    for filename in DDL_FILES:
        path = DDL_DIR / filename
        if path.exists():
            run_ddl(engine, path)
        else:
            logger.warning("DDL file not found: %s", path)

    logger.info("Database setup complete.")


if __name__ == "__main__":
    main()
