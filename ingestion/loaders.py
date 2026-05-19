"""Unified data loaders for all FinShield source datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


@dataclass
class LoadStats:
    dataset: str
    rows_loaded: int
    fraud_count: int
    fraud_rate: float
    columns: list[str] = field(default_factory=list)


class DatasetLoader:
    """Loads, normalizes, and stages source datasets into a common schema.

    Each loader method returns a DataFrame with a minimal common schema plus
    dataset-specific columns. The ``load_all`` method concatenates available
    datasets and tolerates missing files with a warning so partial runs work.
    """

    def __init__(self, data_dir: Path, engine: Optional[Engine] = None) -> None:
        self.data_dir = data_dir
        self.engine = engine

    # ── Individual dataset loaders ────────────────────────────────────────────

    def load_creditcard(self) -> pd.DataFrame:
        """Load ULB/Worldline credit card fraud dataset (285k rows)."""
        path = self.data_dir / "raw" / "creditcard.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Download creditcard.csv to {path}\n"
                "Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

        df = pd.read_csv(path)
        logger.info("creditcard raw shape: %s", df.shape)

        base_ts = pd.Timestamp("2013-09-01 00:00:00")
        df = df.rename(columns={"Class": "is_fraud", "Time": "time_seconds", "Amount": "amount"})
        df["transaction_id"] = "CC_" + df.index.astype(str)
        df["source_dataset"] = "creditcard_ulb"
        df["raw_timestamp"] = base_ts + pd.to_timedelta(df["time_seconds"], unit="s")
        df["customer_id"] = "CUST_CC_" + (df.index % 5000).astype(str).str.zfill(5)
        df["card_id"] = "CARD_CC_" + (df.index % 8000).astype(str).str.zfill(5)
        df["merchant_id"] = "MERCH_CC_" + (df.index % 500).astype(str).str.zfill(4)
        df["merchant_category"] = "online_retail"
        df["country"] = "BE"
        df["currency"] = "EUR"

        return df

    def load_paysim(self) -> pd.DataFrame:
        """Load PaySim mobile money simulation dataset (6.3M rows)."""
        candidates = sorted((self.data_dir / "raw").glob("PS_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "PaySim CSV not found in data/raw/\n"
                "Source: https://www.kaggle.com/datasets/ealaxi/paysim1"
            )

        df = pd.read_csv(candidates[0])
        logger.info("paysim raw shape: %s", df.shape)

        df = df.rename(
            columns={
                "isFraud": "is_fraud",
                "type": "merchant_category",
                "nameOrig": "customer_id",
                "nameDest": "merchant_id",
            }
        )
        df["transaction_id"] = "PS_" + df.index.astype(str)
        df["source_dataset"] = "paysim"
        df["raw_timestamp"] = pd.Timestamp("2022-01-01") + pd.to_timedelta(df["step"], unit="h")
        df["amount"] = df["amount"].round(2)
        df["card_id"] = "CARD_PS_" + df["customer_id"].str[1:].str.zfill(8)
        df["country"] = "KE"
        df["currency"] = "KES"
        df["merchant_category"] = df["merchant_category"].str.lower()

        return df

    def load_ieee(self, nrows: int = 200_000) -> pd.DataFrame:
        """Load IEEE-CIS fraud detection dataset (590k rows, joined with identity)."""
        tx_path = self.data_dir / "raw" / "train_transaction.csv"
        id_path = self.data_dir / "raw" / "train_identity.csv"

        if not tx_path.exists():
            raise FileNotFoundError(
                f"Download train_transaction.csv to {tx_path}\n"
                "Source: https://www.kaggle.com/c/ieee-fraud-detection"
            )

        tx = pd.read_csv(tx_path, nrows=nrows)
        logger.info("ieee transactions raw shape: %s", tx.shape)

        if id_path.exists():
            identity = pd.read_csv(id_path)
            df = tx.merge(identity, on="TransactionID", how="left")
            logger.info("ieee merged with identity: %s", df.shape)
        else:
            logger.warning("train_identity.csv not found — proceeding without identity features")
            df = tx

        df = df.rename(
            columns={
                "TransactionID": "_tx_id_raw",
                "isFraud": "is_fraud",
                "TransactionAmt": "amount",
                "ProductCD": "merchant_category",
                "card1": "card_id",
            }
        )
        df["transaction_id"] = "IEEE_" + df["_tx_id_raw"].astype(str)
        df["source_dataset"] = "ieee_cis"
        df["raw_timestamp"] = pd.Timestamp("2017-12-01") + pd.to_timedelta(
            df["TransactionDT"], unit="s"
        )
        df["customer_id"] = "CUST_IEEE_" + df["card_id"].astype(str).str.zfill(6)
        df["merchant_id"] = "MERCH_IEEE_" + df.get("addr1", pd.Series(dtype=str)).fillna("0").astype(str)
        df["amount"] = df["amount"].round(2)
        df["country"] = "US"
        df["currency"] = "USD"
        df["merchant_category"] = df["merchant_category"].fillna("other").str.lower()

        return df

    def load_all(
        self,
        datasets: Optional[list[str]] = None,
        ieee_nrows: int = 200_000,
    ) -> tuple[pd.DataFrame, list[LoadStats]]:
        """Load and concatenate all available datasets.

        Returns the combined DataFrame and per-dataset load statistics.
        Missing files emit a warning but do not abort the run.
        """
        loaders: dict[str, Any] = {
            "creditcard": self.load_creditcard,
            "paysim": self.load_paysim,
            "ieee": lambda: self.load_ieee(nrows=ieee_nrows),
        }
        datasets = datasets or list(loaders.keys())

        frames: list[pd.DataFrame] = []
        stats: list[LoadStats] = []

        for name in datasets:
            try:
                df = loaders[name]()
                fraud_count = int(df["is_fraud"].sum())
                stats.append(
                    LoadStats(
                        dataset=name,
                        rows_loaded=len(df),
                        fraud_count=fraud_count,
                        fraud_rate=df["is_fraud"].mean(),
                        columns=df.columns.tolist(),
                    )
                )
                frames.append(df)
                logger.info(
                    "Loaded %s: %d rows, fraud_rate=%.4f%%",
                    name,
                    len(df),
                    df["is_fraud"].mean() * 100,
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", name, exc)

        if not frames:
            raise RuntimeError(
                "No datasets loaded. Download at least one dataset to data/raw/.\n"
                "See data/raw/.gitkeep for download instructions."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Combined dataset: %d total rows", len(combined))
        return combined, stats

    # ── Staging ingestion ─────────────────────────────────────────────────────

    def ingest_to_staging(
        self,
        df: pd.DataFrame,
        table: str = "stg_raw_transactions",
        if_exists: str = "append",
    ) -> int:
        """Write DataFrame to staging table and return number of rows written."""
        if self.engine is None:
            raise RuntimeError("engine must be provided to use ingest_to_staging")

        # Keep only columns that exist in the target table to avoid schema drift
        safe_cols = [
            "transaction_id", "customer_id", "card_id", "merchant_id",
            "merchant_category", "amount", "currency", "country",
            "raw_timestamp", "is_fraud", "source_dataset",
        ]
        cols_to_write = [c for c in safe_cols if c in df.columns]
        subset = df[cols_to_write].copy()

        subset.to_sql(
            table,
            self.engine,
            schema="staging",
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=5_000,
        )
        logger.info("Ingested %d rows → staging.%s", len(subset), table)
        return len(subset)


# Allow direct execution for quick data validation
if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    loader = DatasetLoader(data_dir=Path("data"))
    df, stats = loader.load_all()
    for s in stats:
        print(f"{s.dataset}: {s.rows_loaded:,} rows | fraud_rate={s.fraud_rate:.4%}")
    print(f"\nTotal: {len(df):,} rows")
