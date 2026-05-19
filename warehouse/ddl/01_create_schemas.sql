-- FinShield Data Warehouse: Schema Bootstrap
-- Run order: 01 → 02 → 03 → 04
-- Compatible with PostgreSQL 13+

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS warehouse;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS ml;

COMMENT ON SCHEMA staging   IS 'Raw ingested data before transformation';
COMMENT ON SCHEMA warehouse IS 'Star schema DWH: facts + dimensions';
COMMENT ON SCHEMA features  IS 'Computed feature vectors per transaction';
COMMENT ON SCHEMA ml        IS 'Model scores, predictions, drift metrics';

-- ── Staging table ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS staging.stg_raw_transactions (
    id                BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(100),
    customer_id       VARCHAR(100),
    card_id           VARCHAR(100),
    merchant_id       VARCHAR(100),
    merchant_category VARCHAR(50),
    amount            NUMERIC(15, 2),
    currency          CHAR(3),
    country           CHAR(2),
    raw_timestamp     TIMESTAMP,
    is_fraud          SMALLINT,
    source_dataset    VARCHAR(50),
    ingested_at       TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_ingested_at ON staging.stg_raw_transactions (ingested_at DESC);
