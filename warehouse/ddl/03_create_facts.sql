-- FinShield: Fact Table + ML scoring history

-- ── fact_transactions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse.fact_transactions (
    transaction_key   BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(100) NOT NULL UNIQUE,

    -- Foreign keys to dimensions
    customer_key      INT REFERENCES warehouse.dim_customer(customer_key),
    merchant_key      INT REFERENCES warehouse.dim_merchant(merchant_key),
    card_key          INT REFERENCES warehouse.dim_card(card_key),
    time_key          INT REFERENCES warehouse.dim_time(time_key),
    geography_key     INT REFERENCES warehouse.dim_geography(geography_key),

    -- Measures
    amount            NUMERIC(15, 2) NOT NULL,
    currency          CHAR(3)        NOT NULL DEFAULT 'USD',

    -- Ground-truth fraud label (from dataset)
    is_fraud          SMALLINT       NOT NULL DEFAULT 0 CHECK (is_fraud IN (0, 1)),
    fraud_scenario    VARCHAR(50),

    -- ML scoring output (populated asynchronously after ingestion)
    fraud_probability NUMERIC(6, 5),
    risk_tier         VARCHAR(20),
    decision          VARCHAR(30),
    model_version     VARCHAR(50),

    -- Metadata
    source_dataset    VARCHAR(50)    NOT NULL,
    ingested_at       TIMESTAMP      NOT NULL DEFAULT NOW(),
    scored_at         TIMESTAMP
);

-- ── Indexes — documented rationale in docs/architecture.md ───────────────────
CREATE INDEX IF NOT EXISTS idx_fact_tx_customer
    ON warehouse.fact_transactions (customer_key);
CREATE INDEX IF NOT EXISTS idx_fact_tx_merchant
    ON warehouse.fact_transactions (merchant_key);
CREATE INDEX IF NOT EXISTS idx_fact_tx_time
    ON warehouse.fact_transactions (time_key);
CREATE INDEX IF NOT EXISTS idx_fact_tx_fraud
    ON warehouse.fact_transactions (is_fraud)
    WHERE is_fraud = 1;
CREATE INDEX IF NOT EXISTS idx_fact_tx_amount
    ON warehouse.fact_transactions (amount);
CREATE INDEX IF NOT EXISTS idx_fact_tx_ingested
    ON warehouse.fact_transactions (ingested_at DESC);
-- Covering index for velocity feature SQL queries
CREATE INDEX IF NOT EXISTS idx_fact_tx_velocity
    ON warehouse.fact_transactions (customer_key, ingested_at DESC)
    INCLUDE (amount, is_fraud);

-- ── ml.model_scores: full audit log of every score ───────────────────────────
CREATE TABLE IF NOT EXISTS ml.model_scores (
    score_id          BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(100) NOT NULL,
    model_name        VARCHAR(100) NOT NULL,
    model_version     VARCHAR(50)  NOT NULL,
    fraud_probability NUMERIC(6, 5) NOT NULL,
    risk_tier         VARCHAR(20)  NOT NULL,
    decision          VARCHAR(30)  NOT NULL,
    latency_ms        NUMERIC(8, 2),
    shap_values       JSONB,
    scored_at         TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scores_tx_id ON ml.model_scores (transaction_id);
CREATE INDEX IF NOT EXISTS idx_scores_time  ON ml.model_scores (scored_at DESC);

-- ── ml.drift_metrics: weekly PSI per feature ─────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.drift_metrics (
    drift_id       BIGSERIAL PRIMARY KEY,
    feature_name   VARCHAR(100) NOT NULL,
    model_version  VARCHAR(50)  NOT NULL,
    psi_score      NUMERIC(8, 5) NOT NULL,
    status         VARCHAR(20)  NOT NULL,  -- stable / monitor / retrain
    week_start     DATE         NOT NULL,
    computed_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
    UNIQUE (feature_name, model_version, week_start)
);

-- ── features.feature_vectors ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features.feature_vectors (
    feature_vector_id BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(100) NOT NULL UNIQUE,
    feature_version   VARCHAR(20)  NOT NULL DEFAULT 'v1',
    computed_at       TIMESTAMP    NOT NULL DEFAULT NOW(),
    features          JSONB        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fv_tx_id      ON features.feature_vectors (transaction_id);
CREATE INDEX IF NOT EXISTS idx_fv_version    ON features.feature_vectors (feature_version);
CREATE INDEX IF NOT EXISTS idx_fv_computed   ON features.feature_vectors (computed_at DESC);
-- GIN index enables fast JSONB key/value lookups
CREATE INDEX IF NOT EXISTS idx_fv_features   ON features.feature_vectors USING GIN (features);
