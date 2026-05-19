-- FinShield: Dimension Tables
-- dim_time, dim_geography, dim_merchant, dim_card, dim_customer (SCD Type 2)

-- ── dim_time ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse.dim_time (
    time_key          SERIAL PRIMARY KEY,
    full_datetime     TIMESTAMP NOT NULL UNIQUE,
    date_key          DATE NOT NULL,
    hour              SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    minute            SMALLINT NOT NULL CHECK (minute BETWEEN 0 AND 59),
    day_of_week       SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    day_name          VARCHAR(10) NOT NULL,
    week_of_year      SMALLINT NOT NULL,
    month             SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name        VARCHAR(10) NOT NULL,
    quarter           SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    year              SMALLINT NOT NULL,
    is_weekend        BOOLEAN NOT NULL DEFAULT FALSE,
    is_business_hours BOOLEAN NOT NULL DEFAULT FALSE
);

COMMENT ON TABLE warehouse.dim_time IS
    'Calendar dimension — one row per timestamp minute; pre-populated by setup_db.py';

-- ── dim_geography ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse.dim_geography (
    geography_key       SERIAL PRIMARY KEY,
    country_code        CHAR(2)       NOT NULL,
    country_name        VARCHAR(100),
    region              VARCHAR(100),
    city                VARCHAR(100),
    latitude            NUMERIC(9, 6),
    longitude           NUMERIC(9, 6),
    is_high_risk_country BOOLEAN      NOT NULL DEFAULT FALSE,
    UNIQUE (country_code, city)
);

-- ── dim_merchant ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse.dim_merchant (
    merchant_key          SERIAL PRIMARY KEY,
    merchant_id           VARCHAR(100) NOT NULL UNIQUE,
    merchant_name         VARCHAR(200),
    category              VARCHAR(50)  NOT NULL,
    is_high_risk_category BOOLEAN      NOT NULL DEFAULT FALSE,
    country_code          CHAR(2),
    registered_at         DATE,
    created_at            TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_merchant_category ON warehouse.dim_merchant (category);

-- ── dim_card ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse.dim_card (
    card_key     SERIAL PRIMARY KEY,
    card_id      VARCHAR(100) NOT NULL UNIQUE,
    card_type    VARCHAR(20),   -- credit / debit / prepaid
    card_network VARCHAR(20),   -- visa / mastercard / amex / discover
    expiry_year  SMALLINT,
    is_active    BOOLEAN        NOT NULL DEFAULT TRUE,
    issued_at    DATE,
    created_at   TIMESTAMP      NOT NULL DEFAULT NOW()
);

-- ── dim_customer: SCD Type 2 ──────────────────────────────────────────────────
-- When a customer's risk_score or home_country changes, the old row is expired
-- (effective_to set, is_current = FALSE) and a new row is inserted.
CREATE TABLE IF NOT EXISTS warehouse.dim_customer (
    customer_key   SERIAL PRIMARY KEY,
    customer_id    VARCHAR(100) NOT NULL,
    customer_name  VARCHAR(200),
    email_domain   VARCHAR(100),
    age_bucket     VARCHAR(20),    -- 18-25, 26-35, 36-50, 51+
    risk_score     NUMERIC(5, 4)  NOT NULL DEFAULT 0.0,
    home_country   CHAR(2),
    -- SCD Type 2 versioning columns
    effective_from TIMESTAMP      NOT NULL DEFAULT NOW(),
    effective_to   TIMESTAMP,
    is_current     BOOLEAN        NOT NULL DEFAULT TRUE,
    UNIQUE (customer_id, effective_from)
);

CREATE INDEX IF NOT EXISTS idx_dim_customer_id_current
    ON warehouse.dim_customer (customer_id)
    WHERE is_current = TRUE;
