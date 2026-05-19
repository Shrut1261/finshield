-- Core fact table: one row per transaction, joined to all dimension keys.
-- Incremental model: only processes rows ingested since the last run.
{{
    config(
        materialized='incremental',
        unique_key='transaction_id',
        on_schema_change='append_new_columns'
    )
}}

WITH staged AS (
    SELECT * FROM {{ ref('stg_raw_transactions') }}
    {% if is_incremental() %}
    WHERE ingested_at > (SELECT MAX(ingested_at) FROM {{ this }})
    {% endif %}
),

with_keys AS (
    SELECT
        s.transaction_id,

        -- Dimension surrogate keys
        dc.customer_key,
        dm.merchant_key,

        -- Measures
        s.amount,
        s.currency,

        -- Labels
        s.is_fraud,
        NULL::VARCHAR(50)                   AS fraud_scenario,

        -- Placeholders for scoring (populated by the API after ingestion)
        NULL::NUMERIC(6,5)                  AS fraud_probability,
        NULL::VARCHAR(20)                   AS risk_tier,
        NULL::VARCHAR(30)                   AS decision,
        NULL::VARCHAR(50)                   AS model_version,

        -- Metadata
        s.source_dataset,
        s.ingested_at,
        NULL::TIMESTAMP                     AS scored_at

    FROM staged s
    LEFT JOIN {{ ref('dim_customer') }} dc
        ON s.customer_id = dc.customer_id AND dc.is_current = TRUE
    LEFT JOIN {{ ref('dim_merchant') }} dm
        ON s.merchant_id = dm.merchant_id
)

SELECT * FROM with_keys
