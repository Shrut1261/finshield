-- Fraud KPI mart: pre-aggregated metrics consumed by the BI dashboard.
-- Refreshed daily (or on-demand) via dbt run --select mart_fraud_kpis
{{ config(materialized='table') }}

WITH base AS (
    SELECT
        ft.transaction_key,
        ft.amount,
        ft.is_fraud,
        ft.fraud_probability,
        ft.risk_tier,
        ft.decision,
        ft.source_dataset,
        ft.ingested_at,
        dm.category                             AS merchant_category,
        dm.is_high_risk_category,
        dc.home_country,
        -- Time bucketing
        DATE_TRUNC('day',  ft.ingested_at)      AS day_bucket,
        DATE_TRUNC('week', ft.ingested_at)      AS week_bucket,
        EXTRACT(HOUR FROM ft.ingested_at)::INT  AS hour_of_day
    FROM {{ ref('fact_transactions') }} ft
    LEFT JOIN {{ ref('dim_merchant') }} dm ON ft.merchant_key = dm.merchant_key
    LEFT JOIN {{ ref('dim_customer') }} dc ON ft.customer_key = dc.customer_key
                                          AND dc.is_current = TRUE
),

daily_summary AS (
    SELECT
        day_bucket,
        merchant_category,
        home_country,
        hour_of_day,
        source_dataset,

        -- Volume
        COUNT(*)                                        AS total_transactions,
        COUNT(*) FILTER (WHERE is_fraud = 1)            AS fraud_transactions,
        COUNT(*) FILTER (WHERE is_fraud = 0)            AS legit_transactions,

        -- Dollar amounts
        SUM(amount)                                     AS total_amount,
        SUM(amount) FILTER (WHERE is_fraud = 1)         AS fraud_amount,
        AVG(amount) FILTER (WHERE is_fraud = 1)         AS avg_fraud_amount,
        AVG(amount) FILTER (WHERE is_fraud = 0)         AS avg_legit_amount,

        -- Risk tier distribution
        COUNT(*) FILTER (WHERE risk_tier = 'low')       AS risk_low,
        COUNT(*) FILTER (WHERE risk_tier = 'medium')    AS risk_medium,
        COUNT(*) FILTER (WHERE risk_tier = 'high')      AS risk_high,
        COUNT(*) FILTER (WHERE risk_tier = 'critical')  AS risk_critical,

        -- Decision distribution
        COUNT(*) FILTER (WHERE decision = 'auto_approve')  AS auto_approved,
        COUNT(*) FILTER (WHERE decision = 'manual_review') AS manual_review,
        COUNT(*) FILTER (WHERE decision = 'auto_decline')  AS auto_declined

    FROM base
    GROUP BY 1, 2, 3, 4, 5
)

SELECT
    *,
    ROUND(
        100.0 * fraud_transactions / NULLIF(total_transactions, 0),
        4
    )                                                   AS fraud_rate_pct,
    -- Estimated $ saved: fraud_amount * industry avg recovery rate (30%)
    ROUND(fraud_amount * 0.30, 2)                       AS estimated_loss_prevented
FROM daily_summary
