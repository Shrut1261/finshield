-- Merchant dimension: one row per unique merchant_id
{{ config(materialized='table') }}

WITH source AS (
    SELECT DISTINCT merchant_id, merchant_category, country
    FROM {{ ref('stg_raw_transactions') }}
    WHERE merchant_id IS NOT NULL
),

enriched AS (
    SELECT
        merchant_id,
        merchant_category,
        country                                                     AS country_code,
        merchant_category IN (
            'wire_transfer', 'crypto', 'atm_withdrawal'
        )                                                           AS is_high_risk_category,
        -- Synthetic merchant name for demo purposes
        CONCAT('Merchant ', UPPER(LEFT(merchant_id, 8)))            AS merchant_name,
        NOW()                                                       AS created_at
    FROM source
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['merchant_id']) }}         AS merchant_key,
    merchant_id,
    merchant_name,
    merchant_category,
    is_high_risk_category,
    country_code,
    created_at
FROM enriched
