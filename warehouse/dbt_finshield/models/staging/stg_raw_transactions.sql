-- Staging model: normalize raw ingested transactions into a clean intermediate layer.
-- Deduplicates by transaction_id and casts all types.
{{ config(materialized='view') }}

WITH deduplicated AS (
    SELECT
        transaction_id,
        customer_id,
        card_id,
        merchant_id,
        COALESCE(LOWER(TRIM(merchant_category)), 'other')   AS merchant_category,
        ROUND(amount::NUMERIC, 2)                           AS amount,
        UPPER(COALESCE(currency, 'USD'))                    AS currency,
        UPPER(COALESCE(country, 'US'))                      AS country,
        raw_timestamp::TIMESTAMP                            AS raw_timestamp,
        is_fraud::SMALLINT                                  AS is_fraud,
        source_dataset,
        ingested_at,
        ROW_NUMBER() OVER (
            PARTITION BY transaction_id
            ORDER BY ingested_at DESC
        ) AS _row_num
    FROM {{ source('staging', 'stg_raw_transactions') }}
    WHERE
        transaction_id IS NOT NULL
        AND amount > 0
        AND amount <= 1000000
)

SELECT
    transaction_id,
    customer_id,
    card_id,
    merchant_id,
    merchant_category,
    amount,
    currency,
    country,
    raw_timestamp,
    EXTRACT(HOUR FROM raw_timestamp)::SMALLINT        AS hour_of_day,
    EXTRACT(DOW  FROM raw_timestamp)::SMALLINT        AS day_of_week,
    (EXTRACT(DOW FROM raw_timestamp) IN (0, 6))       AS is_weekend,
    is_fraud,
    source_dataset,
    ingested_at
FROM deduplicated
WHERE _row_num = 1
