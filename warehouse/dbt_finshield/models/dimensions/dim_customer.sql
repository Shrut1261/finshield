-- Customer dimension with SCD Type 2 support.
-- Each customer gets a surrogate key per version; is_current flags the live row.
{{ config(materialized='table') }}

WITH base AS (
    SELECT DISTINCT
        customer_id,
        country                                     AS home_country,
        source_dataset,
        MIN(raw_timestamp) OVER (PARTITION BY customer_id) AS first_seen
    FROM {{ ref('stg_raw_transactions') }}
    WHERE customer_id IS NOT NULL
),

-- In a real SCD Type 2 pipeline, previous versions would be loaded from a
-- snapshot table. Here we create the current-version snapshot.
snapshot AS (
    SELECT
        customer_id,
        home_country,
        first_seen::TIMESTAMP                       AS effective_from,
        NULL::TIMESTAMP                             AS effective_to,
        TRUE                                        AS is_current,
        -- Derived attributes
        CASE
            WHEN source_dataset = 'creditcard_ulb' THEN '36-50'
            WHEN source_dataset = 'paysim'         THEN '18-25'
            ELSE '26-35'
        END                                         AS age_bucket,
        0.0::NUMERIC(5,4)                           AS risk_score
    FROM base
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['customer_id', 'effective_from']) }} AS customer_key,
    customer_id,
    CONCAT('Customer ', customer_id)                AS customer_name,
    'gmail.com'                                     AS email_domain,
    age_bucket,
    risk_score,
    home_country,
    effective_from,
    effective_to,
    is_current
FROM snapshot
