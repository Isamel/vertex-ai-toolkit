-- =============================================================================
-- stored_procedures.sql
-- Stored procedures and supporting tables for the daily sales ETL pipeline
-- Target schemas: dw, staging (PostgreSQL)
-- =============================================================================

BEGIN;

-- =============================================================================
-- TABLE: dw.agg_daily_store_sales
-- Pre-aggregated daily metrics per store (populated by sp_update_aggregates)
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.agg_daily_store_sales (
    agg_id          BIGSERIAL       PRIMARY KEY,
    sales_date      DATE            NOT NULL,
    store_sk        INTEGER         NOT NULL REFERENCES dw.dim_store (store_sk),
    store_id        VARCHAR(32)     NOT NULL,
    store_name      VARCHAR(128),
    region          VARCHAR(32),
    total_sales     NUMERIC(14, 2)  NOT NULL DEFAULT 0,
    total_units     INTEGER         NOT NULL DEFAULT 0,
    txn_count       INTEGER         NOT NULL DEFAULT 0,
    avg_basket      NUMERIC(12, 2)  NOT NULL DEFAULT 0,
    updated_at      TIMESTAMP       NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_agg_store_date UNIQUE (store_sk, sales_date)
);

CREATE INDEX IF NOT EXISTS idx_agg_sales_date
    ON dw.agg_daily_store_sales (sales_date);
CREATE INDEX IF NOT EXISTS idx_agg_region
    ON dw.agg_daily_store_sales (region);

-- =============================================================================
-- TABLE: staging.archive_daily_sales
-- Long-term archive of staged data, partitioned by month
-- =============================================================================
CREATE TABLE IF NOT EXISTS staging.archive_daily_sales (
    row_id          BIGINT          NOT NULL,
    txn_id          VARCHAR(64)     NOT NULL,
    store_id        VARCHAR(32),
    order_date      DATE            NOT NULL,
    sku             VARCHAR(32)     NOT NULL,
    product_name    VARCHAR(200),
    quantity        INTEGER         NOT NULL,
    unit_price      NUMERIC(12, 2)  NOT NULL,
    discount_pct    NUMERIC(5, 2)   DEFAULT 0,
    discount_amount NUMERIC(12, 2)  DEFAULT 0,
    net_amount      NUMERIC(12, 2),
    gross_amount    NUMERIC(12, 2),
    payment_method  VARCHAR(32),
    payment_category VARCHAR(32),
    source_system   VARCHAR(32)     NOT NULL,
    load_timestamp  TIMESTAMP       NOT NULL,
    category_name   VARCHAR(64),
    region          VARCHAR(32),
    district        VARCHAR(64),
    store_name      VARCHAR(128),
    batch_id        VARCHAR(64)     NOT NULL,
    archived_at     TIMESTAMP       NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (order_date);

-- Create monthly partitions for 2025
CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_01
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_02
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_03
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_04
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_05
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_06
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_07
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_08
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_09
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_10
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_11
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS staging.archive_daily_sales_2025_12
    PARTITION OF staging.archive_daily_sales
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE INDEX IF NOT EXISTS idx_archive_order_date
    ON staging.archive_daily_sales (order_date);
CREATE INDEX IF NOT EXISTS idx_archive_batch_id
    ON staging.archive_daily_sales (batch_id);

-- =============================================================================
-- PROCEDURE 1: sp_update_aggregates
-- Computes daily store-level aggregates and upserts into agg_daily_store_sales.
-- =============================================================================
CREATE OR REPLACE PROCEDURE dw.sp_update_aggregates(p_sales_date DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO dw.agg_daily_store_sales (
        sales_date,
        store_sk,
        store_id,
        store_name,
        region,
        total_sales,
        total_units,
        txn_count,
        avg_basket,
        updated_at
    )
    SELECT
        f.date_key::TEXT::DATE          AS sales_date,
        f.store_sk,
        s.store_id,
        s.store_name,
        s.region,
        SUM(f.net_amount)              AS total_sales,
        SUM(f.quantity)                AS total_units,
        COUNT(DISTINCT f.txn_id)       AS txn_count,
        CASE
            WHEN COUNT(DISTINCT f.txn_id) > 0
            THEN ROUND(SUM(f.net_amount) / COUNT(DISTINCT f.txn_id), 2)
            ELSE 0
        END                            AS avg_basket,
        NOW()                          AS updated_at
    FROM dw.fact_daily_sales f
    JOIN dw.dim_store s ON s.store_sk = f.store_sk
    WHERE f.date_key = TO_CHAR(p_sales_date, 'YYYYMMDD')::INTEGER
    GROUP BY f.date_key, f.store_sk, s.store_id, s.store_name, s.region
    ON CONFLICT (store_sk, sales_date)
    DO UPDATE SET
        total_sales  = EXCLUDED.total_sales,
        total_units  = EXCLUDED.total_units,
        txn_count    = EXCLUDED.txn_count,
        avg_basket   = EXCLUDED.avg_basket,
        updated_at   = NOW();

    RAISE NOTICE '[sp_update_aggregates] Aggregates updated for %', p_sales_date;
END;
$$;

-- =============================================================================
-- PROCEDURE 2: sp_check_data_quality
-- Returns a result set describing data-quality issues for a given sales date.
-- Callers should query: SELECT * FROM dw.sp_check_data_quality('2025-03-15');
-- (Implemented as a function returning SETOF RECORD for Pentaho compatibility.)
-- =============================================================================
DROP FUNCTION IF EXISTS dw.sp_check_data_quality(DATE);

CREATE OR REPLACE FUNCTION dw.sp_check_data_quality(p_sales_date DATE)
RETURNS TABLE (
    check_name        VARCHAR,
    severity          VARCHAR,
    issue_count       BIGINT,
    issue_description TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- 1. Orphan SKUs — in fact table but not in dim_product
    RETURN QUERY
    SELECT
        'orphan_sku'::VARCHAR           AS check_name,
        'ERROR'::VARCHAR                AS severity,
        COUNT(*)                        AS issue_count,
        'SKU(s) in fact_daily_sales not found in dim_product: ' ||
            STRING_AGG(DISTINCT sub.sku_val, ', ')  AS issue_description
    FROM (
        SELECT f.txn_id, p.sku AS sku_val
        FROM dw.fact_daily_sales f
        LEFT JOIN dw.dim_product p ON p.product_sk = f.product_sk AND p.is_current = TRUE
        WHERE f.date_key = TO_CHAR(p_sales_date, 'YYYYMMDD')::INTEGER
          AND p.product_sk IS NULL
    ) sub
    HAVING COUNT(*) > 0;

    -- 2. Negative amounts
    RETURN QUERY
    SELECT
        'negative_amount'::VARCHAR      AS check_name,
        'WARNING'::VARCHAR              AS severity,
        COUNT(*)                        AS issue_count,
        'Rows with negative net_amount or gross_amount'::TEXT AS issue_description
    FROM dw.fact_daily_sales
    WHERE date_key = TO_CHAR(p_sales_date, 'YYYYMMDD')::INTEGER
      AND (net_amount < 0 OR gross_amount < 0)
    HAVING COUNT(*) > 0;

    -- 3. Duplicate txn_id + product_sk combos
    RETURN QUERY
    SELECT
        'duplicate_txn_line'::VARCHAR   AS check_name,
        'ERROR'::VARCHAR                AS severity,
        COUNT(*)                        AS issue_count,
        'Duplicate txn_id + product_sk combinations detected'::TEXT AS issue_description
    FROM (
        SELECT txn_id, product_sk, COUNT(*) AS cnt
        FROM dw.fact_daily_sales
        WHERE date_key = TO_CHAR(p_sales_date, 'YYYYMMDD')::INTEGER
        GROUP BY txn_id, product_sk
        HAVING COUNT(*) > 1
    ) dupes
    HAVING COUNT(*) > 0;

    -- 4. Null required fields
    RETURN QUERY
    SELECT
        'null_required_fields'::VARCHAR AS check_name,
        'ERROR'::VARCHAR                AS severity,
        COUNT(*)                        AS issue_count,
        'Rows with NULL in txn_id, date_key, product_sk, store_sk, quantity, or unit_price'::TEXT
                                        AS issue_description
    FROM dw.fact_daily_sales
    WHERE date_key = TO_CHAR(p_sales_date, 'YYYYMMDD')::INTEGER
      AND (
          txn_id IS NULL
          OR date_key IS NULL
          OR product_sk IS NULL
          OR store_sk IS NULL
          OR quantity IS NULL
          OR unit_price IS NULL
      )
    HAVING COUNT(*) > 0;
END;
$$;

-- =============================================================================
-- PROCEDURE 3: sp_archive_staging
-- Copies stg_daily_sales rows for the given date into the partitioned archive
-- table, then truncates staging.
-- =============================================================================
CREATE OR REPLACE PROCEDURE dw.sp_archive_staging(p_sales_date DATE)
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows_archived BIGINT;
BEGIN
    -- Insert into archive (partition routing is automatic)
    INSERT INTO staging.archive_daily_sales (
        row_id, txn_id, store_id, order_date, sku, product_name,
        quantity, unit_price, discount_pct, discount_amount,
        net_amount, gross_amount, payment_method, payment_category,
        source_system, load_timestamp, category_name,
        region, district, store_name, batch_id, archived_at
    )
    SELECT
        row_id, txn_id, store_id, order_date, sku, product_name,
        quantity, unit_price, discount_pct, discount_amount,
        net_amount, gross_amount, payment_method, payment_category,
        source_system, load_timestamp, category_name,
        region, district, store_name, batch_id, NOW()
    FROM staging.stg_daily_sales
    WHERE order_date = p_sales_date;

    GET DIAGNOSTICS v_rows_archived = ROW_COUNT;

    -- Truncate all staging tables
    TRUNCATE TABLE staging.stg_pos_sales;
    TRUNCATE TABLE staging.stg_ecommerce_sales;
    TRUNCATE TABLE staging.stg_inventory;
    TRUNCATE TABLE staging.stg_daily_sales;

    RAISE NOTICE '[sp_archive_staging] Archived % rows for %, staging truncated.',
                 v_rows_archived, p_sales_date;
END;
$$;

COMMIT;
