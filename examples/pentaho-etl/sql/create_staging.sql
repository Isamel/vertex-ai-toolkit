-- =============================================================================
-- create_staging.sql
-- Staging tables for the daily sales ETL pipeline
-- Target schema: staging (PostgreSQL)
-- These tables are ephemeral — truncated at the start of every ETL run.
-- =============================================================================

BEGIN;

-- ---- Truncate all staging tables (idempotent refresh) -----------------------
TRUNCATE TABLE IF EXISTS staging.stg_pos_sales;
TRUNCATE TABLE IF EXISTS staging.stg_ecommerce_sales;
TRUNCATE TABLE IF EXISTS staging.stg_inventory;
TRUNCATE TABLE IF EXISTS staging.stg_daily_sales;

-- ---- Schema -----------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS staging;

-- =============================================================================
-- 1. stg_pos_sales — raw POS terminal transactions
-- =============================================================================
CREATE TABLE IF NOT EXISTS staging.stg_pos_sales (
    txn_id          VARCHAR(64)     NOT NULL,
    store_id        VARCHAR(32)     NOT NULL,
    order_date      DATE            NOT NULL,
    order_time      TIME            NOT NULL,
    sku             VARCHAR(32)     NOT NULL,
    product_name    VARCHAR(200)    NOT NULL,
    quantity        INTEGER         NOT NULL,
    unit_price      NUMERIC(12, 2)  NOT NULL,
    discount_pct    NUMERIC(5, 2)   DEFAULT 0,
    payment_method  VARCHAR(32),
    source_file     VARCHAR(256),
    source_system   VARCHAR(32)     NOT NULL DEFAULT 'POS',
    load_timestamp  TIMESTAMP       NOT NULL DEFAULT NOW(),
    batch_id        VARCHAR(64)     NOT NULL
);

-- =============================================================================
-- 2. stg_ecommerce_sales — online order lines
-- =============================================================================
CREATE TABLE IF NOT EXISTS staging.stg_ecommerce_sales (
    txn_id          VARCHAR(64)     NOT NULL,
    customer_id     VARCHAR(64),
    order_date      DATE            NOT NULL,
    status          VARCHAR(32)     NOT NULL,
    ship_city       VARCHAR(128),
    ship_state      VARCHAR(64),
    sku             VARCHAR(32)     NOT NULL,
    product_name    VARCHAR(200)    NOT NULL,
    quantity        INTEGER         NOT NULL,
    unit_price      NUMERIC(12, 2)  NOT NULL,
    discount_amount NUMERIC(12, 2)  DEFAULT 0,
    payment_method  VARCHAR(32),
    tax_amount      NUMERIC(12, 2)  DEFAULT 0,
    line_total      NUMERIC(12, 2),
    net_total       NUMERIC(12, 2),
    source_system   VARCHAR(32)     NOT NULL DEFAULT 'ECOMMERCE',
    load_timestamp  TIMESTAMP       NOT NULL DEFAULT NOW(),
    batch_id        VARCHAR(64)     NOT NULL
);

-- =============================================================================
-- 3. stg_inventory — nightly inventory snapshot from source system
-- =============================================================================
CREATE TABLE IF NOT EXISTS staging.stg_inventory (
    sku             VARCHAR(32)     NOT NULL,
    product_name    VARCHAR(200)    NOT NULL,
    category_id     INTEGER,
    unit_cost       NUMERIC(12, 2),
    supplier_id     VARCHAR(32),
    warehouse_id    VARCHAR(32),
    stock_qty       INTEGER         NOT NULL DEFAULT 0,
    reserved_qty    INTEGER         NOT NULL DEFAULT 0,
    reorder_point   INTEGER,
    last_restock_date DATE,
    source_system   VARCHAR(32)     NOT NULL DEFAULT 'INVENTORY',
    load_timestamp  TIMESTAMP       NOT NULL DEFAULT NOW(),
    batch_id        VARCHAR(64)     NOT NULL
);

-- =============================================================================
-- 4. stg_daily_sales — unified staging (POS + e-commerce merged & enriched)
-- =============================================================================
CREATE TABLE IF NOT EXISTS staging.stg_daily_sales (
    row_id          BIGSERIAL       PRIMARY KEY,
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
    load_timestamp  TIMESTAMP       NOT NULL DEFAULT NOW(),
    category_name   VARCHAR(64),
    region          VARCHAR(32),
    district        VARCHAR(64),
    store_name      VARCHAR(128),
    batch_id        VARCHAR(64)     NOT NULL
);

-- ---- Indexes on lookup / join keys ------------------------------------------

-- stg_pos_sales
CREATE INDEX IF NOT EXISTS idx_stg_pos_txn_sku
    ON staging.stg_pos_sales (txn_id, sku);
CREATE INDEX IF NOT EXISTS idx_stg_pos_store
    ON staging.stg_pos_sales (store_id);
CREATE INDEX IF NOT EXISTS idx_stg_pos_order_date
    ON staging.stg_pos_sales (order_date);

-- stg_ecommerce_sales
CREATE INDEX IF NOT EXISTS idx_stg_ecom_txn_sku
    ON staging.stg_ecommerce_sales (txn_id, sku);
CREATE INDEX IF NOT EXISTS idx_stg_ecom_order_date
    ON staging.stg_ecommerce_sales (order_date);

-- stg_inventory
CREATE INDEX IF NOT EXISTS idx_stg_inv_sku
    ON staging.stg_inventory (sku);

-- stg_daily_sales
CREATE INDEX IF NOT EXISTS idx_stg_daily_txn_sku
    ON staging.stg_daily_sales (txn_id, sku);
CREATE INDEX IF NOT EXISTS idx_stg_daily_store
    ON staging.stg_daily_sales (store_id);
CREATE INDEX IF NOT EXISTS idx_stg_daily_order_date
    ON staging.stg_daily_sales (order_date);

COMMIT;
