-- =============================================================================
-- create_warehouse.sql
-- Star-schema warehouse tables for the daily sales ETL pipeline
-- Target schema: dw (PostgreSQL)
-- =============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS dw;

-- =============================================================================
-- DIMENSION: dim_date
-- Grain: one row per calendar day
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key        INTEGER         PRIMARY KEY,           -- YYYYMMDD
    full_date       DATE            NOT NULL UNIQUE,
    day_of_week     SMALLINT        NOT NULL,              -- 1=Mon … 7=Sun (ISO)
    day_name        VARCHAR(10)     NOT NULL,
    month           SMALLINT        NOT NULL,
    month_name      VARCHAR(10)     NOT NULL,
    quarter         SMALLINT        NOT NULL,
    year            SMALLINT        NOT NULL,
    fiscal_quarter  SMALLINT        NOT NULL,
    fiscal_year     SMALLINT        NOT NULL,
    is_weekend      BOOLEAN         NOT NULL DEFAULT FALSE,
    is_holiday      BOOLEAN         NOT NULL DEFAULT FALSE
);

-- =============================================================================
-- DIMENSION: dim_product (SCD Type 2)
-- Grain: one row per product version
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.dim_product (
    product_sk      SERIAL          PRIMARY KEY,
    sku             VARCHAR(32)     NOT NULL,
    product_name    VARCHAR(200)    NOT NULL,
    category_id     INTEGER,
    category_name   VARCHAR(64),
    subcategory     VARCHAR(64),
    brand           VARCHAR(64),
    unit_cost       NUMERIC(12, 2),
    price_tier      VARCHAR(16),                           -- Budget / Mid / Premium
    effective_date  DATE            NOT NULL DEFAULT CURRENT_DATE,
    expiry_date     DATE            NOT NULL DEFAULT '9999-12-31',
    is_current      BOOLEAN         NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_dim_product_sku_effective UNIQUE (sku, effective_date)
);

CREATE INDEX IF NOT EXISTS idx_dim_product_sku
    ON dw.dim_product (sku) WHERE is_current = TRUE;

-- =============================================================================
-- DIMENSION: dim_store
-- Grain: one row per store location
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.dim_store (
    store_sk        SERIAL          PRIMARY KEY,
    store_id        VARCHAR(32)     NOT NULL UNIQUE,
    store_name      VARCHAR(128)    NOT NULL,
    region          VARCHAR(32)     NOT NULL,
    district        VARCHAR(64),
    city            VARCHAR(128),
    state           VARCHAR(64),
    open_date       DATE,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_dim_store_region
    ON dw.dim_store (region);

-- =============================================================================
-- DIMENSION: dim_customer
-- Grain: one row per customer
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.dim_customer (
    customer_sk         SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(64)     NOT NULL UNIQUE,
    first_name          VARCHAR(64),
    last_name           VARCHAR(64),
    email               VARCHAR(256),
    loyalty_tier        VARCHAR(16),                       -- Bronze / Silver / Gold / Platinum
    registration_date   DATE
);

-- =============================================================================
-- FACT: fact_daily_sales
-- Grain: one row per transaction line item per day
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.fact_daily_sales (
    fact_sale_sk        BIGSERIAL       PRIMARY KEY,
    date_key            INTEGER         NOT NULL,
    product_sk          INTEGER         NOT NULL,
    store_sk            INTEGER         NOT NULL,
    customer_sk         INTEGER,                           -- nullable for POS w/o loyalty
    txn_id              VARCHAR(64)     NOT NULL,
    quantity            INTEGER         NOT NULL,
    unit_price          NUMERIC(12, 2)  NOT NULL,
    gross_amount        NUMERIC(12, 2)  NOT NULL,
    discount_amount     NUMERIC(12, 2)  DEFAULT 0,
    net_amount          NUMERIC(12, 2)  NOT NULL,
    tax_amount          NUMERIC(12, 2)  DEFAULT 0,
    payment_method      VARCHAR(32),
    payment_category    VARCHAR(32),
    source_system       VARCHAR(32)     NOT NULL,
    loyalty_points      INTEGER         DEFAULT 0,
    is_high_value_txn   BOOLEAN         DEFAULT FALSE,
    txn_hash            VARCHAR(64),
    load_timestamp      TIMESTAMP       NOT NULL DEFAULT NOW(),
    batch_id            VARCHAR(64)     NOT NULL,

    -- Foreign key constraints
    CONSTRAINT fk_fact_date
        FOREIGN KEY (date_key) REFERENCES dw.dim_date (date_key),
    CONSTRAINT fk_fact_product
        FOREIGN KEY (product_sk) REFERENCES dw.dim_product (product_sk),
    CONSTRAINT fk_fact_store
        FOREIGN KEY (store_sk) REFERENCES dw.dim_store (store_sk),
    CONSTRAINT fk_fact_customer
        FOREIGN KEY (customer_sk) REFERENCES dw.dim_customer (customer_sk)
);

-- Indexes on FK columns and commonly filtered columns
CREATE INDEX IF NOT EXISTS idx_fact_date_key
    ON dw.fact_daily_sales (date_key);
CREATE INDEX IF NOT EXISTS idx_fact_product_sk
    ON dw.fact_daily_sales (product_sk);
CREATE INDEX IF NOT EXISTS idx_fact_store_sk
    ON dw.fact_daily_sales (store_sk);
CREATE INDEX IF NOT EXISTS idx_fact_customer_sk
    ON dw.fact_daily_sales (customer_sk);
CREATE INDEX IF NOT EXISTS idx_fact_txn_id
    ON dw.fact_daily_sales (txn_id);
CREATE INDEX IF NOT EXISTS idx_fact_source_system
    ON dw.fact_daily_sales (source_system);
CREATE INDEX IF NOT EXISTS idx_fact_batch_id
    ON dw.fact_daily_sales (batch_id);

-- =============================================================================
-- SEED DATA: dim_product (6 core products)
-- =============================================================================
INSERT INTO dw.dim_product (sku, product_name, category_id, category_name, subcategory, brand, unit_cost, price_tier, effective_date, expiry_date, is_current)
VALUES
    ('SKU-10042', 'Organic Whole Milk',  1, 'Dairy',     'Milk',          'Green Valley', 2.50, 'Mid',     '2025-01-01', '9999-12-31', TRUE),
    ('SKU-20015', 'Sourdough Bread',     2, 'Bakery',    'Artisan Bread', 'Hearth & Co',  1.80, 'Mid',     '2025-01-01', '9999-12-31', TRUE),
    ('SKU-30088', 'Coffee Beans',        3, 'Beverages', 'Coffee',        'Dark Roast Co',5.20, 'Premium', '2025-01-01', '9999-12-31', TRUE),
    ('SKU-40022', 'Free-Range Eggs',     5, 'Eggs',      'Shell Eggs',    'Happy Hen',    3.10, 'Mid',     '2025-01-01', '9999-12-31', TRUE),
    ('SKU-50001', 'Avocado',             4, 'Produce',   'Fruit',         'SunFresh',     1.20, 'Budget',  '2025-01-01', '9999-12-31', TRUE),
    ('SKU-60010', 'Sparkling Water',     3, 'Beverages', 'Water',         'CrystalSpring',0.80, 'Budget',  '2025-01-01', '9999-12-31', TRUE)
ON CONFLICT (sku, effective_date) DO NOTHING;

-- =============================================================================
-- SEED DATA: dim_store (4 stores across regions)
-- =============================================================================
INSERT INTO dw.dim_store (store_id, store_name, region, district, city, state, open_date, is_active)
VALUES
    ('STR-101', 'Downtown Market',    'NORTHEAST', 'Metro NY',       'New York',      'NY', '2018-03-15', TRUE),
    ('STR-102', 'Lakeside Grocery',   'MIDWEST',   'Great Lakes',    'Chicago',       'IL', '2019-07-01', TRUE),
    ('STR-103', 'Sunshine Foods',     'SOUTHEAST', 'Atlantic South', 'Atlanta',       'GA', '2020-01-10', TRUE),
    ('STR-104', 'Pacific Fresh',      'WEST',      'Pacific NW',     'Portland',      'OR', '2021-05-22', TRUE)
ON CONFLICT (store_id) DO NOTHING;

-- =============================================================================
-- SEED DATA: dim_date (March 2025)
-- =============================================================================
INSERT INTO dw.dim_date (
    date_key, full_date, day_of_week, day_name,
    month, month_name, quarter, year,
    fiscal_quarter, fiscal_year, is_weekend, is_holiday
)
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INTEGER                    AS date_key,
    d                                                    AS full_date,
    EXTRACT(ISODOW FROM d)::SMALLINT                    AS day_of_week,
    TO_CHAR(d, 'Day')                                   AS day_name,
    EXTRACT(MONTH FROM d)::SMALLINT                     AS month,
    TO_CHAR(d, 'Month')                                 AS month_name,
    EXTRACT(QUARTER FROM d)::SMALLINT                   AS quarter,
    EXTRACT(YEAR FROM d)::SMALLINT                      AS year,
    -- Fiscal year starts February (Q1 = Feb-Apr)
    CASE
        WHEN EXTRACT(MONTH FROM d) >= 2
        THEN ((EXTRACT(MONTH FROM d)::INT - 2) / 3) + 1
        ELSE 4
    END::SMALLINT                                       AS fiscal_quarter,
    CASE
        WHEN EXTRACT(MONTH FROM d) >= 2
        THEN EXTRACT(YEAR FROM d)::SMALLINT
        ELSE (EXTRACT(YEAR FROM d) - 1)::SMALLINT
    END                                                 AS fiscal_year,
    EXTRACT(ISODOW FROM d) IN (6, 7)                    AS is_weekend,
    FALSE                                                AS is_holiday
FROM generate_series('2025-03-01'::DATE, '2025-03-31'::DATE, '1 day'::INTERVAL) AS d
ON CONFLICT (date_key) DO NOTHING;

COMMIT;
