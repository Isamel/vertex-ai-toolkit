# Pentaho ETL — Sample Project for Migration Testing

This is a realistic Pentaho Data Integration (PDI/Kettle) project designed to
test VAIG's ability to migrate ETL pipelines to **AWS Glue (PySpark)**.

## Business Scenario

A retail company processes daily sales data from multiple sources:

1. **CSV files** from POS systems (point of sale)
2. **Database tables** from the inventory management system (PostgreSQL)
3. **API responses** stored as JSON files from the e-commerce platform

The ETL pipeline:
- Ingests raw data from all three sources
- Cleans and validates records (null handling, type casting, deduplication)
- Enriches data with dimension lookups (product categories, store regions)
- Calculates aggregated metrics (daily sales by store, product performance)
- Loads results into a star-schema data warehouse

## Pentaho Operations Covered

| # | Operation                  | Pentaho Step                  | File                                   |
|---|---------------------------|-------------------------------|----------------------------------------|
| 1 | CSV Input                 | CSV file input                | `transformations/ingest_pos_sales.ktr` |
| 2 | DB Connection             | Table input                   | `transformations/ingest_inventory.ktr` |
| 3 | JSON Input                | JSON input                    | `transformations/ingest_ecommerce.ktr` |
| 4 | Filter Rows               | Filter rows                   | `transformations/clean_sales.ktr`      |
| 5 | Replace/Regex             | Replace in string / Regex     | `transformations/clean_sales.ktr`      |
| 6 | Select Values             | Select values (rename/remove) | `transformations/clean_sales.ktr`      |
| 7 | Sort Rows                 | Sort rows                     | `transformations/clean_sales.ktr`      |
| 8 | Unique Rows               | Unique rows (deduplication)   | `transformations/clean_sales.ktr`      |
| 9 | Database Lookup           | Database lookup               | `transformations/enrich_dimensions.ktr`|
| 10| Stream Lookup             | Stream lookup                 | `transformations/enrich_dimensions.ktr`|
| 11| Calculator                | Calculator / Formula          | `transformations/calc_metrics.ktr`     |
| 12| Group By                  | Group by (aggregation)        | `transformations/calc_metrics.ktr`     |
| 13| Row Denormaliser          | Row denormaliser (pivot)      | `transformations/calc_metrics.ktr`     |
| 14| Merge Join                | Merge join                    | `transformations/merge_sources.ktr`    |
| 15| Switch/Case               | Switch/Case                   | `transformations/route_output.ktr`     |
| 16| Table Output              | Table output / Insert/Update  | `transformations/load_warehouse.ktr`   |
| 17| Set Variables             | Set variables                 | `transformations/load_warehouse.ktr`   |
| 18| JavaScript                | Modified JavaScript Value     | `transformations/custom_logic.ktr`     |
| 19| Error Handling            | Error handling (step)         | `transformations/clean_sales.ktr`      |
| 20| Job orchestration         | START → Trans → Mail → Success| `jobs/daily_sales_job.kjb`             |
| 21| Job conditional logic     | Evaluate / Simple eval        | `jobs/daily_sales_job.kjb`             |
| 22| Shell script execution    | Shell step                    | `jobs/daily_sales_job.kjb`             |
| 23| DB Procedure Call         | Call DB procedure             | `transformations/load_warehouse.ktr`   |
| 24| Value Mapper              | Value mapper                  | `transformations/enrich_dimensions.ktr`|
| 25| Add Sequence              | Add sequence                  | `transformations/load_warehouse.ktr`   |

## Directory Structure

```
pentaho-etl/
├── README.md
├── config/
│   ├── db_connections.xml        # Shared DB connection definitions
│   └── kettle.properties         # Environment variables / parameters
├── transformations/
│   ├── ingest_pos_sales.ktr      # CSV input from POS
│   ├── ingest_inventory.ktr      # DB table input from PostgreSQL
│   ├── ingest_ecommerce.ktr      # JSON input from e-commerce API
│   ├── clean_sales.ktr           # Data quality: filter, dedupe, null handling
│   ├── enrich_dimensions.ktr     # Lookup product categories, store regions
│   ├── calc_metrics.ktr          # Aggregations, formulas, pivoting
│   ├── merge_sources.ktr         # Merge join all three data sources
│   ├── route_output.ktr          # Switch/Case routing by store region
│   ├── load_warehouse.ktr        # Insert/Update into star schema
│   └── custom_logic.ktr          # JavaScript step for complex business rules
├── jobs/
│   └── daily_sales_job.kjb       # Master orchestration job
├── sql/
│   ├── create_staging.sql        # Staging tables DDL
│   ├── create_warehouse.sql      # Star schema DDL (facts + dimensions)
│   └── stored_procedures.sql     # Post-load procedures
├── data/
│   ├── input/
│   │   ├── pos_sales_sample.csv  # Sample POS data
│   │   └── ecommerce_orders.json # Sample e-commerce data
│   └── output/                   # (generated at runtime)
└── scripts/
    └── run_daily_etl.sh          # Shell wrapper for scheduled execution
```

## Migration Target: AWS Glue (PySpark)

When migrating with VAIG, the expected output should be:

1. **Glue Jobs** — PySpark scripts for each transformation
2. **Glue Crawlers** — For source data discovery
3. **Glue Connections** — For database sources
4. **Step Functions** — For job orchestration (replacing Pentaho jobs)
5. **CloudFormation/CDK** — Infrastructure as code
