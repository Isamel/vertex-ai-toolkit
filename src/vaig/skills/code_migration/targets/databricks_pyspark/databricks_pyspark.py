"""Databricks PySpark 13.3 LTS target pack — scaffold."""
from vaig.skills.code_migration.targets.pack import CodeExample, TargetPack

__all__ = ["DATABRICKS_PYSPARK_PACK"]

DATABRICKS_PYSPARK_PACK = TargetPack(
    name="databricks-pyspark",
    version="13.3-lts",
    description=(
        "Databricks Runtime 13.3 LTS PySpark migration target. "
        "Scaffold — patterns to be expanded."
    ),
    imports=[
        "from pyspark.sql import SparkSession",
        "from pyspark.sql import functions as F",
        "from delta.tables import DeltaTable",
    ],
    patterns={
        "table_read": (
            "df = spark.read.format('delta').table('{catalog}.{schema}.{table}')"
        ),
        "table_write": (
            "df.write.format('delta').mode('overwrite')"
            ".saveAsTable('{catalog}.{schema}.{table}')"
        ),
    },
    forbidden_apis=[
        "pd.read_csv",
        "open(",
        "os.path",
        "local_path",
    ],
    required_boilerplate="spark = SparkSession.builder.getOrCreate()",
    examples=[
        CodeExample(
            description="Read Delta table and write back",
            source="df = pd.read_csv('data.csv')\ndf.to_parquet('output.parquet')",
            target=(
                "df = spark.read.format('delta').table('catalog.schema.table')\n"
                "df.write.format('delta').mode('overwrite')"
                ".saveAsTable('catalog.schema.output')"
            ),
        ),
    ],
)
