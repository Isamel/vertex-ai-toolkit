"""AWS Glue PySpark 4.0 knowledge pack."""
from vaig.skills.code_migration.targets.pack import CodeExample, TargetPack

__all__ = ["AWS_GLUE_PYSPARK_PACK"]

AWS_GLUE_PYSPARK_PACK = TargetPack(
    name="aws-glue-pyspark",
    version="4.0",
    description=(
        "AWS Glue 4.0 PySpark migration target. "
        "Converts ETL pipelines to Glue DynamicFrames with GlueContext."
    ),
    imports=[
        "from awsglue.context import GlueContext",
        "from awsglue.dynamicframe import DynamicFrame",
        "from pyspark.context import SparkContext",
        "import boto3",
    ],
    patterns={
        # Pentaho Table Input → Glue catalog read
        "table_input": (
            "datasource = glueContext.create_dynamic_frame.from_catalog("
            "database='{database}', table_name='{table_name}', "
            "transformation_ctx='datasource')"
        ),
        # Pentaho Table Output → Glue catalog write
        "table_output": (
            "glueContext.write_dynamic_frame.from_catalog("
            "frame=frame, database='{database}', table_name='{table_name}', "
            "transformation_ctx='sink')"
        ),
        # Pentaho Select Values / Mapping → ApplyMapping
        "select_values": (
            "mapped = ApplyMapping.apply("
            "frame=frame, mappings=[{mappings}], "
            "transformation_ctx='mapped')"
        ),
        # Pentaho Filter Rows → Filter
        "filter_rows": (
            "filtered = Filter.apply("
            "frame=frame, f=lambda x: {condition}, "
            "transformation_ctx='filtered')"
        ),
        # Pentaho S3 file input → S3 read
        "s3_input": (
            "datasource = glueContext.create_dynamic_frame.from_options("
            "connection_type='s3', connection_options={{'paths': ['{s3_path}']}},"
            " format='{format}', transformation_ctx='datasource')"
        ),
        # Pentaho S3 file output → S3 write
        "s3_output": (
            "glueContext.write_dynamic_frame.from_options("
            "frame=frame, connection_type='s3', "
            "connection_options={{'path': '{s3_path}'}}, "
            "format='{format}', transformation_ctx='sink')"
        ),
        # Pandas DataFrame → DynamicFrame conversion
        "pandas_to_dynamic_frame": (
            "spark_df = spark.createDataFrame(pandas_df)\n"
            "dynamic_frame = DynamicFrame.fromDF(spark_df, glueContext, 'convert')"
        ),
    },
    forbidden_apis=[
        "pd.read_csv",
        "pd.DataFrame",
        "open(",
        "os.path",
        "local_path",
    ],
    required_boilerplate=(
        "sc = SparkContext()\n"
        "glueContext = GlueContext(sc)\n"
        "spark = glueContext.spark_session\n"
        "job = Job(glueContext)\n"
        "job.init(args['JOB_NAME'], args)"
    ),
    examples=[
        CodeExample(
            description="Read from Glue Data Catalog and write to S3 as Parquet",
            source=(
                "# Pentaho: Table Input → Table Output\n"
                "df = pd.read_sql('SELECT * FROM sales.orders', conn)\n"
                "df.to_parquet('s3://bucket/output/orders.parquet')"
            ),
            target=(
                "from awsglue.context import GlueContext\n"
                "from pyspark.context import SparkContext\n\n"
                "sc = SparkContext()\n"
                "glueContext = GlueContext(sc)\n"
                "spark = glueContext.spark_session\n\n"
                "datasource = glueContext.create_dynamic_frame.from_catalog(\n"
                "    database='sales', table_name='orders',\n"
                "    transformation_ctx='datasource'\n"
                ")\n"
                "glueContext.write_dynamic_frame.from_options(\n"
                "    frame=datasource, connection_type='s3',\n"
                "    connection_options={'path': 's3://bucket/output/orders/'},\n"
                "    format='parquet', transformation_ctx='sink'\n"
                ")"
            ),
        ),
        CodeExample(
            description="Filter rows and apply column mapping",
            source=(
                "# Pentaho: Filter Rows + Select Values\n"
                "df = df[df['status'] == 'active']\n"
                "df = df.rename(columns={'cust_id': 'customer_id', 'amt': 'amount'})"
            ),
            target=(
                "from awsglue.transforms import Filter, ApplyMapping\n\n"
                "filtered = Filter.apply(\n"
                "    frame=datasource,\n"
                "    f=lambda x: x['status'] == 'active',\n"
                "    transformation_ctx='filtered'\n"
                ")\n"
                "mapped = ApplyMapping.apply(\n"
                "    frame=filtered,\n"
                "    mappings=[\n"
                "        ('cust_id', 'string', 'customer_id', 'string'),\n"
                "        ('amt', 'double', 'amount', 'double'),\n"
                "    ],\n"
                "    transformation_ctx='mapped'\n"
                ")"
            ),
        ),
    ],
)
