"""Apache Spark (PySpark) version of the same job. Optional: needs `pip install pyspark` and Java 8/11/17/21.
Not executable in the development sandbox (PyPI unreachable), so it is covered by a test that is skipped
automatically when pyspark is missing. Run `python run_benchmark.py --engines spark` on your own machine."""
from __future__ import annotations
import pandas as pd
from .common import KEY, VALUES, USECOLS, tidy

SCHEMA = ("date string, campaign string, platform string, impressions long, clicks long, spend double, "
          "conversions long, revenue double, age string, gender string, city string, device string")


def available() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


def get_session(master: str = "local[*]", memory: str = "4g"):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master(master).appName("ad-budget-optimizer")
            .config("spark.driver.memory", memory).config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "16").getOrCreate())


def run(path, spark=None) -> pd.DataFrame:
    from pyspark.sql import functions as F
    spark = spark or get_session()
    df = spark.read.csv(str(path), header=True, schema=SCHEMA).select(*USECOLS)
    agg = df.groupBy(*KEY).agg(F.count(F.lit(1)).alias("rows"), *[F.sum(c).alias(c) for c in VALUES])
    rows = agg.collect()
    out = pd.DataFrame([r.asDict() for r in rows])
    return tidy(out)
