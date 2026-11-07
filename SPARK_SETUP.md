# Running the Spark engine (optional)

1. Install Java (8, 11, 17 or 21) and check `java -version`.
2. `pip install -r requirements-optional.txt` (installs `pyspark`).
3. `python run_benchmark.py --engines pandas mapreduce spark`

Windows: if Spark cannot find Hadoop winutils you may see warnings; they are usually harmless for local CSV reads.
Spark's JVM start-up time is reported separately (`startup_seconds`) and is not included in `seconds_median`.
Expect Spark to be slower than pandas on small files (start-up and scheduling overhead) and to pull ahead only when the
data no longer fits comfortably in one machine's memory or when it runs on several machines. Report whatever you measure;
this repository ships no Spark timings because they could not be measured in the development sandbox.
`tests/test_engines.py::test_spark_matches_pandas` runs automatically when pyspark is installed.
