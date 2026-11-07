"""Usage: python run_benchmark.py [--engines pandas mapreduce spark] [--scales 10000 100000 1000000 10000000]
Generates the CSV scale files in data/ (first run only), runs each engine in its own process, checks that all engines
return identical results, and writes outputs/benchmark_results.csv. Engines that are unavailable are recorded as skipped."""
import argparse, platform, os
from pathlib import Path
import pandas as pd
from src.engines.benchmark import benchmark, SCALES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["pandas", "mapreduce", "spark"])
    ap.add_argument("--scales", nargs="+", type=int, default=SCALES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--big-repeats", type=int, default=3, help="repeats for scales >= 5M rows")
    a = ap.parse_args()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    df = benchmark(a.engines, a.scales, repeats=a.repeats, big_repeats=a.big_repeats)
    df.to_csv(out / "benchmark_results.csv", index=False)
    (out / "benchmark_machine.txt").write_text(f"python {platform.python_version()} | {platform.platform()} | cpus={os.cpu_count()}\n")
    print(df.to_string(index=False))
