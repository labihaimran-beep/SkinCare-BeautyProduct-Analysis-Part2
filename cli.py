"""Run ONE engine on ONE file in its own process and print a JSON line (time, peak memory, checksum).
Separate processes give clean timings and per-engine memory peaks.  usage: python -m src.engines.cli ENGINE FILE"""
from __future__ import annotations
import json, resource, sys, time
from .common import checksum


def _hwm_kb() -> int:
    """This process's own peak RSS (VmHWM). ru_maxrss would also carry over the launcher's peak across fork/exec."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _peak_mb() -> float:
    """Peak RSS of the largest single process of the run (main process or, for MapReduce, its worker processes)."""
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(_hwm_kb(), kids) / 1024, 1)


def main(engine: str, path: str):
    info = {"engine": engine, "file": path}
    if engine == "pandas":
        from . import pandas_engine as e
        t = time.perf_counter(); cube = e.run(path); info["seconds"] = time.perf_counter() - t
    elif engine == "mapreduce":
        from . import mapreduce_engine as e
        t = time.perf_counter(); cube = e.run(path); info["seconds"] = time.perf_counter() - t
    elif engine == "spark":
        from . import spark_engine as e
        t0 = time.perf_counter(); spark = e.get_session(); info["startup_seconds"] = time.perf_counter() - t0
        t = time.perf_counter(); cube = e.run(path, spark); info["seconds"] = time.perf_counter() - t
    else:
        raise SystemExit(f"unknown engine {engine}")
    info.update(peak_mb=_peak_mb(), checksum=checksum(cube))
    print("RESULT " + json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
