"""Benchmark harness: generate scale files, run each engine in a subprocess, verify agreement, collect timings."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
from .common import same

SCALES = [10_000, 100_000, 1_000_000, 10_000_000]
ROOT = Path(__file__).resolve().parents[2]


def scale_file(n: int, data_dir: Path | str = ROOT / "data", seed: int = 42) -> Path:
    p = Path(data_dir) / f"ads_{n}.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        # generated in a child process so this process stays small (keeps per-engine memory readings clean)
        subprocess.run([sys.executable, "-m", "src.data_generation.simulator", str(p), str(n), str(seed)], cwd=ROOT, check=True)
    return p


def run_once(engine: str, path: Path, timeout: int = 3600) -> dict:
    proc = subprocess.run([sys.executable, "-m", "src.engines.cli", engine, str(path)], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"{engine} failed on {path.name}: {proc.stderr[-500:]}")


def benchmark(engines=("pandas", "mapreduce", "spark"), scales=SCALES, repeats: int = 3, big_repeats: int = 1,
              data_dir=ROOT / "data", log=print) -> pd.DataFrame:
    """Median wall time over `repeats` runs (big_repeats for scales >= 5M). Engines that cannot run are recorded as skipped."""
    rows = []
    for n in scales:
        path = scale_file(n, data_dir)
        reference = None
        for eng in engines:
            if eng == "spark":
                try:
                    import pyspark  # noqa: F401
                except ImportError:
                    rows.append(dict(rows=n, engine=eng, status="skipped: pyspark not installed")); continue
            reps = big_repeats if n >= 5_000_000 else repeats
            res = []
            try:
                for _ in range(reps):
                    res.append(run_once(eng, path))
            except Exception as e:      # record failures instead of crashing the whole benchmark
                rows.append(dict(rows=n, engine=eng, status=f"failed: {str(e)[:120]}")); continue
            cs = res[0]["checksum"]
            reference = reference or cs
            secs = sorted(r["seconds"] for r in res)
            rows.append(dict(rows=n, engine=eng, status="ok", runs=reps, seconds_median=secs[len(secs) // 2],
                             seconds_min=secs[0], seconds_max=secs[-1], peak_mb=max(r["peak_mb"] for r in res),
                             startup_seconds=res[0].get("startup_seconds"), agrees_with_first=same(cs, reference)))
            log(rows[-1])
    return pd.DataFrame(rows)
