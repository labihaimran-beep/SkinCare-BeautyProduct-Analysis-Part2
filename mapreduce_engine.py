"""A faithful MapReduce simulation in plain Python (standard library + multiprocessing).

  input splits  : the file is cut into byte ranges aligned to line boundaries (like HDFS blocks)
  map           : each mapper process parses its lines and emits (key, value) with key=(date, platform, campaign)
                  and value=(1, impressions, clicks, spend, conversions, revenue)
  combine       : optional in-mapper combiner sums values per key before the shuffle (cuts shuffle volume)
  shuffle       : pairs are partitioned to reducers by a STABLE hash of the key (crc32; Python's hash() is
                  randomised per process and would break partitioning)
  reduce        : each reducer process sums the values for its keys
This is a teaching simulation of the programming model on one machine, not Hadoop: no HDFS, no fault tolerance,
no cluster. Its speed reflects pure-Python parsing, which is exactly the point of the comparison."""
from __future__ import annotations
import os, zlib
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd
from .common import KEY, VALUES, USECOLS, tidy

N_FIELDS = 6


def input_splits(path, n_splits: int):
    """Return [(start, end)] byte ranges aligned to newlines, after the header line."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header_end = len(f.readline())
        bounds = [header_end]
        for i in range(1, n_splits):
            f.seek(max(header_end, size * i // n_splits))
            f.readline()                      # advance to the next line start
            bounds.append(min(f.tell(), size))
        bounds.append(size)
    bounds = sorted(set(bounds))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _partition(key, n_reducers: int) -> int:
    return zlib.crc32("|".join(key).encode()) % n_reducers


def _mapper(args):
    """Map one split. Returns {reducer_id: {key: value-list}} (combined) or {reducer_id: [(key, value), ...]}."""
    path, start, end, cols, n_reducers, combine = args
    ki = [cols[c] for c in KEY]; vi = [cols[c] for c in VALUES]
    out_c = [defaultdict(lambda: [0, 0, 0, 0.0, 0, 0.0]) for _ in range(n_reducers)] if combine else None
    out_p = [[] for _ in range(n_reducers)] if not combine else None
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8")
    for line in data.splitlines():
        if not line:
            continue
        p = line.split(",")
        key = (p[ki[0]], p[ki[1]], p[ki[2]])
        val = (1, int(p[vi[0]]), int(p[vi[1]]), float(p[vi[2]]), int(p[vi[3]]), float(p[vi[4]]))
        r = _partition(key, n_reducers)
        if combine:
            a = out_c[r][key]
            a[0] += 1; a[1] += val[1]; a[2] += val[2]; a[3] += val[3]; a[4] += val[4]; a[5] += val[5]
        else:
            out_p[r].append((key, val))
    if combine:
        return [dict(d) for d in out_c]
    return out_p


def _reducer(partition_chunks):
    """Reduce: merge everything that the shuffle routed to this reducer."""
    acc = defaultdict(lambda: [0, 0, 0, 0.0, 0, 0.0])
    for chunk in partition_chunks:
        items = chunk.items() if isinstance(chunk, dict) else chunk
        for key, v in items:
            a = acc[key]
            a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3] += v[3]; a[4] += v[4]; a[5] += v[5]
    return dict(acc)


def run(path, workers: int | None = None, n_reducers: int | None = None, combine: bool = True,
        splits_per_worker: int = 2) -> pd.DataFrame:
    workers = workers or os.cpu_count() or 1
    n_reducers = n_reducers or workers
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    cols = {c: i for i, c in enumerate(header)}
    missing = [c for c in USECOLS if c not in cols]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    splits = input_splits(path, max(workers * splits_per_worker, 1))
    jobs = [(str(path), a, b, cols, n_reducers, combine) for a, b in splits]
    if workers == 1:
        mapped = [_mapper(j) for j in jobs]
        reduced = [_reducer([m[r] for m in mapped]) for r in range(n_reducers)]
    else:
        with Pool(workers) as pool:
            mapped = pool.map(_mapper, jobs)                                  # MAP
            shuffled = [[m[r] for m in mapped] for r in range(n_reducers)]    # SHUFFLE (group by reducer)
            reduced = pool.map(_reducer, shuffled)                            # REDUCE
    rows = [list(k) + [v[0], v[1], v[2], v[3], v[4], v[5]] for part in reduced for k, v in part.items()]
    out = pd.DataFrame(rows, columns=KEY + ["rows"] + VALUES)
    return tidy(out)
