import pandas as pd, pytest
from src.data_generation.simulator import write_csv
from src.engines import pandas_engine, mapreduce_engine, spark_engine
from src.engines.common import checksum, same, KEY, VALUES
from src.engines.mapreduce_engine import input_splits, _partition


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "ads.csv"
    write_csv(p, 60_000, seed=3)
    return p


def test_mapreduce_matches_pandas_in_every_configuration(csv):
    ref = pandas_engine.run(csv)
    for kw in (dict(), dict(workers=1), dict(workers=1, combine=False), dict(workers=3, n_reducers=5), dict(workers=2, splits_per_worker=5)):
        out = mapreduce_engine.run(csv, **kw)
        assert same(checksum(ref), checksum(out)), kw
        pd.testing.assert_frame_equal(ref[KEY + ["rows"]], out[KEY + ["rows"]])
        for c in VALUES:
            assert (ref[c] - out[c]).abs().max() < 1e-6 * max(1, ref[c].abs().max())


def test_input_splits_cover_the_file_exactly_once_on_line_boundaries(csv):
    import os
    splits = input_splits(csv, 7)
    data = open(csv, "rb").read()
    assert splits[0][0] == len(data.split(b"\n")[0]) + 1 and splits[-1][1] == os.path.getsize(csv)
    assert all(a[1] == b[0] for a, b in zip(splits, splits[1:]))          # contiguous, no gaps or overlaps
    assert all(data[s - 1:s] == b"\n" for s, _ in splits)                 # every split starts at a line start


def test_partition_is_stable_and_in_range():
    k = ("2024-01-01", "Google", "Eid_Sale")
    assert _partition(k, 4) == _partition(k, 4) and 0 <= _partition(k, 4) < 4
    assert len({_partition(("d", "p", str(i)), 8) for i in range(200)}) == 8


def test_missing_columns_rejected(tmp_path):
    p = tmp_path / "bad.csv"; p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        mapreduce_engine.run(p, workers=1)


@pytest.mark.skipif(not spark_engine.available(), reason="pyspark not installed (optional)")
def test_spark_matches_pandas(csv):
    assert same(checksum(pandas_engine.run(csv)), checksum(spark_engine.run(csv)))
