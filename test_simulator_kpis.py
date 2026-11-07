import numpy as np, pandas as pd, pytest
from src.data_generation.simulator import generate, COLUMNS, PLATFORMS, TRUE_ELASTICITY, PLATFORM_PARAMS
from src.metrics.kpis import add_kpis, rollup


@pytest.fixture(scope="module")
def df():
    return generate(150_000, seed=7)


def test_schema_and_invariants(df):
    assert list(df.columns) == COLUMNS and len(df) == 150_000
    assert df.isna().sum().sum() == 0
    assert (df.clicks <= df.impressions).all() and (df.conversions <= df.clicks).all()
    assert (df[["impressions", "clicks", "spend", "conversions", "revenue"]] >= 0).all().all()
    assert set(df.platform) == set(PLATFORMS)
    assert (df.loc[df.conversions == 0, "revenue"] == 0).all()


def test_deterministic_and_prefix_stable(df):
    assert generate(150_000, seed=7).equals(df)
    assert generate(10_000, seed=7).equals(df.iloc[:10_000])          # 10k is the first 10k of any larger set
    assert not generate(10_000, seed=8).equals(df.iloc[:10_000])


def test_true_elasticities_are_one_minus_gamma():
    for p, (_, _, _, _, gamma) in PLATFORM_PARAMS.items():
        assert TRUE_ELASTICITY[p] == pytest.approx(1 - gamma)


def test_kpi_formulas():
    d = pd.DataFrame({"impressions": [1000, 0], "clicks": [50, 0], "spend": [500.0, 0.0], "conversions": [5, 0], "revenue": [2500.0, 0.0]})
    k = add_kpis(d).iloc[0]
    assert k.ctr == 0.05 and k.cpc == 10 and k.cpm == 500 and k.cpa == 100 and k.conversion_rate == 0.1
    assert k.roas == 5 and k.profit == 2000
    z = add_kpis(d).iloc[1]
    assert np.isnan(z.ctr) and np.isnan(z.cpa) and np.isnan(z.roas)       # no inf / divide-by-zero


def test_rollup_uses_sums_not_mean_of_ratios():
    c = pd.DataFrame({"g": ["a", "a"], "impressions": [100, 900], "clicks": [50, 90], "spend": [10.0, 90.0],
                      "conversions": [5, 9], "revenue": [100.0, 100.0], "rows": [1, 1]})
    r = rollup(c, "g").iloc[0]
    assert r.ctr == pytest.approx(140 / 1000) and r.roas == pytest.approx(2.0)
