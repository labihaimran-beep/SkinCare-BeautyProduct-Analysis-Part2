import numpy as np, pandas as pd, pytest
from src.data_generation.simulator import generate, TRUE_ELASTICITY
from src.engines.pandas_engine import aggregate
from src.recommendation.response_curves import fit_curve, Curve
from src.recommendation.optimizer import recommend, allocate, bounds, format_split, total_revenue, sensitivity
from src.recommendation.validation import bootstrap_shares, backtest, ground_truth_check


@pytest.fixture(scope="module")
def cube():
    return aggregate(generate(300_000, seed=11))


@pytest.fixture(scope="module")
def rec(cube):
    return recommend(cube, 1_000_000)


def test_fit_recovers_known_elasticity():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(8, 0.5, 400)); r = 3 * s ** 0.6 * np.exp(rng.normal(0, 0.05, 400))
    c = fit_curve("X", s, r)
    assert c.b == pytest.approx(0.6, abs=0.03) and c.a == pytest.approx(3, rel=0.25) and c.r2 > 0.9


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_curve("X", np.full(50, 100.0), np.full(50, 5.0))


def test_allocation_sums_to_budget_and_respects_bounds(rec):
    lo, hi = rec["bounds"]; a = rec["allocations"]["recommended"]
    assert sum(a.values()) == pytest.approx(rec["budget"])
    assert all(lo[p] - 1e-6 <= a[p] <= hi[p] + 1e-6 for p in a)
    assert rec["table"].recommended_share.sum() == pytest.approx(1)


def test_optimised_beats_every_strategy_that_respects_the_same_constraints(rec):
    c = rec["comparison"].set_index("strategy")
    best = c.loc["Optimised (recommended)", "expected_revenue"]
    assert c.loc["Optimised (recommended)", "within_constraints"]
    assert (c[c.within_constraints].expected_revenue <= best + 1e-6).all()
    assert c.loc["Current mix", "within_constraints"] and best > c.loc["Current mix", "expected_revenue"]


def test_unconstrained_optimum_equalises_marginal_returns():
    curves = {"A": Curve("A", 5.0, 0.6, .6, 0, 1, 10, 1, 1, .1, 9, False), "B": Curve("B", 2.0, 0.8, .8, 0, 1, 10, 1, 1, .1, 9, False),
              "C": Curve("C", 3.0, 0.5, .5, 0, 1, 10, 1, 1, .1, 9, False)}
    B = 300_000.0
    lo = {p: 0.0 for p in curves}; hi = {p: B for p in curves}
    a = allocate(curves, B, lo, hi, steps=3000)
    m = [curves[p].marginal(a[p] / 30) for p in curves]
    assert max(m) / min(m) < 1.02                                         # KKT: equal marginal revenue per rupee
    equal = {p: B / 3 for p in curves}
    assert total_revenue(curves, a) > total_revenue(curves, equal)


def test_infeasible_constraints_raise():
    with pytest.raises(ValueError):
        bounds({"A": 0.5, "B": 0.5}, 100.0, min_share=0.6)
    with pytest.raises(ValueError):
        bounds({"A": 0.5, "B": 0.5}, 100.0, max_share=0.4)


def test_larger_budget_is_allocated_in_full(cube):
    b = recommend(cube, 5_000_000, baseline_monthly_budget=1_000_000)
    assert b["table"].recommended_budget.sum() == pytest.approx(5_000_000)


def test_fitted_elasticities_close_to_truth(rec):
    for p, c in rec["curves"].items():
        assert abs(c.b_raw - TRUE_ELASTICITY[p]) < 0.05


def test_ground_truth_check_recommendation_captures_most_of_oracle_gain(rec):
    g = ground_truth_check(rec, TRUE_ELASTICITY)
    assert g["uplift_recommended_pct"] > 0 and g["capture_ratio"] > 0.9


def test_bootstrap_and_backtest(rec, cube):
    b = bootstrap_shares(rec, n_boot=20, seed=1)
    assert (b.p05 <= b["median"] + 1e-9).all() and (b["median"] <= b.p95 + 1e-9).all()
    bt = backtest(cube)["table"]
    assert (bt.r2_log_out_of_sample > 0.5).all() and (bt.mape_pct < 25).all()


def test_format_split_sums_to_100(rec):
    lines = format_split(rec["table"]).splitlines()
    assert sum(float(l.split(": ")[1].rstrip("%")) for l in lines) == pytest.approx(100)


def test_sensitivity_returns_one_row_per_setting(cube):
    s = sensitivity(cube, settings=[("a", dict()), ("b", dict(max_share=1.0))])
    assert len(s) == 2 and (s.drop(columns=["setting", "expected_revenue"]).sum(axis=1).round(6) == 1).all()
