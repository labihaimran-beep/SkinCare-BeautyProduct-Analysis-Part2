"""Validation of the recommendation engine: uncertainty (bootstrap), out-of-time backtest, and (because the data are
simulated) a check against the simulator's known ground truth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .response_curves import Curve, fit_all, fit_curve, daily_platform_table, scale_factor
from .optimizer import bounds, allocate, total_revenue, monthly_revenue, HORIZON_DAYS


def bootstrap_shares(rec: dict, n_boot: int = 200, seed: int = 0, steps: int = 400) -> pd.DataFrame:
    """Resample DAYS with replacement, refit curves, re-optimise. Returns 5th/50th/95th percentile shares per platform."""
    daily, budget = rec["daily"], rec["budget"]
    dates = np.array(sorted(daily["date"].unique()))
    rng = np.random.default_rng(seed)
    by_date = {d: g for d, g in daily.groupby("date")}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(dates, len(dates), replace=True)
        boot = pd.concat([by_date[d] for d in pick], ignore_index=True)
        try:
            curves = fit_all(boot, rec["factor"])
            sp = boot.groupby("platform")["spend"].sum()
            cur = (sp / sp.sum()).to_dict()
            lo, hi = bounds(cur, budget, **{"min_share": rec["params"]["min_share"], "max_share": rec["params"]["max_share"],
                                            "trust": rec["params"]["trust"]})
            a = allocate(curves, budget, lo, hi, steps, rec["horizon_days"])
        except ValueError:
            continue
        out.append({p: v / budget for p, v in a.items()})
    if not out:
        raise ValueError("Bootstrap produced no valid replicates.")
    b = pd.DataFrame(out)
    return pd.DataFrame({"p05": b.quantile(0.05), "median": b.quantile(0.5), "p95": b.quantile(0.95),
                         "replicates": len(b)}).reset_index(names="platform")


def backtest(cube: pd.DataFrame, train_frac: float = 0.75, baseline: float = 1_000_000) -> dict:
    """Fit on the first train_frac of days and predict the later days' platform revenue from their spend."""
    daily = daily_platform_table(cube)
    dates = sorted(daily["date"].unique())
    cut = dates[int(len(dates) * train_frac)]
    train, test = daily[daily["date"] < cut], daily[daily["date"] >= cut]
    f = scale_factor(train, baseline)
    curves = fit_all(train, f)
    rows = []
    for p, c in curves.items():
        t = test[test.platform == p]
        s, r = t["spend"].to_numpy(float) * f, t["revenue"].to_numpy(float) * f
        pred = c.revenue(s)
        ok = r > 0
        lr, lp = np.log(r[ok]), np.log(pred[ok])
        ss = ((lr - lr.mean()) ** 2).sum()
        rows.append(dict(platform=p, train_days=c.n_days, test_days=int(ok.sum()),
                         r2_log_out_of_sample=float(1 - ((lr - lp) ** 2).sum() / ss) if ss > 0 else np.nan,
                         mape_pct=float(100 * np.mean(np.abs(pred[ok] - r[ok]) / r[ok]))))
    return dict(table=pd.DataFrame(rows), cutoff=cut)


def ground_truth_check(rec: dict, true_b: dict) -> dict:
    """SIMULATION ONLY. Re-run the optimisation with the simulator's TRUE elasticities (same level calibration) and
    score the recommended allocation, the current mix and the oracle allocation under those true curves."""
    daily, budget, f = rec["daily"], rec["budget"], rec["factor"]
    true_curves = {}
    for p, c in rec["curves"].items():
        g = daily[daily.platform == p]
        s, r = g["spend"].to_numpy(float) * f, g["revenue"].to_numpy(float) * f
        b = true_b[p]
        a = float(np.mean(r) / np.mean(np.power(s, b)))
        true_curves[p] = Curve(p, a, b, b, 0.0, 1.0, len(g), float(s.mean()), float(r.mean()), float(s.min()), float(s.max()), False)
    lo, hi = rec["bounds"]
    oracle = allocate(true_curves, budget, lo, hi, 1000, rec["horizon_days"])
    h = rec["horizon_days"]
    cur, recm = rec["allocations"]["current"], rec["allocations"]["recommended"]
    rv = {k: total_revenue(true_curves, v, h) for k, v in (("current", cur), ("recommended", recm), ("oracle", oracle))}
    gap = rv["oracle"] - rv["current"]
    elast = pd.DataFrame([dict(platform=p, fitted_b=rec["curves"][p].b_raw, true_b=true_b[p],
                               abs_error=abs(rec["curves"][p].b_raw - true_b[p]), b_se=rec["curves"][p].b_se) for p in true_b])
    shares = pd.DataFrame([dict(platform=p, recommended=recm[p] / budget, oracle=oracle[p] / budget,
                                abs_diff_pts=100 * abs(recm[p] - oracle[p]) / budget) for p in true_b])
    return dict(revenue_under_true_curves=rv, uplift_recommended_pct=100 * (rv["recommended"] / rv["current"] - 1),
                uplift_oracle_pct=100 * (rv["oracle"] / rv["current"] - 1),
                capture_ratio=(rv["recommended"] - rv["current"]) / gap if gap > 1e-9 else float("nan"),
                elasticities=elast, shares=shares)
